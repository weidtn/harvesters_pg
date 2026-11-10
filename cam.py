"""
The model to handle interaction with the camera.
"""

from PyQt6.QtWidgets import QProgressBar
from harvesters.core import Harvester, ImageAcquirer
from cv2 import VideoCapture
import cv2
import math
import numpy as np
import os
from pathlib import Path
import glob
import json
import logging
import re
from tqdm import trange
import time
import threading
from dataclasses import dataclass
from typing import Callable

from camera_profiles import detect_profile, max_frame_rate

# Serializes all access to the GenICam device (fetch + node_map reads/writes).
# The worker thread fetches frames while the GUI thread polls/writes node values;
# GenICam producers are not thread-safe, so concurrent device access from two
# threads can corrupt the producer's state and hang the machine.
IA_LOCK = threading.RLock()

# Upper bound, in seconds, for every buffer fetch in this module.
#
# ImageAcquirer.fetch() defaults to timeout=0, which switches its timeout check
# off entirely and spins in `while not buffer` forever (harvesters.core
# ImageAcquirer._fetch). A camera that never delivers - trigger mode still
# enabled, a GigE packet-size/MTU mismatch, a device left wedged by an earlier
# crash - then blocks the calling thread with no way to cancel: a permanent GUI
# freeze when the caller is the GUI thread, and a permanently held IA_LOCK when
# it is the worker. Every fetch below goes through try_fetch(timeout=...),
# which gives up and returns None instead of hanging.
FETCH_TIMEOUT_S = 5.0

# Longest single wait inside the device. Waits for a frame are split into slices
# of this length, with IA_LOCK released between them, so a stop request or a
# GUI-thread node write never waits for a whole FETCH_TIMEOUT_S - even when
# the camera has stopped delivering.
FETCH_SLICE_S = 0.1

# try_fetch(timeout=0) switches the timeout off and waits forever (harvesters
# only watches the clock for timeout > 0), so a computed wait is never allowed
# to reach zero.
_MIN_FETCH_S = 0.001

# Pause between two empty fetch slices, with IA_LOCK released, so a thread
# waiting for the lock gets it (see get_harvester_image).
_LOCK_YIELD_S = 0.001


class RecordingInterrupted(TimeoutError):
    """The camera stopped delivering in the middle of a recording.

    ``frames`` holds the frames recorded up to that point, so a long recording
    is not thrown away because its tail is missing.
    """

    def __init__(self, message: str, frames: np.ndarray):
        super().__init__(message)
        self.frames = frames


# One Harvester - and therefore one load of each GenTL producer - for the whole
# process. Building a fresh Harvester per connect meant unloading and reloading
# the producer library, plus a full device re-enumeration, on every
# connect/disconnect cycle. That per-cycle control-plane churn is what the
# Active Silicon FireBird's fatal-PCIe defect correlates with (raw-PHX
# workloads never trigger it, GenTL producer churn does), and several vendor
# producers are not safe to unload and reload in-process at all. Reset it only
# at application exit, via release_harvester(). Note that Harvester.update()
# also unloads and reloads every producer (and destroys every ImageAcquirer it
# created), so it runs only for an explicit scan and when a connect cannot be
# served from the cached device list.
_SHARED_HARVESTER: Harvester | None = None
_SHARED_HARVESTER_CTIS: tuple[str, ...] = ()


def get_harvester(ctis: list[str] | tuple[str, ...]) -> Harvester:
    """Return the process-wide Harvester, loading *ctis* on first use.

    The Harvester is rebuilt only when the set of producers changes (a new
    ``.cti`` was picked), because add_file() cannot be undone selectively.

    Args:
        ctis (list[str] | tuple[str, ...]): Producer paths, duplicates ignored.

    Returns:
        Harvester: The shared Harvester with those producers loaded.
    """
    global _SHARED_HARVESTER, _SHARED_HARVESTER_CTIS
    wanted = tuple(dict.fromkeys(ctis))
    if _SHARED_HARVESTER is not None and wanted != _SHARED_HARVESTER_CTIS:
        logging.info("Producer set changed; rebuilding the shared harvester")
        release_harvester()
    if _SHARED_HARVESTER is None:
        logging.info(f"Loading GenTL producers: {list(wanted)}")
        harvester = Harvester()
        for path in wanted:
            harvester.add_file(path)
        _SHARED_HARVESTER = harvester
        _SHARED_HARVESTER_CTIS = wanted
    return _SHARED_HARVESTER


def release_harvester() -> None:
    """Reset and drop the process-wide Harvester, unloading its producers.

    Call at application exit only - see MainWindow.closeEvent. Doing this per
    disconnect is the churn get_harvester exists to avoid.
    """
    teardown_cam(_SHARED_HARVESTER, None)


def abandon_harvester() -> None:
    """Drop the process-wide Harvester WITHOUT resetting it.

    For application exit while a capture worker is still stuck inside the
    producer: reset() would destroy the ImageAcquirer and unload the producer
    under that call, which is the use-after-free release_harvester() must never
    cause. harvesters also registers its own atexit finalizers that would do the
    same reset at interpreter exit, so those are detached here and the OS
    reclaims the device when the process ends.
    """
    global _SHARED_HARVESTER, _SHARED_HARVESTER_CTIS
    harvester = _SHARED_HARVESTER
    _SHARED_HARVESTER = None
    _SHARED_HARVESTER_CTIS = ()
    if harvester is None:
        return
    logging.warning(
        "Leaving the GenTL producers loaded at exit: a capture worker is still "
        "inside the producer, and unloading it under that call can crash."
    )
    for obj in (*getattr(harvester, "_ias", ()), harvester):
        finalizer = getattr(obj, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()


def _forget_shared_harvester(harvester: Harvester) -> None:
    """Drop the module reference if *harvester* is the shared one.

    Stops get_harvester() from later handing out an already-reset object, which
    is what would happen when a standalone script tears its camera down with
    teardown_cam(h, ia).
    """
    global _SHARED_HARVESTER, _SHARED_HARVESTER_CTIS
    if harvester is _SHARED_HARVESTER:
        _SHARED_HARVESTER = None
        _SHARED_HARVESTER_CTIS = ()


def _find_device(harvester: Harvester, search_key):
    """Return the cached DeviceInfo that *search_key* selects, or None.

    Used to skip h.update(): a re-enumeration unloads and reloads every
    producer and puts discovery traffic on every interface, and it used to run
    on every single connect even though the startup scan had just enumerated
    the very same devices. A dict key resolves to the FIRST match: when two
    producers (e.g. classic Vimba and Vimba X) enumerate the same camera, both
    entries lead to that camera, and harvesters' own create(dict) would refuse
    with "multiple devices found".
    """
    devices = list(harvester.device_info_list or [])
    if not devices:
        return None
    if search_key is None:
        return devices[0]
    if isinstance(search_key, int):
        return devices[search_key] if -len(devices) <= search_key < len(devices) else None
    if isinstance(search_key, dict):
        for device in devices:
            if all(
                value == getattr(device, key, None)
                for key, value in search_key.items()
            ):
                return device
    return None


# Well-known GenTL producer install locations of the common SDKs (checked in
# addition to GENICAM_GENTL64_PATH, e.g. for a tarball that was never
# registered): Vimba X 2024+ and classic Vimba layouts.
_WELL_KNOWN_GENTL_DIRS = [
    "/opt/alliedvision/VimbaX*/cti",
    "/opt/alliedvision/VimbaX*/GenTL/*/cti",
    "/opt/Vimba*/VimbaGigETL/cti",
    "/opt/Vimba*/VimbaUSBTL/cti",
]


def load_cti_envvar() -> list[str]:
    """Find the common *.cti files for the camera.

    Scans every directory listed in GENICAM_GENTL64_PATH (colon-separated,
    like PATH -- Allied Vision's Install_GenTL_Path.sh produces exactly this
    format), then the well-known SDK install locations. Entries pointing at
    the same file are reported once.

    Returns:
        list[str]: A list of paths to the *.cti files.
    """
    logging.info("Looking for *.cti files...")
    search_dirs = []
    env_var = os.getenv("GENICAM_GENTL64_PATH")
    if env_var:
        search_dirs.extend(d for d in env_var.split(os.pathsep) if d)
    for pattern in _WELL_KNOWN_GENTL_DIRS:
        search_dirs.extend(sorted(glob.glob(pattern)))
    ctis = []
    seen = set()
    for directory in search_dirs:
        for cti in sorted(glob.glob(os.path.join(directory, "*.cti"))):
            real = os.path.realpath(cti)
            if real not in seen:
                seen.add(real)
                ctis.append(cti)
    logging.info(f"Found *.cti files: {ctis}")
    return ctis


def setup_cam(
    cti: str | list[str],
    height: int | None = None,
    width: int | None = None,
    fps: int = 30,
    blacklevel: int = 125,
    exposure: int = 999,
    search_key: int | dict | None = None,
    start: bool = True,
    configure: bool = True,
    saved_aoi: bool = False,
) -> tuple[Harvester, ImageAcquirer]:
    """Set the camera up using the Harvester API.

    A frame grabber takes the camera's image size over once, when the device
    is opened (the Active Silicon CoaXPress grabber latched Width/Height and
    the stream payload size there and ignored every later camera write).
    Whatever this function writes to the camera's size therefore needs one
    more destroy + create before the first start: when Width, Height or
    PayloadSize differ from what the camera reported at open, the device is
    re-opened exactly once (see _reopen) and checked (see check_start).
    Nothing is ever started with a grabber that disagrees with the camera.

    Args:
        cti (str | list[str]): Path(s) to the GenTL Producer file(s).
        height (int): The height of the image.
        width (int): The width of the image.
        fps (int): The frames per second.
        blacklevel (int): The black level of the image.
        exposure (int): The exposure time of the image.
        search_key (int | dict | None): Which enumerated device to open - a
            list index, or a device-info filter such as
            ``{"display_name": ...}``. Defaults to the first device.
        start (bool): Start acquisition and collect a few warm-up frames.
            The GUI passes False: it starts acquisition itself when live view
            or a recording begins, and a start/warm-up/kill-stop cycle per
            connect is pure GenTL churn on the frame grabber.
        configure (bool): Write the connect defaults (frame rate, exposure,
            full-sensor AOI, TriggerMode Off, info fields, black level).
            False only opens the device and keeps every camera setting as it
            is (aoi_probe.py opens the camera that way).
        saved_aoi (bool): Write the AOI saved for this camera by an earlier
            session (see load_resolution_config) after the defaults.

    Returns:
        tuple[Harvester, ImageAcquirer]: The Harvester object and the ImageAcquirer object.

    Raises:
        StartRefused: if the re-opened grabber still disagrees with the camera.
        Exception: whatever the producer raised. The camera is always released
            before the exception leaves this function.
    """
    ctis = [cti] if isinstance(cti, str) else list(cti)
    # The process-wide Harvester, with the producers already loaded.
    h = get_harvester(ctis)
    ia = None
    try:
        ia = _open_device(h, search_key, rescan=True)
        opened = camera_size(ia)
        if configure:
            _configure_cam(ia, height, width, fps, blacklevel, exposure)
        if saved_aoi:
            _reapply_saved_aoi(ia)
        now = camera_size(ia)
        if now != opened:
            logging.info(
                f"Camera size changed since the device was opened "
                f"({_describe_size(opened)} -> {_describe_size(now)}); re-opening "
                f"once so the frame grabber takes it over"
            )
            previous, ia = ia, None  # _reopen releases it, also on failure
            ia = _reopen(h, previous, search_key, current_aoi(previous))
        else:
            logging.info(f"Camera size unchanged since open ({_describe_size(now)}); no re-open")
        if start:
            start_acquisition(ia)
            warmup_cam(ia)
    except Exception:
        # Never let a half-configured camera escape. An ImageAcquirer dropped
        # without destroy() keeps the device exclusively opened, and possibly
        # acquiring, so the next connect fails or blocks on a device the
        # producer still considers in use - the "I have to replug the cable"
        # symptom, produced entirely in software.
        # Release the device but keep the shared harvester: the producers stay
        # loaded for the next attempt, and nothing else is using this one.
        logging.exception("Camera setup failed; releasing the device")
        quiesce_and_stop(ia)
        raise
    return h, ia


def _open_device(h: Harvester, search_key, rescan: bool) -> ImageAcquirer:
    """Create the ImageAcquirer for *search_key* from the shared Harvester.

    Args:
        h (Harvester): The shared Harvester.
        search_key: See setup_cam.
        rescan (bool): Re-enumerate when the cached device list cannot serve
            the request. h.update() unloads and reloads every producer, so a
            re-open of a camera that was open a moment ago never does it.
    """
    # Enumerate only when the cached device list cannot serve the request.
    device = _find_device(h, search_key)
    enumerated = device is None and rescan
    if enumerated:
        h.update()
        device = _find_device(h, search_key)

    # Open exactly the device that was picked. Opening index 0 regardless
    # of what the user picked meant a second camera (or a camera behind a
    # second producer) silently opened the wrong device - possibly one
    # another process already holds. With nothing found, harvesters raises
    # its own "no device" error from the original key.
    fallback_key = 0 if search_key is None else search_key
    started = time.monotonic()
    try:
        ia = h.create(fallback_key if device is None else device)
    except Exception:
        if enumerated or not rescan:
            raise
        # The cached entry may be stale: a camera that was unplugged,
        # power-cycled or re-linked since the last scan keeps failing to
        # open from it until something re-enumerates. Do that once here
        # instead of leaving every further Connect failing the same way.
        logging.warning(
            "Opening the camera from the cached device list failed; "
            "re-enumerating and retrying once",
            exc_info=True,
        )
        h.update()
        device = _find_device(h, search_key)
        ia = h.create(fallback_key if device is None else device)
    logging.info(
        f"Device opened in {time.monotonic() - started:.3f} s: {describe_aoi_state(ia)}"
    )
    return ia


def _reapply_saved_aoi(ia: ImageAcquirer) -> None:
    """Write the AOI an earlier session saved for this camera, if any."""
    cfg = load_resolution_config(camera_config_key(ia))
    if not (isinstance(cfg.get("width"), int) and isinstance(cfg.get("height"), int)):
        return
    w, h, ox, oy = set_resolution(
        ia,
        width=cfg["width"],
        height=cfg["height"],
        offset_x=cfg.get("offset_x"),
        offset_y=cfg.get("offset_y"),
    )
    where = "" if ox is None and oy is None else f" at offset ({ox}, {oy})"
    logging.info(f"Reapplied saved resolution {w}x{h}{where}")


def camera_size(ia: ImageAcquirer) -> tuple[int | None, int | None, int | None]:
    """The camera's (Width, Height, PayloadSize); None where unreadable."""
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        return tuple(_read_int(nodes, name) for name in ("Width", "Height", "PayloadSize"))


def current_aoi(ia: ImageAcquirer) -> tuple[int | None, int | None, int | None, int | None]:
    """The camera's (Width, Height, OffsetX, OffsetY); None where it has no such node."""
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        return tuple(_read_int(nodes, name) for name in ("Width", "Height", "OffsetX", "OffsetY"))


def _read_int(node_map, name: str) -> int | None:
    """A node's numeric value as int; None if missing, unreadable or not a number."""
    value = _read_number(node_map, name)
    return None if value is None else int(value)


def _describe_size(size: tuple[int | None, int | None, int | None]) -> str:
    w, h, payload = (("-" if v is None else v) for v in size)
    return f"{w}x{h} payload {payload}"


def _reopen(
    h: Harvester,
    ia: ImageAcquirer,
    search_key,
    aoi: tuple[int | None, int | None, int | None, int | None],
) -> ImageAcquirer:
    """Release *ia* and open the same camera again, so the grabber latches *aoi*.

    The one destroy + create of an AOI change (and of a connect whose AOI
    changed the camera's size). The camera keeps its registers across the
    close/open on real hardware (the MC2066 kept its OffsetY); a producer
    that resets them (TLSimu does) gets *aoi* written again on the new,
    never-started ImageAcquirer - check_start then refuses it if a frame
    grabber latched something else meanwhile.

    Args:
        h (Harvester): The shared Harvester (never reset or updated here).
        ia (ImageAcquirer): The camera to release; always destroyed.
        search_key: The device to open again, see setup_cam.
        aoi (tuple): (width, height, offset_x, offset_y) the camera must report.

    Returns:
        ImageAcquirer: The new, verified, not acquiring camera.

    Raises:
        StartRefused: if the new grabber still disagrees with the camera; the
            new camera has been released as well.
        Exception: whatever create() raised.
    """
    quiesce_and_stop(ia)
    new = _open_device(h, search_key, rescan=False)
    try:
        reported = current_aoi(new)
        if any(want is not None and got != want for want, got in zip(aoi, reported)):
            logging.warning(
                f"The camera did not keep its AOI across the re-open ({reported} instead "
                f"of {aoi}); writing it again before the first start"
            )
            set_resolution(new, *aoi)
        logging.info(f"AOI after re-open, before start: {describe_aoi_state(new)}")
        ok, line = check_start(new)
        if not ok:
            raise StartRefused(line)
    except Exception:
        quiesce_and_stop(new)
        raise
    finally:
        sync_log()
    return new


def change_aoi(
    h: Harvester,
    ia: ImageAcquirer,
    aoi: tuple[int | None, int | None, int | None, int | None],
    search_key,
) -> tuple[ImageAcquirer, tuple[int | None, int | None, int | None, int | None]]:
    """Change the camera AOI so that the frame grabber follows it.

    The order is: quiesced stop (see quiesce_and_stop) -> write the AOI to
    the camera on this stopped ImageAcquirer -> destroy it -> create a new
    one, at whose open the grabber latches the camera's new size -> check
    camera == grabber (check_start). Exactly one destroy + create, and none
    when the camera already has *aoi* and the grabber agrees (every open
    and close is a crash risk on the CoaXPress rig). Writing the AOI after
    the re-open (the old order) left the grabber at the previous size:
    frames of the wrong geometry, or a camera frame larger than the
    grabber's DMA buffers, followed by a fatal PCIe machine check.

    The caller must have stopped every thread that fetches from *ia*. On
    any failure both ImageAcquirers are stopped and destroyed before the
    exception propagates; nothing is started here.

    Args:
        h (Harvester): The shared Harvester.
        ia (ImageAcquirer): The open camera; unusable afterwards unless it is
            returned again (AOI unchanged).
        aoi (tuple): (width, height, offset_x, offset_y) for set_resolution.
        search_key: The device, see setup_cam.

    Returns:
        tuple: (the new ImageAcquirer, the applied (w, h, offset_x, offset_y)).
    """
    logging.info(f"AOI change to {aoi} (width, height, offset_x, offset_y). Before: {describe_aoi_state(ia)}")
    try:
        quiesce_and_stop(ia, destroy=False)
        current = current_aoi(ia)
        if all(want is None or got == want for want, got in zip(aoi, current)) and check_start(ia)[0]:
            logging.info("AOI unchanged and the grabber agrees; no re-open")
            return ia, current
        applied = set_resolution(ia, *aoi)
        logging.info(f"AOI written to the stopped camera: {describe_aoi_state(ia)}")
    except Exception:
        quiesce_and_stop(ia)
        sync_log()
        raise
    return _reopen(h, ia, search_key, applied), applied


def _configure_cam(
    ia: ImageAcquirer,
    height: int | None,
    width: int | None,
    fps: int,
    blacklevel: int,
    exposure: int,
) -> None:
    """Apply the node-map settings to a fresh camera that is not acquiring.

    Split out of setup_cam so the whole create -> configure -> warm-up sequence
    fits under a single try/except there: every step below can raise, and a
    camera left open after a failure wedges the next connect.
    """
    nodes = ia.remote_device.node_map
    profile = detect_profile(nodes)
    logging.info(f"Using camera feature profile: {profile.key}")
    # Check if the FPS are okay:
    max_fps = max_frame_rate(nodes, profile)
    if max_fps is None:
        # e.g. TLSimu or cameras reporting neither a max node nor node.max
        logging.warning("Camera reports no frame rate maximum, skipping clamp")
    if max_fps is not None and fps > max_fps:
        fps = max_fps
        logging.warning(
            f"FPS is too high for the camera. Setting it to the maximum possible: {fps}"
        )
    try:
        getattr(nodes, profile.frame_rate_node).value = fps
    except Exception as e:
        logging.warning(f"Could not set {profile.frame_rate_node} to {fps}: {e}")

    exposure = cap_at_max_exposure(exposure, fps)
    try:
        getattr(nodes, profile.exposure_node).set_value(exposure)
    except Exception as e:
        logging.warning(f"Could not set {profile.exposure_node} to {exposure}: {e}")

    for size_name, offset_name, value in (("Width", "OffsetX", width), ("Height", "OffsetY", height)):
        if value is None:
            # Whole sensor; a saved AOI is applied over it next (setup_cam's
            # saved_aoi). The camera keeps the AOI of the last session,
            # offsets included, until this resets it.
            _full_sensor_axis(nodes, size_name, offset_name)
            continue
        try:
            size = getattr(nodes, size_name)
            size.value = _snap_to_increment(size, value)
        except Exception as e:
            logging.warning(f"Could not set {size_name} to {value}: {e}")

    try:
        nodes.TriggerMode.value = "Off"
    except Exception as e:
        logging.warning(f"Could not disable TriggerMode: {e}")

    for node_name, node_value in (
        ("InfoFieldFrameCounterEnable", True),
        ("InfoFieldTimeStampEnable", True),
    ):
        try:
            getattr(nodes, node_name).value = node_value
        except Exception as e:
            logging.warning(f"Could not set {node_name} = {node_value}: {e}")

    try:
        logging.info(f"Camera pixel format: {nodes.PixelFormat.value}")
    except Exception as e:
        logging.debug(f"PixelFormat node not available: {e}")
    try:
        nodes.BlackLevel.value = blacklevel
    except Exception as e:
        logging.warning(f"Could not set BlackLevel to {blacklevel}: {e}")


def warmup_cam(ia, frames: int = 10, timeout: float = FETCH_TIMEOUT_S) -> None:
    """Warm up the camera by collecting a few frames.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        frames (int): The number of frames to collect.
        timeout (float): Seconds to wait for each frame.

    Returns:
        None

    Raises:
        TimeoutError: if the camera delivers no frame within *timeout*. This
            runs on the GUI thread during connect, so it must never be allowed
            to wait indefinitely - an unbounded fetch here was a permanent,
            uncancellable freeze of the whole application.
    """
    logging.info(f"Warming up camera by collecting {frames} frames.")
    for i in range(frames):
        buffer = ia.try_fetch(timeout=timeout)
        if buffer is None:
            raise TimeoutError(
                f"The camera delivered no frame within {timeout}s while warming "
                f"up ({i} of {frames} collected). It is most likely waiting for "
                f"a trigger, or the stream cannot get through (packet size / MTU)."
            )
        buffer.queue()

def setup_webcam(idx: int) -> VideoCapture:
    """Set the webcam up using OpenCV.

    Args:
        idx (int): The index of the webcam.

    Returns:
        cv2.VideoCapture: The video capture object.
    """
    cap = VideoCapture(idx)
    return cap

def get_webcam_fps(cap):
    return cap.get(cv2.CAP_PROP_FPS)

def return_harvester_cameras(ctis: list[str]) -> list[dict]:
    """Return the cameras detected by the Harvester API.

    Args:
        ctis (list[str]): The list of paths to the GenTL Producer files.

    Returns:
        list[dict]: A list of dictionaries containing the properties of the cameras.
    """
    # The shared Harvester, so the startup scan loads each producer once and
    # the connect that follows reuses both the producers and this very device
    # list - instead of unloading them here and enumerating all over again.
    h = get_harvester(ctis)
    # Enumerate the available devices that GenTL Producers can handle:
    h.update()
    return [cam.property_dict for cam in h.device_info_list]

def get_camera_indices() -> list[int]:
    """Get the indices of the first 10 cameras that are detectable by OpenCV.

    Returns:
        list[int]: A list of indices of the cameras.
    """
    arr = []
    for i in range(10):
        try:
            cap = VideoCapture(i)
        except cv2.error:
            continue
        try:
            if cap.read()[0]:
                arr.append(i)
        finally:
            # Release every probe, not only the ones that answered. The
            # release() used to sit inside the `if`, so the nine indices that
            # did not respond leaked their V4L2 handles and kept those devices
            # busy for the rest of the session - and this runs again on every
            # rescan.
            cap.release()
    return arr


def teardown_cam(h: Harvester | None, ia: ImageAcquirer | None) -> None:
    """Release a camera, then unload its GenTL producer, best effort.

    The camera goes through quiesce_and_stop (camera stop, drain, stream
    stop, settle, destroy) before h.reset() unloads the producer; each step
    is guarded, so a failing one never skips the next. Skipping any of them
    leaves the device exclusively opened or the producer loaded, which is
    exactly what makes the *next* connect hang.

    Pass *h* only at process exit (release_harvester, the standalone
    scripts): the GUI keeps the producer loaded for the whole session.

    Callers must make sure no thread is still inside a fetch() on ``ia``:
    resetting the harvester under an in-flight fetch unloads the producer out
    from under it (see VideoThread.close).

    Args:
        h (Harvester | None): The Harvester object, if any.
        ia (ImageAcquirer | None): The ImageAcquirer object, if any.
    """
    quiesce_and_stop(ia)
    if h is not None:
        _forget_shared_harvester(h)
        try:
            # This unloads the GenTL producer library. If the app ever hangs
            # here, the log stops after this line: some vendor producers keep
            # threads or device handles alive across a reset and do not
            # survive being unloaded and reloaded per connect.
            logging.info("teardown: resetting the harvester (unloads the producer)")
            sync_log()
            h.reset()
        except Exception as e:
            logging.warning(f"Error resetting harvester during teardown: {e}")
    logging.debug("teardown: done")


# quiesce_and_stop: after the camera's own AcquisitionStop, wait until no
# buffer has arrived for QUIESCE_FRAME_PERIODS frame periods, at least
# QUIESCE_MIN_QUIET_S, and give up after QUIESCE_MAX_S in total. Fetches use
# QUIESCE_SLICE_S timeouts. QUIESCE_SETTLE_S separates the stream stop from
# destroy().
QUIESCE_FRAME_PERIODS = 3
QUIESCE_MIN_QUIET_S = 0.2
QUIESCE_MAX_S = 2.0
QUIESCE_SLICE_S = 0.02
QUIESCE_SETTLE_S = 0.1


def sync_log() -> None:
    """Flush the log files and fsync them to disk.

    A fatal machine check resets the PC without writing back the page cache:
    the whole log of the last crashed session was lost. Called after every
    stop/teardown step, AOI change and start check.
    """
    for handler in logging.getLogger().handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        try:
            handler.flush()
            if handler.stream is not None:
                os.fsync(handler.stream.fileno())
        except (OSError, ValueError, AttributeError):
            pass


def _frame_period_s(nodes) -> float | None:
    """The camera's frame period from its frame rate and exposure (µs), if readable."""
    profile = detect_profile(nodes)
    periods = []
    fps = _read_number(nodes, profile.frame_rate_node)
    if fps is not None and fps > 0:
        periods.append(1.0 / fps)
    exposure_us = _read_number(nodes, profile.exposure_node)
    if exposure_us is not None and exposure_us > 0:
        periods.append(exposure_us / 1e6)
    return max(periods) if periods else None


def _read_number(node_map, name: str) -> float | None:
    try:
        value = getattr(node_map, name).value
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _acquisition_active(nodes) -> bool:
    """Whether the camera's AcquisitionStatus says it is still acquiring."""
    try:
        value = nodes.AcquisitionStatus.value
    except Exception:
        return False
    return isinstance(value, (bool, int)) and bool(value)


def _drain(ia: ImageAcquirer, nodes, quiet_s: float, deadline: float) -> tuple[int, int, bool]:
    """Fetch and requeue until nothing arrived for *quiet_s*, or *deadline*.

    harvesters requeues an incomplete buffer itself and returns None at once,
    so a None well before the fetch timeout is also a delivery.

    Returns:
        tuple: (complete buffers requeued, incomplete deliveries, capped).
    """
    complete = incomplete = 0
    last = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= deadline:
            return complete, incomplete, True
        if now - last >= quiet_s and not _acquisition_active(nodes):
            return complete, incomplete, False
        timeout = max(min(QUIESCE_SLICE_S, deadline - now), _MIN_FETCH_S)
        asked = time.monotonic()
        try:
            buffer = ia.try_fetch(timeout=timeout)
        except Exception as e:
            logging.warning(f"quiesce: fetch failed while draining ({e}); not draining further")
            return complete, incomplete, False
        if buffer is not None:
            try:
                buffer.queue()
            except Exception as e:
                logging.warning(f"quiesce: could not requeue a drained buffer: {e}")
            complete += 1
            last = time.monotonic()
        elif time.monotonic() - asked < timeout / 2:
            incomplete += 1
            last = time.monotonic()


def quiesce_and_stop(ia: ImageAcquirer | None, destroy: bool = True) -> None:
    """Stop a camera so no DMA can reach a buffer after it is revoked.

    harvesters' ImageAcquirer.stop() executes the camera's AcquisitionStop
    and then at once kills the data stream, flushes and revokes the buffers.
    A CoaXPress camera keeps sending until it has really stopped, so a frame
    still in flight could be DMA'd by the grabber into revoked (and, with
    the IOMMU on, unmapped) memory - the suspected trigger of the fatal PCIe
    machine checks on Disconnect + Connect. So, in this order:

    1. the camera's AcquisitionStop, executed here first;
    2. fetch and requeue with short timeouts until no buffer has arrived for
       max(QUIESCE_FRAME_PERIODS frame periods, QUIESCE_MIN_QUIET_S) and
       AcquisitionStatus (if the camera has it) reads false - at most
       QUIESCE_MAX_S in total;
    3. ia.stop(): data stream stop, flush, revoke;
    4. with *destroy*: QUIESCE_SETTLE_S of settle time, then ia.destroy().

    Every step is logged at INFO with its timing and the log is fsynced.
    Never raises. The one path for every stop and teardown of an
    ImageAcquirer (pause, disconnect, AOI change, recording end, app exit,
    the standalone scripts). The caller must have stopped every thread that
    fetches from *ia*.

    Args:
        ia (ImageAcquirer | None): The camera; None does nothing.
        destroy (bool): Destroy it afterwards (closes stream and device).
    """
    if ia is None:
        return
    with IA_LOCK:
        began = time.monotonic()
        try:
            acquiring = bool(ia.is_acquiring())
        except Exception:
            acquiring = True  # unknown: take the careful path
        if acquiring:
            try:
                nodes = ia.remote_device.node_map
            except Exception:
                nodes = None
            period = _frame_period_s(nodes) if nodes is not None else None
            quiet = max(QUIESCE_FRAME_PERIODS * (period or 0.0), QUIESCE_MIN_QUIET_S)
            step = time.monotonic()
            try:
                nodes.AcquisitionStop.execute()
                logging.info(f"quiesce: camera AcquisitionStop executed ({time.monotonic() - step:.3f} s)")
            except Exception as e:
                logging.info(f"quiesce: no camera AcquisitionStop ({type(e).__name__}: {e})")
            sync_log()
            step = time.monotonic()
            complete, incomplete, capped = _drain(ia, nodes, quiet, began + QUIESCE_MAX_S)
            frame = "unknown" if period is None else f"{period:.4f} s"
            message = (
                f"quiesce: drained {complete} buffer(s) + {incomplete} incomplete in "
                f"{time.monotonic() - step:.3f} s (quiet period {quiet:.3f} s, frame period {frame})"
            )
            if capped:
                logging.warning(f"{message}; still not quiet after {QUIESCE_MAX_S:g} s, stopping anyway")
            else:
                logging.info(message)
        else:
            logging.info("quiesce: not acquiring, nothing to drain")
        step = time.monotonic()
        try:
            ia.stop()
            logging.info(f"quiesce: stream stopped, buffers revoked ({time.monotonic() - step:.3f} s)")
        except Exception as e:
            logging.warning(f"quiesce: ia.stop() failed: {e}")
        sync_log()
        if destroy:
            time.sleep(QUIESCE_SETTLE_S)
            step = time.monotonic()
            try:
                ia.destroy()
                logging.info(
                    f"quiesce: destroyed after {QUIESCE_SETTLE_S:g} s settle "
                    f"({time.monotonic() - step:.3f} s)"
                )
            except Exception as e:
                logging.warning(f"quiesce: ia.destroy() failed: {e}")
        logging.info(f"quiesce: done in {time.monotonic() - began:.3f} s")
    sync_log()


class StartRefused(RuntimeError):
    """Acquisition was not started: the frame grabber disagrees with the camera."""


def check_start(ia: ImageAcquirer) -> tuple[bool, str]:
    """Whether *ia* may start: the grabber and stream buffers fit the camera.

    harvesters sizes the buffers it announces on start() from the data
    stream's payload_size whenever the stream defines one, and the grabber
    lays frames out by its own Width/Height - both latched when the device
    was opened. A camera frame larger than that is a DMA overrun (the
    suspected cause of the fatal PCIe errors); a smaller one gives timeouts
    or frames of the wrong geometry. Refused when:

    - the grabber (GenTL device module) has Width/Height nodes and they
      differ from the camera's Width/Height;
    - a stream's payload_size is smaller than the camera's PayloadSize.

    Producers without grabber nodes (TLSimu, GigE, USB3) skip the first
    check. Logs ``start check: camera WxH payload P, grabber WxH, stream
    payload S -> OK/REFUSED`` and fsyncs the log.

    Returns:
        tuple[bool, str]: (may start, that log line).
    """
    with IA_LOCK:
        cam_w, cam_h, cam_payload = camera_size(ia)
        try:
            grabber_nodes = ia.device.node_map
        except Exception:
            grabber_nodes = None
        grabber = (
            (_read_int(grabber_nodes, "Width"), _read_int(grabber_nodes, "Height"))
            if grabber_nodes is not None
            else (None, None)
        )
        payloads = []
        try:
            streams = list(ia.data_streams)
        except Exception:
            streams = []
        for stream in streams:
            try:
                if stream.defines_payload_size():
                    payloads.append(int(stream.payload_size))
            except Exception as e:
                logging.warning(f"start check: stream payload size unreadable: {e}")
    problems = []
    for axis, got, want in (("Width", grabber[0], cam_w), ("Height", grabber[1], cam_h)):
        if got is not None and got != want:
            problems.append(f"grabber {axis} {got} != camera {axis} {want}")
    for payload in payloads:
        if cam_payload is not None and payload < cam_payload:
            problems.append(f"stream payload {payload} < camera PayloadSize {cam_payload}")

    def dash(value):
        return "-" if value is None else value

    grabber_text = "-" if grabber == (None, None) else f"{dash(grabber[0])}x{dash(grabber[1])}"
    stream_text = "/".join(str(p) for p in payloads) or "-"
    line = (
        f"start check: camera {dash(cam_w)}x{dash(cam_h)} payload {dash(cam_payload)}, "
        f"grabber {grabber_text}, stream payload {stream_text} -> "
        + ("OK" if not problems else f"REFUSED ({'; '.join(problems)})")
    )
    (logging.info if not problems else logging.error)(line)
    sync_log()
    return not problems, line


def start_acquisition(ia: ImageAcquirer) -> None:
    """Start acquisition, but only after check_start has passed.

    The one way anything in this code base starts an ImageAcquirer.

    Raises:
        StartRefused: the check failed; nothing was started.
    """
    with IA_LOCK:
        ok, line = check_start(ia)
        if not ok:
            raise StartRefused(
                f"Acquisition not started: the frame grabber is not set up for the "
                f"camera's image size. {line}"
            )
        ia.start()


# A recording (or a loaded file) may use at most this share of the RAM that is
# available when it starts. np.empty/np.zeros succeed instantly under Linux
# overcommit and only commit pages as frames are written, so an oversized
# recording does not fail up front - it fills RAM frame by frame and then
# swap-thrashes the whole machine into what looks like a freeze.
MEMORY_BUDGET_FRACTION = 0.8


def available_memory_bytes() -> int | None:
    """RAM that can be used without swapping, or None if it cannot be determined."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def ensure_memory_for(nbytes: int, what: str) -> None:
    """Raise MemoryError if *nbytes* do not fit in the available RAM budget.

    Args:
        nbytes (int): Size of the array about to be allocated.
        what (str): What it is for, used in the error message.
    """
    available = available_memory_bytes()
    if available is None:
        return
    budget = int(available * MEMORY_BUDGET_FRACTION)
    if nbytes > budget:
        gib = 1024**3
        raise MemoryError(
            f"The {what} needs {nbytes / gib:.1f} GiB, but only "
            f"{available / gib:.1f} GiB of RAM is available ({budget / gib:.1f} GiB "
            f"usable). Use fewer frames or a smaller image size, or free memory "
            f"first (e.g. save and close the current recording)."
        )


def get_webcam_image(cap: VideoCapture) -> np.ndarray:
    """Get an image from the webcam.

    Args:
        cap (VideoCapture): The video capture object.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def record_webcam_frames(
    cap: VideoCapture,
    n_frames: int,
    progress: None | QProgressBar,
    preview: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Record n_frames from the webcam.

    Args:
        cap (VideoCapture): The video capture object.
        n_frames (int): The number of frames to record.
        preview (Callable[[np.ndarray], None] | None): Optional callback
            invoked with each recorded frame, e.g. to show it live in the UI.

    Returns:
        np.ndarray: The recorded frames as a numpy array.
    """
    if progress is not None:
        if n_frames == 1:
            progress.setMaximum(1)
        else:
            progress.setMaximum(n_frames - 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ensure_memory_for(n_frames * height * width, "recording")
    frames = np.zeros((n_frames, height, width), dtype=np.uint8)
    for i in trange(n_frames, desc="Recording frames"):
        frames[i] = get_webcam_image(cap)
        if preview is not None:
            preview(frames[i])
        if progress is not None:
            if n_frames == 1:
                progress.setValue(1)
            else:
                progress.setValue(i)
    return frames

def get_random_image(_: None) -> np.ndarray:
    """Get a random image.

    Args:
        _: None

    Returns:
        np.ndarray: A random image.
    """
    return np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)

def record_random_frames(
    n_frames: int,
    progress: None | QProgressBar,
    preview: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Record n_frames of random images.

    Args:
        n_frames (int): The number of frames to record.
        preview (Callable[[np.ndarray], None] | None): Optional callback
            invoked with each recorded frame, e.g. to show it live in the UI.

    Returns:
        np.ndarray: The recorded frames as a numpy array.
    """
    if progress is not None:
        if n_frames == 1:
            progress.setMaximum(1)
        else:
            progress.setMaximum(n_frames - 1)
    ensure_memory_for(n_frames * 1080 * 1920, "recording")
    frames = np.zeros((n_frames, 1080, 1920), dtype=np.uint8)
    for i in trange(n_frames, desc="Recording frames"):
        frames[i] = get_random_image(None)
        if preview is not None:
            preview(frames[i])
        if progress is not None:
            if n_frames == 1:
                progress.setValue(1)
            else:
                progress.setValue(i)
    return frames

# Pixel format support -------------------------------------------------------
#
# harvesters already expands every delivered format through its bundled PFNC
# dictionary (Component2DImage._data = proxy.expand(...)): packed formats such
# as Mono12Packed, Bayer*10Packed or RGB10p arrive as natural uint8/uint16
# arrays of one value per pixel component. What remains for us is to turn
# whatever "one value per component" payload we get into an owned uint8
# (height, width) grayscale frame: demosaic Bayer mosaics, extract luma from
# YUV, drop the high byte of >8-bit formats, and keep Mono8 as a plain copy.

# GenICam names a Bayer mosaic by its top-left 2x2 cell (BayerRG8 = R G / G B).
# OpenCV's short names are shifted by one (COLOR_BayerRG2GRAY is RGGB's
# neighbour, == COLOR_BayerBGGR2GRAY), which swaps the red and blue weights of
# the luminance; its explicit four-letter names match GenICam's.
_BAYER_DEMOSAIC = {
    "GR": cv2.COLOR_BayerGRBG2GRAY,
    "RG": cv2.COLOR_BayerRGGB2GRAY,
    "GB": cv2.COLOR_BayerGBRG2GRAY,
    "BG": cv2.COLOR_BayerBGGR2GRAY,
}

# Packed YUV / YCbCr layouts, upper-cased format name ->
# (bytes per group, pixels per group, positions of the luma bytes in a group).
# Legacy GigE Vision names plus their PFNC spellings, which is what current
# cameras report; without them those formats degraded to black frames.
_YUV_LUMA = {
    # 4:2:2, U Y0 V Y1 (UYVY)
    "YUV422PACKED": (4, 2, (1, 3)),
    "YUV422_8_UYVY": (4, 2, (1, 3)),
    "YCBCR422_8_CBYCRY": (4, 2, (1, 3)),
    "YCBCR601_422_8_CBYCRY": (4, 2, (1, 3)),
    "YCBCR709_422_8_CBYCRY": (4, 2, (1, 3)),
    # 4:2:2, Y0 U Y1 V (YUYV)
    "YUV422_8": (4, 2, (0, 2)),
    "YCBCR422_8": (4, 2, (0, 2)),
    "YCBCR601_422_8": (4, 2, (0, 2)),
    "YCBCR709_422_8": (4, 2, (0, 2)),
    # 4:1:1, U Y0 Y1 V Y2 Y3
    "YUV411PACKED": (6, 4, (1, 2, 4, 5)),
    "YUV411_8_UYYVYY": (6, 4, (1, 2, 4, 5)),
    "YCBCR411_8_CBYYCRYY": (6, 4, (1, 2, 4, 5)),
    "YCBCR601_411_8_CBYYCRYY": (6, 4, (1, 2, 4, 5)),
    "YCBCR709_411_8_CBYYCRYY": (6, 4, (1, 2, 4, 5)),
    # 4:4:4 with luma first (legacy YUV444Packed as handled before, YCbCr8)
    "YUV444PACKED": (3, 1, (0,)),
    "YCBCR8": (3, 1, (0,)),
    # 4:4:4, U Y V / Cb Y Cr
    "YUV8_UYV": (3, 1, (1,)),
    "YCBCR8_CBYCR": (3, 1, (1,)),
    "YCBCR601_8_CBYCR": (3, 1, (1,)),
    "YCBCR709_8_CBYCR": (3, 1, (1,)),
}

_BITS_FROM_FMT = re.compile(r"(1[0-6])")


def _bits_from_fmt(fmt: str) -> int | None:
    """Bit depth implied by a format name (Mono12Packed -> 12), if any."""
    m = _BITS_FROM_FMT.search(fmt)
    return int(m.group(1)) if m else None


def grayscale8_from_bytes(
    fmt: str, data, width: int, height: int
) -> np.ndarray:
    """Convert one frame's payload to an owned uint8 (height, width) array.

    Args:
        fmt: GenICam/PFNC pixel format name (Mono8, BayerRG10Packed, ...).
        data: the payload as expanded by harvesters' PFNC dictionary (a flat
            uint8/uint16 numpy array).
        width, height: logical pixel dimensions of the frame.

    Returns:
        A new uint8 (height, width) grayscale array.

    Raises:
        ValueError: for formats that cannot be interpreted (floating point,
            addressing/padding layouts we do not reconstruct). The caller is
            expected to degrade to a black frame rather than crash.
    """
    h, w = int(height), int(width)
    fmt = str(fmt)
    data = np.asarray(data)
    if data.ndim != 1:
        data = data.reshape(-1)
    if data.dtype == np.float32:
        raise ValueError("floating-point pixel data is not displayable")
    if data.dtype not in (np.uint8, np.uint16, np.int8):
        # Defensive: uint32/float64 payloads would silently misbehave below.
        raise ValueError(f"unsupported payload dtype {data.dtype}")

    def shift8(arr: np.ndarray) -> np.ndarray:
        """Map an integral per-component array to an owned uint8 (bitshift)."""
        bits = _bits_from_fmt(fmt) or max(8, 8 * arr.dtype.itemsize)
        shift = bits - 8
        if shift <= 0:
            # Always copy: the input may be a read-only view of the producer's
            # DMA buffer, which gets reused for later frames.
            return np.array(arr, dtype=np.uint8, copy=True)
        return (arr >> shift).astype(np.uint8)

    up = fmt.upper()

    # 1) Bayer mosaics (single component; needs demosaic).
    bayer = re.match(r"Bayer(GR|RG|GB|BG)", fmt)
    if bayer:
        if data.size != h * w:
            raise ValueError(f"{fmt}: expected {h * w} pixels, got {data.size}")
        return cv2.cvtColor(
            shift8(data).reshape(h, w), _BAYER_DEMOSAIC[bayer.group(1)]
        )

    # 2) YUV / YCbCr: luma extraction from the packed byte order.
    yuv = _YUV_LUMA.get(up)
    if yuv is not None:
        group, pixels, luma = yuv
        if (h * w) % pixels or data.size != (h * w // pixels) * group:
            raise ValueError(
                f"{fmt}: expected {(h * w // pixels) * group} bytes for {w}x{h}, "
                f"got {data.size}"
            )
        return data.reshape(-1, group)[:, list(luma)].reshape(h, w).copy()

    # 3) 3- and 4-component color (RGB/BGR or RGBA/BGRA; channel order does not
    #    matter for luminance).
    if data.size == 4 * h * w:
        return cv2.cvtColor(
            shift8(data).reshape(h, w, 4),
            cv2.COLOR_RGBA2GRAY if "RGBA" in up else cv2.COLOR_BGRA2GRAY,
        )
    if data.size == 3 * h * w:
        return cv2.cvtColor(shift8(data).reshape(h, w, 3), cv2.COLOR_RGB2GRAY)

    # 4) Single-component monochrome (Mono8, Mono10/12/14/16, packed mono,
    #    Confidence*, R/G/B-planes).
    if data.size == h * w:
        bits = _bits_from_fmt(fmt) or max(8, 8 * data.dtype.itemsize)
        shift = bits - 8
        if shift > 0:
            flat = (data >> shift).astype(np.uint8)
        elif data.dtype == np.uint8:
            flat = data  # no copy yet; the final .copy() below is the only one
        else:  # e.g. Mono8Signed (int8) payloads
            flat = data.astype(np.uint8)
        # reshape() would return a view of (possibly producer-owned) memory;
        # .copy() guarantees an owned (h, w) frame. For Mono8 this is exactly
        # one linear memcpy — matching the historic fast path.
        return np.ascontiguousarray(flat).reshape(h, w).copy()

    raise ValueError(f"unsupported pixel format {fmt!r} for {w}x{h} ({data.size} values)")


def component_to_grayscale8(component) -> np.ndarray:
    """Return an owned uint8 (height, width) grayscale frame for a payload.

    Unknown/unhandled formats log a warning (once per format) and yield a
    black frame so a camera with an exotic pixel format degrades to a visible
    "no image" instead of killing the video thread.
    """
    width, height = int(component.width), int(component.height)
    try:
        return grayscale8_from_bytes(
            component.data_format, component.data, width, height
        )
    except (ValueError, cv2.error) as e:
        fmt = str(getattr(component, "data_format", "unknown"))
        if fmt not in _warned_formats:
            logging.warning(
                f"Unsupported pixel format {fmt!r} ({e}); showing a black frame"
            )
            _warned_formats.add(fmt)
        return np.zeros((height, width), dtype=np.uint8)


_warned_formats = set()

def get_harvester_image(
    ia: ImageAcquirer,
    timeout: float | None = None,
    keep_waiting: Callable[[], bool] | None = None,
) -> np.ndarray | None:
    """
    Get the image from the camera using the Harvester API.

    The wait for a frame is split into FETCH_SLICE_S slices with IA_LOCK
    released in between: a GUI-thread node write waits at most one slice for
    the lock, and *keep_waiting* is checked after every slice so a stop request
    ends the wait within a slice instead of after the full timeout.
    Incomplete buffers (harvesters requeues them and returns None at once) just
    mean another slice; they are not mistaken for a stalled camera.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        timeout (float | None): Seconds to wait for a frame, FETCH_TIMEOUT_S by
            default.
        keep_waiting (Callable[[], bool] | None): Polled between slices; once it
            returns False the wait is abandoned.

    Returns:
        np.ndarray | None: The frame; a black frame if the camera delivered
        nothing within *timeout*; None if *keep_waiting* ended the wait.
    """
    timeout = FETCH_TIMEOUT_S if timeout is None else timeout
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        with IA_LOCK:
            buffer = ia.try_fetch(timeout=max(min(FETCH_SLICE_S, remaining), _MIN_FETCH_S))
            if buffer is not None:
                with buffer:
                    # See component_to_grayscale8: Mono8 stays a plain copy,
                    # while 10/12/16-bit, Bayer and color formats are converted
                    # to a grayscale uint8 frame here so the whole app
                    # (display, recording, export) only ever deals with
                    # natural (h, w).
                    return component_to_grayscale8(buffer.payload.components[0])
        if keep_waiting is not None and not keep_waiting():
            return None
        if time.monotonic() >= deadline:
            break
        # Off the lock and the GIL for a moment. Releasing IA_LOCK alone is
        # not enough: this thread re-took it before a waiting GUI thread got
        # to run, so a node write (or the old AOI change) waited out the whole
        # FETCH_TIMEOUT_S while the camera delivered nothing - most likely the
        # ~5 s per AOI apply in the MC2066 rig's log.
        time.sleep(_LOCK_YIELD_S)
    # Timed out instead of hanging: a camera that stops delivering degrades to
    # a black live view the user can still disconnect from.
    logging.warning(f"No frame within {timeout}s; showing a black image")
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        return np.zeros((nodes.Height.value, nodes.Width.value), dtype=np.uint8)


def _harvester_frame_into(
    ia: ImageAcquirer, out: np.ndarray, timeout: float | None = None
) -> int:
    """Fetch one frame from the camera directly into a preallocated buffer.

    Writes the (height, width) C-contiguous frame into ``out`` with a single
    linear copy instead of allocating a new array per frame, which is what the
    hot recording path needs to sustain high frame rates.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        out (np.ndarray): The output buffer of shape (height, width).
        timeout (float | None): Seconds to wait for a complete frame,
            FETCH_TIMEOUT_S by default.

    Returns:
        int: How many incomplete buffers were dropped before this frame.

    Raises:
        TimeoutError: if no complete frame arrived within *timeout*.
    """
    timeout = FETCH_TIMEOUT_S if timeout is None else timeout
    deadline = time.monotonic() + timeout
    incomplete = 0
    with IA_LOCK:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Bounded, so a camera that stops delivering aborts the
                # recording instead of hanging the GUI thread. Raising rather
                # than writing a black frame is deliberate: a silent black gap
                # in the middle of a recording is worse than a loud failure.
                raise TimeoutError(
                    f"The camera delivered no frame within {timeout}s "
                    f"while recording."
                )
            buffer = ia.try_fetch(timeout=max(remaining, _MIN_FETCH_S))
            if buffer is not None:
                break
            # harvesters requeues an incomplete buffer and returns None at
            # once; that is one lost frame, not a stalled camera. Keep going
            # until the deadline instead of discarding the whole recording.
            incomplete += 1
        with buffer:
            component = buffer.payload.components[0]
            if component.data_format in ("Mono8", "Mono8Signed"):
                # Fast path: a single linear copy, no intermediate array.
                np.copyto(
                    out,
                    np.frombuffer(component.data, dtype=np.uint8).reshape(
                        component.height, component.width
                    ),
                )
            else:
                np.copyto(out, component_to_grayscale8(component))
    return incomplete

def get_remote_nodes(ia: ImageAcquirer) -> dict:
    """Get all nodes and their values from the camera.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        dict: A dictionary containing all nodes and their values.
    """
    dict = {}
    nodes = ia.remote_device.node_map
    for node in dir(nodes):
        try:
            dict[node] = getattr(ia.remote_device.node_map, node).value
        except Exception as e:
            logging.warning(f"Error while getting node {node}'s value: {e}")
    return dict

def record_harvester_frames(
    ia: ImageAcquirer,
    n_frames: int,
    progress: None | QProgressBar,
    preview: Callable[[np.ndarray], None] | None = None,
) -> np.ndarray:
    """Record n_frames from the camera.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        n_frames (int): The number of frames to record.
        preview (Callable[[np.ndarray], None] | None): Optional callback
            invoked with each recorded frame, e.g. to show it live in the UI.

    Returns:
        np.ndarray: The recorded frames as a numpy array.

    Raises:
        MemoryError: if the recording would not fit in the available RAM.
        RecordingInterrupted: if the camera stopped delivering part-way; its
            ``frames`` attribute holds what was recorded until then.
    """
    if progress is not None:
        if n_frames == 1:
            progress.setMaximum(1)
        else:
            progress.setMaximum(n_frames - 1)
    with IA_LOCK:
        width = ia.remote_device.node_map.Width.value
        height = ia.remote_device.node_map.Height.value
    ensure_memory_for(n_frames * height * width, "recording")
    # Preallocate the full recording once as (n_frames, height, width) and
    # copy each frame in place. Every per-frame step is a linear memcpy; the
    # old append-a-list-then-stack path re-ran the slow strided transpose per
    # frame (~10 ms at 2048x2048), which capped recording well below the
    # camera's nominal frame rate.
    frames = np.empty((n_frames, height, width), dtype=np.uint8)
    recorded = 0
    incomplete = 0
    # Hold IA_LOCK for the whole start->fetch loop->stop sequence, not just the
    # individual fetches: start()/stop() are state-changing GenTL calls on the
    # shared device, and leaving them unlocked let a still-running VideoThread
    # fetch() interleave with them (see ui.on_btn_acquire_clicked). IA_LOCK is
    # reentrant, so _harvester_frame_into's own per-frame lock nests fine here.
    with IA_LOCK:
        # Checked: refused (StartRefused) when the frame grabber was set up
        # for another image size than the camera's.
        start_acquisition(ia)
        try:
            for i in trange(n_frames):
                incomplete += _harvester_frame_into(ia, frames[i])
                recorded = i + 1
                if preview is not None:
                    preview(frames[i])
                if progress is not None:
                    if n_frames == 1:
                        progress.setValue(1)
                    else:
                        progress.setValue(i)
        except TimeoutError as e:
            # Keep what was recorded. frames[:recorded] is a view, but the
            # untouched tail of np.empty was never committed, so it costs no RAM.
            raise RecordingInterrupted(
                f"{e} {recorded} of {n_frames} frames were recorded.",
                frames[:recorded],
            ) from e
        finally:
            # Stop on every exit, including a fetch timeout. A camera left
            # acquiring keeps filling buffers nobody drains, and the next
            # start() would then run against an already-started device.
            # Quiesced: the frames still in flight are drained before the
            # buffers are revoked. Never raises.
            quiesce_and_stop(ia, destroy=False)
    if incomplete:
        logging.warning(
            f"{incomplete} incomplete frame(s) were dropped by the producer during "
            f"the recording; the recording has gaps there."
        )
    return frames

def get_all_attributes_harvesters(ia: ImageAcquirer, budget_s: float = 10.0) -> dict:
    """Get all attributes of the camera.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        budget_s (float): Stop reading after this many seconds. Every node is a
            register read; on a camera that stopped answering each one waits
            for its own GenCP timeout, and hundreds of them froze the GUI for
            minutes.

    Returns:
        dict: A dictionary containing all attributes of the camera.
    """
    attributes = {}
    deadline = time.monotonic() + budget_s
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        for node in dir(nodes):
            if time.monotonic() > deadline:
                logging.warning(
                    f"Reading the camera attributes took over {budget_s:.0f}s; "
                    f"stored the {len(attributes)} read so far"
                )
                break
            try:
                attributes[node] = getattr(nodes, node).value
            except Exception as e:
                logging.debug(f"Error while getting node {node}'s value: {e}")
    return attributes


def info_fields_enabled(ia: ImageAcquirer) -> bool:
    """Whether the camera superimposes the Mikrotron info fields on Mono8 frames.

    Only then do the first pixels of a frame carry a frame counter and a
    timestamp. On every other camera (or pixel format) they are image content,
    and decoding them would store pixel values as "frame_counters" and
    "timestamps".
    """
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        try:
            return (
                bool(nodes.InfoFieldFrameCounterEnable.value)
                and bool(nodes.InfoFieldTimeStampEnable.value)
                and str(nodes.PixelFormat.value) == "Mono8"
            )
        except Exception:
            return False

def get_harvester_nodemap(ia: ImageAcquirer):
    return ia.remote_device.node_map


def get_max_exposure(fps: int) -> int:
    return int((1.0 / fps) * 1e6 - 1)

def cap_at_max_exposure(exposure: int, fps: int) -> int:
    max_exposure = get_max_exposure(fps)
    if exposure > max_exposure:
        exposure = max_exposure
        logging.warning(
            f"Exposure time is too high for given FPS. Setting it to the maximum: {exposure}"
        )
    return exposure

def get_max_fps_harvesters(ia: ImageAcquirer) -> int:
    return ia.remote_device.node_map.AcquisitionFrameRateMax.value

def extract_frame_counter(img: np.ndarray) -> dict:
    """Extract the superimposed frame counter from the image.

    If enabled, a frame counter will be superimposed upon each captured frame or ROI.
    The frame counter occupies 4 pixels in the upper left corner of each frame starting
    with pixel number "0". After each activation, the counter starts with "0". When reaching
    the maximal value or after each acquisition start command it will restart with "0".

    Args:
        img (np.ndarray): The image to extract the frame counter from.

    Returns:
        int: The frame counter.
    """
    # .flat[] picks the 4 pixels without copying the whole frame (flatten()
    # copied every frame of a recording just to read 8 pixels).
    pixels = np.asarray(img.flat[0:4]).astype(np.uint32)

    # Extract frame counter parts
    lsb = pixels[0]
    mid = pixels[1]
    msb = pixels[2]
    roi = pixels[3]

    # Combine frame counter parts (little endian)
    frame_counter = lsb | (mid << 8) | (msb << 16)
    _ = roi  # Unused

    return frame_counter

def extract_timestamp(img: np.ndarray) -> int:
    """Extract the superimposed timestamp from the image.

    If enabled, a 32 bit time stamp will be superimposed on each captured frame or ROI.
    The frequency of the time stamp counter amounts to 25 MHz (period = 40 nanoseconds).
    The frame counter occupies 4 pixels in the upper left corner of each frame,
    starting with pixel number 4. The time stamp counter is reset to zero when the camera is powered up.

    Args:
        img (np.ndarray): The image to extract the timestamp from.

    Returns:
        int: The timestamp in 40 ns.
    """
    pixels = np.asarray(img.flat[4:8]).astype(np.uint32)

    # Extract timestamp parts
    lsb = pixels[0]  # Pixel 4: bits 0-7
    mid_low = pixels[1] << 8  # Pixel 5: bits 8-15
    mid_high = pixels[2] << 16  # Pixel 6: bits 16-23
    msb = pixels[3] << 24  # Pixel 7: bits 24-31

    # Combine timestamp parts (little endian)
    timestamp = msb | mid_high | mid_low | lsb

    return timestamp

def set_attribute(ia, attribute, value):
    """Set an attribute of the camera.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        attribute (str): The attribute to set.
        value (int): The value to set the attribute to.

    Returns:
        None
    """
    with IA_LOCK:
        nodemap = ia.remote_device.node_map
        try:
            attribute = getattr(nodemap, attribute)
            attribute.value = value
            logging.debug(f"Set attribute {attribute} to {value}")
        except Exception as e:
            logging.warning(f"Error while setting attribute {attribute} to {value}: {e}")


def snap_value(value: int, lo: int, hi: int, inc: int = 1) -> int:
    """Snap *value* down onto the grid ``lo + k * inc`` and clamp to [lo, hi].

    The grid of a GenICam IInteger node: valid values are its min plus a
    multiple of its increment.

    Args:
        value (int): The requested value.
        lo (int): The node's minimum (the grid origin).
        hi (int): The node's maximum.
        inc (int): The node's increment; values below 1 count as 1.

    Returns:
        int: The largest grid value <= *value* within [lo, hi] (*lo* if
        *value* is below it).
    """
    inc = max(int(inc), 1)
    value = max(min(int(value), hi), lo)
    snapped = lo + ((value - lo) // inc) * inc
    return max(min(snapped, hi), lo)


def _node_inc(node) -> int:
    """A node's increment, 1 when it reports none."""
    try:
        return max(int(node.inc), 1)
    except (AttributeError, TypeError):
        return 1


def _snap_to_increment(node, value: int) -> int:
    """Snap *value* to ``node``'s increment and clamp to [min, max].

    AOI nodes on real cameras usually accept only a subset of values (e.g.
    width must be a multiple of 4). Snapping before writing avoids silent
    producer-side validation errors and gives the user a predictable result.

    Args:
        node: A GenICam IInteger node exposing min/max/inc.
        value (int): The requested value.

    Returns:
        int: The snapped value within [node.min, node.max].
    """
    return snap_value(value, int(node.min), int(node.max), _node_inc(node))


def set_resolution(
    ia: ImageAcquirer,
    width: int | None = None,
    height: int | None = None,
    offset_x: int | None = None,
    offset_y: int | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Set the camera AOI (Width/Height and sensor offsets) on a stopped camera.

    Only half of an AOI change: a frame grabber takes the camera's image
    size over when the device is opened, and harvesters sizes the buffers it
    announces on start() from the stream's payload size, which a grabber
    does not update after a camera-side write. So the camera must be
    re-opened between this write and the next start - see change_aoi (the
    GUI, aoi_probe.py) and setup_cam, which check the grabber before any
    start. The old stop -> write -> start on the same ImageAcquirer gave a
    black live view on the Active Silicon CoaXPress rig, and fatal PCIe
    machine checks followed; a camera that is acquiring is refused here.
    Values are snapped to each node's increment and clamped to [min, max];
    nodes the camera does not expose (e.g. offsets on TLSimu) are skipped.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object; must not be acquiring.
        width (int | None): New Width, or None to leave unchanged.
        height (int | None): New Height, or None to leave unchanged.
        offset_x (int | None): New OffsetX, or None to leave unchanged.
        offset_y (int | None): New OffsetY, or None to leave unchanged.

    Returns:
        tuple[int | None, ...]: The applied (width, height, offset_x, offset_y).

    Raises:
        RuntimeError: if *ia* is acquiring.
    """
    with IA_LOCK:
        if ia.is_acquiring():
            raise RuntimeError(
                "The AOI is only written to a stopped camera, which is then "
                "re-opened before it starts again (see change_aoi)."
            )
        nodes = ia.remote_device.node_map
        # SFNC couples each offset to its size (OffsetX + Width <= WidthMax).
        # Writing the offset first fails while the current size is large;
        # writing the size first clamps a saved offset to the room the old
        # offset left (a camera starting at full size clamped every saved
        # offset to 0). Zero the offset, set the size, then the offset.
        for offset_name, offset_value, size_name, size_value in (
            ("OffsetX", offset_x, "Width", width),
            ("OffsetY", offset_y, "Height", height),
        ):
            if offset_value is not None:
                _write_snapped(nodes, offset_name, 0)
            if size_value is not None:
                _write_snapped(nodes, size_name, size_value)
            if offset_value is not None:
                _write_snapped(nodes, offset_name, offset_value)
        applied: list[int | None] = [None, None, None, None]
        for i, node_name in enumerate(("Width", "Height", "OffsetX", "OffsetY")):
            try:
                applied[i] = int(getattr(nodes, node_name).value)
            except Exception:
                pass
    return tuple(applied)


def _write_snapped(nodes, node_name: str, value: int) -> None:
    """Write *value*, snapped to the node's grid; log instead of raising."""
    try:
        node = getattr(nodes, node_name)
        node.value = _snap_to_increment(node, value)
    except Exception as e:
        logging.warning(f"Could not set {node_name} to {value}: {e}")


@dataclass(frozen=True)
class AoiAxis:
    """The AOI constraints along one sensor axis (Width+OffsetX or Height+OffsetY).

    Attributes:
        size_min (int): Minimum Width/Height.
        size_inc (int): Width/Height increment.
        sensor (int): Full sensor extent, see sensor_extent().
        offset_min (int): Minimum OffsetX/OffsetY.
        offset_inc (int): OffsetX/OffsetY increment.
        has_offset (bool): Whether the camera exposes the offset node at all
            (TLSimu does not: its AOI always starts at the sensor origin).
    """

    size_min: int
    size_inc: int
    sensor: int
    offset_min: int = 0
    offset_inc: int = 1
    has_offset: bool = True


@dataclass(frozen=True)
class AoiLimits:
    """The AOI constraints of a camera, see read_aoi()."""

    x: AoiAxis
    y: AoiAxis


def sensor_extent(nodes, size_name: str, offset_name: str) -> int:
    """The full sensor extent along one axis, read without writing anything.

    WidthMax/HeightMax if the camera has them, else SensorWidth/SensorHeight,
    else Width.max plus the current offset: SFNC caps Width at
    WidthMax - OffsetX, so Width.max alone shrinks with the offset (the
    MC2066 reported Width.max 1120 = 1920 - 800 at OffsetX 800). On a camera
    whose Width.max does not follow the offset that sum overshoots, which is
    harmless where it is used: set_resolution zeroes the offset and clamps
    the size to Width.max, and _full_sensor_axis zeroes the offset first.

    Args:
        nodes: The camera's node map.
        size_name (str): "Width" or "Height".
        offset_name (str): The matching "OffsetX" or "OffsetY".

    Returns:
        int: The sensor extent in pixels.

    Raises:
        AttributeError: if the camera has no *size_name* node.
    """
    for name in (f"{size_name}Max", f"Sensor{size_name}"):
        try:
            return int(getattr(nodes, name).value)
        except Exception:
            pass
    try:
        offset = int(getattr(nodes, offset_name).value)
    except Exception:
        offset = 0
    return int(getattr(nodes, size_name).max) + offset


def _full_sensor_axis(nodes, size_name: str, offset_name: str) -> None:
    """Read out the whole sensor along one axis of a camera that is not acquiring.

    The offset goes to its minimum first: SFNC couples it to the size, so
    writing the full size under an old offset is out of range (the MC2066
    refused Height 1080 under OffsetY 612: "must be equal or smaller than
    Max = 468"). With the offset at 0, Width.max is the sensor width even on
    a camera without WidthMax/SensorWidth.
    """
    try:
        offset = getattr(nodes, offset_name)
        offset.value = int(offset.min)
    except Exception as e:
        logging.debug(f"{offset_name} not reset (not available or not writable): {e}")
    try:
        size = getattr(nodes, size_name)
        size.value = _snap_to_increment(size, sensor_extent(nodes, size_name, offset_name))
    except Exception as e:
        logging.warning(f"Could not set {size_name} to the full sensor: {e}")


def read_aoi(ia: ImageAcquirer) -> tuple[AoiLimits, tuple[int, int, int, int]]:
    """Read the camera's AOI constraints and its current AOI.

    Read-only (it runs while live): the sensor extent comes from
    sensor_extent().

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        tuple: (limits, (width, height, offset_x, offset_y)); offsets are 0
        on cameras without offset nodes.

    Raises:
        AttributeError: if the camera has no Width/Height node.
    """
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        axes = []
        current = []
        for size_name, offset_name in (("Width", "OffsetX"), ("Height", "OffsetY")):
            size = getattr(nodes, size_name)
            try:
                offset = getattr(nodes, offset_name)
                offset_value = int(offset.value)
            except Exception:
                offset, offset_value = None, 0
            sensor = sensor_extent(nodes, size_name, offset_name)
            axes.append(
                AoiAxis(
                    size_min=int(size.min),
                    size_inc=_node_inc(size),
                    sensor=sensor,
                    offset_min=int(offset.min) if offset is not None else 0,
                    offset_inc=_node_inc(offset) if offset is not None else 1,
                    has_offset=offset is not None,
                )
            )
            current.append((int(size.value), offset_value))
    (width, offset_x), (height, offset_y) = current
    return AoiLimits(*axes), (width, height, offset_x, offset_y)


def _roi_axis(
    a: float, b: float, frame_size: int, frame_offset: int, axis: AoiAxis
) -> tuple[int, int | None]:
    """Sensor (size, offset) of the span a..b of one axis of the current frame."""
    lo, hi = sorted((a, b))
    lo = min(max(lo, 0.0), float(frame_size))
    hi = min(max(hi, 0.0), float(frame_size))
    frame_end = frame_offset + frame_size  # the current frame, in sensor pixels
    # Every pixel the box touches, in sensor coordinates (end exclusive).
    start = frame_offset + math.floor(lo)
    end = frame_offset + math.ceil(hi)
    if axis.has_offset:
        offset = snap_value(start, axis.offset_min, axis.sensor, axis.offset_inc)
    else:
        offset = frame_offset  # pinned: the AOI always starts at the origin
    # Snapped down, so the AOI stays inside the selection - except where the
    # minimum size forces it to grow.
    room = max(axis.sensor - offset, axis.size_min)
    size = snap_value(end - offset, axis.size_min, room, axis.size_inc)
    if axis.has_offset and offset + size > frame_end:
        # Grown to the minimum size past the frame's edge: slide it back in.
        offset = snap_value(frame_end - size, axis.offset_min, axis.sensor, axis.offset_inc)
    return size, offset if axis.has_offset else None


def roi_to_sensor_aoi(
    box: tuple[float, float, float, float],
    frame_shape: tuple[int, int],
    frame_offset: tuple[int, int],
    limits: AoiLimits,
) -> tuple[int, int, int | None, int | None]:
    """Turn a box drawn on the current frame into a camera AOI.

    The box is clamped to the frame, shifted by the frame's own sensor
    offset (the frame may already be a cropped AOI), and snapped onto the
    camera's grid: offsets down onto the OffsetX/OffsetY grid, sizes down
    onto the Width/Height grid, so the AOI never reads out more than the box
    covers. A box below the minimum size grows to it, staying inside the
    frame. On a camera without offset nodes the AOI keeps starting at the
    origin and extends to the box's far edge.

    Args:
        box (tuple[float, ...]): (x0, y0, x1, y1) in pixels of the current
            frame - x along its columns, y along its rows; either corner may
            come first (drag in any direction).
        frame_shape (tuple[int, int]): The current frame's (height, width).
        frame_offset (tuple[int, int]): The current (OffsetX, OffsetY).
        limits (AoiLimits): The camera's AOI constraints, see read_aoi().

    Returns:
        tuple: (width, height, offset_x, offset_y) for set_resolution();
        offsets are None on cameras without the respective offset node.
    """
    x0, y0, x1, y1 = box
    frame_h, frame_w = frame_shape
    width, offset_x = _roi_axis(x0, x1, frame_w, frame_offset[0], limits.x)
    height, offset_y = _roi_axis(y0, y1, frame_h, frame_offset[1], limits.y)
    return width, height, offset_x, offset_y


def full_sensor_aoi(limits: AoiLimits) -> tuple[int, int, int | None, int | None]:
    """The AOI that reads out the whole sensor: the sensor extent at offset 0.

    Args:
        limits (AoiLimits): The camera's AOI constraints, see read_aoi().

    Returns:
        tuple: (width, height, offset_x, offset_y) for set_resolution();
        offsets are None on cameras without the respective offset node.
    """
    sizes = [
        snap_value(axis.sensor, axis.size_min, max(axis.sensor, axis.size_min), axis.size_inc)
        for axis in (limits.x, limits.y)
    ]
    offsets = [axis.offset_min if axis.has_offset else None for axis in (limits.x, limits.y)]
    return sizes[0], sizes[1], offsets[0], offsets[1]


# Node names worth logging around an AOI change, on the camera's frame grabber
# (the GenTL device module) and on the data stream.
AOI_NODE_PATTERN = re.compile(r"width|height|payload|offset|roi|aoi", re.IGNORECASE)


def matching_node_values(node_map, pattern: re.Pattern = AOI_NODE_PATTERN) -> dict:
    """Values of the readable nodes of *node_map* whose names match *pattern*.

    Read-only and best effort: nodes that cannot be read are left out, and a
    missing node map gives an empty dict.
    """
    values = {}
    if node_map is None:
        return values
    try:
        names = dir(node_map)
    except Exception:
        return values
    for name in names:
        if name.startswith("_") or not pattern.search(name):
            continue
        try:
            values[name] = getattr(node_map, name).value
        except Exception:
            continue
    return values


def aoi_snapshot(ia: ImageAcquirer) -> dict[str, dict]:
    """What the camera, its frame grabber and the data streams report about the AOI.

    For the log of every AOI change: a frame grabber whose own image format
    does not follow the camera's AOI shows up as a mismatch between the
    camera's PayloadSize and the stream's payload_size, which harvesters
    sizes the announced buffers from whenever defines_payload_size() is true.
    Read-only; never raises (unreadable values are left out).

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        dict: {"camera": {...}, "grabber": {...}, "stream0": {...}, ...}.
    """
    snapshot: dict[str, dict] = {}
    with IA_LOCK:
        camera = {}
        for name in ("Width", "Height", "OffsetX", "OffsetY", "PayloadSize"):
            try:
                camera[name] = getattr(ia.remote_device.node_map, name).value
            except Exception:
                continue
        snapshot["camera"] = camera
        try:
            snapshot["grabber"] = matching_node_values(ia.device.node_map)
        except Exception:
            snapshot["grabber"] = {}
        try:
            streams = list(ia.data_streams)
        except Exception:
            streams = []
        for i, stream in enumerate(streams):
            entry = {}
            try:
                entry.update(matching_node_values(stream.node_map))
            except Exception:
                pass
            try:
                defines = bool(stream.defines_payload_size())
                entry["defines_payload_size()"] = defines
                if defines:
                    entry["payload_size"] = int(stream.payload_size)
            except Exception as e:
                entry["payload_size"] = f"unavailable ({type(e).__name__})"
            snapshot[f"stream{i}"] = entry
    return snapshot


def describe_aoi_state(ia: ImageAcquirer) -> str:
    """aoi_snapshot() as one log line, e.g. ``camera: Width=512, ...; stream0: ...``."""
    try:
        snapshot = aoi_snapshot(ia)
    except Exception as e:  # never let logging break an AOI change
        return f"unavailable ({e})"
    return "; ".join(
        f"{part}: " + (", ".join(f"{name}={value}" for name, value in values.items()) or "-")
        for part, values in snapshot.items()
    )


def _user_config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "harvesters_pg"


# The applied AOI per camera. It used to be one global entry next to the code:
# read-only in a system-wide install (Apply then raised PermissionError), and
# one camera's AOI was applied, clamped, to every other camera.
RESOLUTION_CONFIG_PATH = _user_config_dir() / "camera_resolution.json"


def camera_config_key(ia: ImageAcquirer) -> str:
    """Identify the connected camera for per-camera settings (vendor/model/serial)."""
    parts = []
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        for name in ("DeviceVendorName", "DeviceModelName", "DeviceSerialNumber"):
            try:
                parts.append(str(getattr(nodes, name).value))
            except Exception:
                pass
    if not parts:
        try:
            parts.append(str(ia.device.id_))
        except Exception:
            parts.append("unknown camera")
    return "/".join(parts)


def _read_resolution_file() -> dict:
    try:
        data = json.loads(RESOLUTION_CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_resolution_config(camera: str) -> dict:
    """Load the resolution persisted for *camera*, if any.

    Args:
        camera (str): The camera's key, see camera_config_key().

    Returns:
        dict: Mapping with optional keys width, height, offset_x, offset_y.
    """
    entry = _read_resolution_file().get(camera)
    return entry if isinstance(entry, dict) else {}


def save_resolution_config(camera: str, resolution: dict) -> None:
    """Persist *camera*'s resolution so it can be reapplied on the next connect.

    Args:
        camera (str): The camera's key, see camera_config_key().
        resolution (dict): Mapping containing width/height (and optionally the
            sensor offsets). Non-integer entries are dropped.

    Raises:
        OSError: if the configuration file cannot be written.
    """
    data = _read_resolution_file()
    data[camera] = {k: v for k, v in resolution.items() if isinstance(v, int)}
    RESOLUTION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESOLUTION_CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")