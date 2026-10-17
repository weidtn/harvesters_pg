"""
The model to handle interaction with the camera.
"""

from PyQt6.QtWidgets import QProgressBar
from harvesters.core import Harvester, ImageAcquirer
from cv2 import VideoCapture
import cv2
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
    disconnect_cam().
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
) -> tuple[Harvester, ImageAcquirer]:
    """Set the camera up using the Harvester API.

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

    Returns:
        tuple[Harvester, ImageAcquirer]: The Harvester object and the ImageAcquirer object.

    Raises:
        Exception: whatever the producer raised. The camera is always released
            before the exception leaves this function.
    """
    ctis = [cti] if isinstance(cti, str) else list(cti)
    # The process-wide Harvester, with the producers already loaded.
    h = get_harvester(ctis)
    ia = None
    try:
        # Enumerate only when the cached device list cannot serve the request.
        device = _find_device(h, search_key)
        enumerated = device is None
        if enumerated:
            h.update()
            device = _find_device(h, search_key)

        # Open exactly the device that was picked. Opening index 0 regardless
        # of what the user picked meant a second camera (or a camera behind a
        # second producer) silently opened the wrong device - possibly one
        # another process already holds. With nothing found, harvesters raises
        # its own "no device" error from the original key.
        fallback_key = 0 if search_key is None else search_key
        try:
            ia = h.create(fallback_key if device is None else device)
        except Exception:
            if enumerated:
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
        _configure_cam(ia, height, width, fps, blacklevel, exposure, start=start)
        if start:
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
        teardown_cam(None, ia)
        raise
    return h, ia


def _configure_cam(
    ia: ImageAcquirer,
    height: int | None,
    width: int | None,
    fps: int,
    blacklevel: int,
    exposure: int,
    start: bool = True,
) -> None:
    """Apply the node-map settings to a fresh camera, then optionally start it.

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

    if height is not None:
        try:
            nodes.Height.value = _snap_to_increment(nodes.Height, height)
        except Exception as e:
            logging.warning(f"Could not set Height to {height}: {e}")
    else:
        try:
            nodes.Height.value = nodes.HeightMax.value
        except Exception as e:
            logging.debug(f"HeightMax not available, keeping camera Height: {e}")

    if width is not None:
        try:
            nodes.Width.value = _snap_to_increment(nodes.Width, width)
        except Exception as e:
            logging.warning(f"Could not set Width to {width}: {e}")
    else:
        try:
            nodes.Width.value = nodes.WidthMax.value
        except Exception as e:
            logging.debug(f"WidthMax not available, keeping camera Width: {e}")

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

    if start:
        ia.start()
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
    """Release a camera and unload its GenTL producer, best effort.

    Each step is guarded on its own: a raising ia.stop() must not skip
    ia.destroy(), and a raising ia.destroy() must not skip h.reset(). Skipping
    any of them leaves the device exclusively opened or the producer loaded,
    which is exactly what makes the *next* connect hang.

    Callers must make sure no thread is still inside a fetch() on ``ia``:
    resetting the harvester under an in-flight fetch unloads the producer out
    from under it (see VideoThread.close).

    Args:
        h (Harvester | None): The Harvester object, if any.
        ia (ImageAcquirer | None): The ImageAcquirer object, if any.

    Returns:
        None
    """
    if ia is not None:
        try:
            logging.debug("teardown: stopping acquisition")
            ia.stop()
        except Exception as e:
            logging.debug(f"Error stopping camera during teardown: {e}")
        try:
            logging.debug("teardown: destroying the image acquirer")
            ia.destroy()
        except Exception as e:
            logging.warning(f"Error destroying camera during teardown: {e}")
    if h is not None:
        _forget_shared_harvester(h)
        try:
            # This unloads the GenTL producer library. If the app ever hangs
            # here, the log stops after this line: some vendor producers keep
            # threads or device handles alive across a reset and do not
            # survive being unloaded and reloaded per connect.
            logging.debug("teardown: resetting the harvester (unloads the producer)")
            h.reset()
        except Exception as e:
            logging.warning(f"Error resetting harvester during teardown: {e}")
    logging.debug("teardown: done")


def disconnect_cam(h: Harvester, ia: ImageAcquirer) -> None:
    """Disconnect the camera.

    Args:
        h (Harvester): The Harvester object.
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        None
    """
    teardown_cam(h, ia)


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
        ia.start()
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
            try:
                ia.stop()
            except Exception as e:
                logging.warning(f"Error stopping camera after recording: {e}")
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
    lo = int(node.min)
    hi = int(node.max)
    try:
        inc = int(node.inc)
    except (AttributeError, TypeError):
        inc = 1
    if inc < 1:
        inc = 1
    value = max(min(int(value), hi), lo)
    snapped = lo + ((value - lo) // inc) * inc
    return max(min(snapped, hi), lo)


def set_resolution(
    ia: ImageAcquirer,
    width: int | None = None,
    height: int | None = None,
    offset_x: int | None = None,
    offset_y: int | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    """Set the camera AOI (Width/Height and sensor offsets) safely.

    Many GenICam producers refuse to write Width/Height (and offsets) while
    acquisition is running - TLSimu raises an AccessException - so the camera
    is stopped for the writes and restarted only if it was acquiring. Values
    are snapped to each node's increment and clamped to [min, max]; nodes the
    camera does not expose (e.g. offsets on TLSimu) are skipped.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        width (int | None): New Width, or None to leave unchanged.
        height (int | None): New Height, or None to leave unchanged.
        offset_x (int | None): New OffsetX, or None to leave unchanged.
        offset_y (int | None): New OffsetY, or None to leave unchanged.

    Returns:
        tuple[int | None, ...]: The applied (width, height, offset_x, offset_y).
    """
    with IA_LOCK:
        nodes = ia.remote_device.node_map
        was_running = bool(ia.is_acquiring())
        if was_running:
            ia.stop()
        try:
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
        finally:
            if was_running:
                ia.start()
    return tuple(applied)


def _write_snapped(nodes, node_name: str, value: int) -> None:
    """Write *value*, snapped to the node's grid; log instead of raising."""
    try:
        node = getattr(nodes, node_name)
        node.value = _snap_to_increment(node, value)
    except Exception as e:
        logging.warning(f"Could not set {node_name} to {value}: {e}")


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