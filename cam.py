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

# Serializes all access to the GenICam device (fetch + node_map reads/writes).
# The worker thread fetches frames while the GUI thread polls/writes node values;
# GenICam producers are not thread-safe, so concurrent device access from two
# threads can corrupt the producer's state and hang the machine.
IA_LOCK = threading.RLock()


def load_cti_envvar() -> list[str]:
    """Find the common *.cti files for the camera.

    Returns:
        list[str]: A list of paths to the *.cti files.
    """
    logging.info("Looking for *.cti files...")
    ctis = []
    env_var = os.getenv("GENICAM_GENTL64_PATH")
    if env_var:
        ctis.extend(glob.glob(f"{env_var}/*.cti"))
    logging.info(f"Found *.cti files: {ctis}")
    return ctis


def setup_cam(
    cti: str,
    height: int | None = None,
    width: int | None = None,
    fps: int = 30,
    blacklevel: int = 125,
    exposure: int = 999,
) -> tuple[Harvester, ImageAcquirer]:
    """Set the camera up using the Harvester API.

    Args:
        cti (str): The path to the GenTL Producer file.
        height (int): The height of the image.
        width (int): The width of the image.
        fps (int): The frames per second.
        blacklevel (int): The black level of the image.
        exposure (int): The exposure time of the image.

    Returns:
        tuple[Harvester, ImageAcquirer]: The Harvester object and the ImageAcquirer object.

    """
    # Create a Harvester object:
    h = Harvester()
    # Load a GenTL Producer; you can load many more if you want to:
    h.add_file(cti)
    # Enumerate the available devices that GenTL Producers can handle:
    h.update()

    # Select a target device and create an ImageAcquire object
    ia = h.create(0)

    nodes = ia.remote_device.node_map
    # Check if the FPS are okay:
    try:
        max_fps = nodes.AcquisitionFrameRateMax.value
    except Exception as e:
        # e.g. TLSimu or cameras without such a node
        logging.warning(f"AcquisitionFrameRateMax not available, skipping clamp: {e}")
        max_fps = None
    if max_fps is not None and fps > max_fps:
        fps = max_fps
        logging.warning(
            f"FPS is too high for the camera. Setting it to the maximum possible: {fps}"
        )
    try:
        nodes.AcquisitionFrameRate.value = fps
    except Exception as e:
        logging.warning(f"Could not set AcquisitionFrameRate to {fps}: {e}")

    exposure = cap_at_max_exposure(exposure, fps)
    try:
        nodes.ExposureTime.set_value(exposure)
    except Exception as e:
        print(e)

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

    ia.start()
    try:
        logging.info(f"Camera pixel format: {nodes.PixelFormat.value}")
    except Exception as e:
        logging.debug(f"PixelFormat node not available: {e}")
    try:
        nodes.BlackLevel.value = blacklevel
    except Exception as e:
        logging.warning(f"Could not set BlackLevel to {blacklevel}: {e}")
    warmup_cam(ia)
    return h, ia


def warmup_cam(ia, frames: int = 10) -> None:
    """Warm up the camera by collecting a few frames.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        frames (int): The number of frames to collect.

    Returns:
        None
    """
    logging.info(f"Warming up camera by collecting {frames} frames.")
    for _ in range(frames):
        buffer = ia.fetch()
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
    h = Harvester()
    [h.add_file(cti) for cti in ctis]
    # Enumerate the available devices that GenTL Producers can handle:
    h.update()
    devices = [cam.property_dict for cam in h.device_info_list]
    h.reset()
    return devices

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
        if cap.read()[0]:
            arr.append(i)
            cap.release()
    return arr

def disconnect_cam(h: Harvester, ia: ImageAcquirer) -> None:
    """Disconnect the camera.

    Args:
        h (Harvester): The Harvester object.
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        None
    """
    ia.stop()
    ia.destroy()
    h.reset()

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

_BAYER_DEMOSAIC = {
    "GR": cv2.COLOR_BayerGR2GRAY,
    "RG": cv2.COLOR_BayerRG2GRAY,
    "GB": cv2.COLOR_BayerGB2GRAY,
    "BG": cv2.COLOR_BayerBG2GRAY,
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

    # 2) YUV 4:2:2 / 4:1:1 / 4:4:4: luma extraction from the packed byte order.
    if up == "YUV422PACKED":  # UYVY: luma at the odd byte of each pair
        if data.size != 2 * h * w:
            raise ValueError(f"{fmt}: expected {2 * h * w} bytes, got {data.size}")
        return data.reshape(h, 2 * w)[:, 1::2].copy()
    if up == "YUV444PACKED":  # Y,U,V per pixel; luma is the first byte
        if data.size != 3 * h * w:
            raise ValueError(f"{fmt}: expected {3 * h * w} bytes, got {data.size}")
        return data.reshape(h, w, 3)[..., 0].copy()
    if up == "YUV411PACKED":  # U,Y0,Y1,V,Y2,Y3 per 4 pixels
        if (h * w) % 4 or data.size != (h * w // 4) * 6:
            raise ValueError(
                f"{fmt}: expected {(h * w // 4) * 6} bytes for {w}x{h}, got {data.size}"
            )
        return data.reshape(h * w // 4, 6)[:, [1, 2, 4, 5]].reshape(h, w).copy()

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

def get_harvester_image(ia: ImageAcquirer) -> np.ndarray:
    """
    Get the image from the camera using the Harvester API.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    with IA_LOCK:
        with ia.fetch() as buffer:
            if buffer:
                component = buffer.payload.components[0]
                # See component_to_grayscale8: Mono8 stays a plain copy, while
                # 10/12/16-bit, Bayer and color formats are converted to a
                # grayscale uint8 frame here so the whole app (display,
                # recording, export) only ever deals with natural (h, w).
                img = component_to_grayscale8(component)
            else:  # If no buffer is available, return a black image
                img = np.zeros(
                    (
                        ia.remote_device.node_map.Height.value,
                        ia.remote_device.node_map.Width.value,
                    ),
                    dtype=np.uint8,
                )
    return img


def _harvester_frame_into(ia: ImageAcquirer, out: np.ndarray) -> None:
    """Fetch one frame from the camera directly into a preallocated buffer.

    Writes the (height, width) C-contiguous frame into ``out`` with a single
    linear copy instead of allocating a new array per frame, which is what the
    hot recording path needs to sustain high frame rates.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.
        out (np.ndarray): The output buffer of shape (height, width).
    """
    with IA_LOCK:
        with ia.fetch() as buffer:
            if buffer:
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
            else:  # If no buffer is available, record a black frame
                out.fill(0)

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
    """
    if progress is not None:
        if n_frames == 1:
            progress.setMaximum(1)
        else:
            progress.setMaximum(n_frames - 1)
    width = ia.remote_device.node_map.Width.value
    height = ia.remote_device.node_map.Height.value
    # Preallocate the full recording once as (n_frames, height, width) and
    # copy each frame in place. Every per-frame step is a linear memcpy; the
    # old append-a-list-then-stack path re-ran the slow strided transpose per
    # frame (~10 ms at 2048x2048), which capped recording well below the
    # camera's nominal frame rate.
    frames = np.empty((n_frames, height, width), dtype=np.uint8)
    ia.start()
    for i in trange(n_frames):
        _harvester_frame_into(ia, frames[i])
        if preview is not None:
            preview(frames[i])
        if progress is not None:
            if n_frames == 1:
                progress.setValue(1)
            else:
                progress.setValue(i)
    ia.stop()
    return frames

def get_all_attributes_harvesters(ia: ImageAcquirer) -> dict:
    """Get all attributes of the camera.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        dict: A dictionary containing all attributes of the camera.
    """
    nodes = ia.remote_device.node_map
    dict = {}
    for node in dir(nodes):
        try:
            dict[node] = getattr(ia.remote_device.node_map, node).value
        except Exception as e:
            logging.debug(f"Error while getting node {node}'s value: {e}")
    return dict

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
    pixels = img.flatten()[:4].astype(np.uint32)

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
    pixels = img.flatten()[4:8].astype(np.uint32)

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
            for node_name, value in (
                ("OffsetX", offset_x),
                ("OffsetY", offset_y),
                ("Width", width),
                ("Height", height),
            ):
                if value is None:
                    continue
                try:
                    node = getattr(nodes, node_name)
                    node.value = _snap_to_increment(node, value)
                except Exception as e:
                    logging.warning(f"Could not set {node_name} to {value}: {e}")
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


RESOLUTION_CONFIG_PATH = Path(__file__).resolve().parent / "camera_resolution.json"


def load_resolution_config() -> dict:
    """Load the persisted camera resolution, if any.

    Returns:
        dict: Mapping with optional keys width, height, offset_x, offset_y.
    """
    try:
        data = json.loads(RESOLUTION_CONFIG_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_resolution_config(resolution: dict) -> None:
    """Persist the camera resolution so it can be reapplied on the next connect.

    Args:
        resolution (dict): Mapping containing width/height (and optionally the
            sensor offsets). Non-integer entries are dropped.
    """
    data = {k: v for k, v in resolution.items() if isinstance(v, int)}
    RESOLUTION_CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")