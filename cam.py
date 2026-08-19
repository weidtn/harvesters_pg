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
import logging
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
            nodes.Height.value = height
        except Exception as e:
            logging.warning(f"Could not set Height to {height}: {e}")
    else:
        try:
            nodes.Height.value = nodes.HeightMax.value
        except Exception as e:
            logging.debug(f"HeightMax not available, keeping camera Height: {e}")

    if width is not None:
        try:
            nodes.Width.value = width
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
                # Copy: component.data is a read-only view of the producer's DMA
                # buffer, which is reused for later frames. Copying it (row-major,
                # linear memcpy) snapshots the frame so the buffer can be reused,
                # and yields an owned (height, width) C-contiguous array.
                img = np.frombuffer(
                    component.data, dtype=np.uint8
                ).reshape(component.height, component.width).copy()
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
                np.copyto(
                    out,
                    np.frombuffer(component.data, dtype=np.uint8).reshape(
                        component.height, component.width
                    ),
                )
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