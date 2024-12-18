"""The model to handle interaction with the camera."""

from harvesters.core import Harvester, ImageAcquirer
import cv2
import numpy as np
import os
from pathlib import Path
import glob
import logging


def find_common_cti_files() -> list[str]:
    """Find the common *.cti files for the camera.

    Returns:
        list[str]: A list of paths to the *.cti files.
    """
    logging.info("Looking for *.cti files...")
    ctis = []
    env_var = os.getenv("GENICAM_GENTL64_PATH")
    if env_var:
        ctis.extend(glob.glob(f"{env_var}/*.cti"))
    cti = (
        r"C:\Program Files\Active Silicon\FireBird CXP\GenTL\win64\TLActiveSilicon.cti"
    )
    if Path(cti).exists():
        ctis.append(cti)
    cti = "/usr/local/activesilicon/lib64/TLActiveSilicon.cti"
    if Path(cti).exists():
        ctis.append(cti)
    logging.info(f"Found *.cti files: {ctis}")
    return ctis


def setup_cam(
    cti: str = r"C:\Program Files\Active Silicon\FireBird CXP\GenTL\win64\TLActiveSilicon.cti",
    height: int = 1080,
    width: int = 1920,
    fps: int = 30,
    blacklevel: int = 255,
    exposure: int = 5000,
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
    max_fps = nodes.AcquisitionFrameRateMax.value
    if fps > max_fps:
        fps = max_fps
        logging.warning(f"FPS is too high for the camera. Setting it to the maximum possible: {fps}")
    nodes.AcquisitionFrameRate.value = fps

    max_exposure = int((1.0/fps)*1e6 - 1)
    if exposure >= max_exposure:
        exposure = int(max_exposure)
        logging.warning(f"Exposure time is too high for given FPS. Setting it to the maximum: {exposure}")

    nodes.ExposureTime.set_value(exposure)

    # Allow the ImageAcquire object to start image acquisition:
    nodes.Height.value = height
    nodes.Width.value = width

    ia.start()
    nodes.BlackLevel.value = blacklevel
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


def setup_webcam(idx: int) -> cv2.VideoCapture:
    """Set the webcam up using OpenCV.

    Args:
        idx (int): The index of the webcam.

    Returns:
        cv2.VideoCapture: The video capture object.
    """
    cap = cv2.VideoCapture(idx)
    return cap


def return_harvester_cameras(
    ctis: list[str] = [r"C:\Program Files\Active Silicon\FireBird CXP\GenTL\win64\TLActiveSilicon.cti"],
) -> list[dict]:
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
            cap = cv2.VideoCapture(i)
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


def get_webcam_image(cap: cv2.VideoCapture) -> np.ndarray:
    """Get an image from the webcam.

    Args:
        cap (cv2.VideoCapture): The video capture object.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).T
    return gray


def get_random_image(_: None) -> np.ndarray:
    """Get a random image.

    Args:
        _: None

    Returns:
        np.ndarray: A random image.
    """
    return np.random.randint(0, 255, (1920, 1080), dtype=np.uint8)


def get_harvester_image(ia: ImageAcquirer) -> np.ndarray:
    """
    Get the image from the camera using the Harvester API.

    Args:
        ia (ImageAcquirer): The ImageAcquirer object.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    with ia.fetch() as buffer:
        component = buffer.payload.components[0]
        img = component.dataimg = buffer.payload.components[0].data
        img = np.reshape(img, (buffer.payload.components[0].height, buffer.payload.components[0].width)).T
    return img


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
