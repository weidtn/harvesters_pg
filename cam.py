"""
The model to handle interaction with the camera
"""

from harvesters.core import Harvester, ImageAcquirer
import cv2
import numpy as np
import os
from pathlib import Path
import glob


def find_common_cti_files() -> list[str]:
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
    return ctis


def setup_cam(
    cti: str = r"C:\Program Files\Active Silicon\FireBird CXP\GenTL\win64\TLActiveSilicon.cti",
    height: int = 1080,
    width: int = 1920,
    fps: int = 30,
    blacklevel: int = 255,
) -> tuple[Harvester, ImageAcquirer]:
    """Setup the camera using the Harvester API.


    Args:
        cti (str): The path to the GenTL Producer file.
        height (int): The height of the image.
        width (int): The width of the image.
        fps (int): The frames per second.
        blacklevel (int): The black level of the image.

    Returns:
        tuple[Harvester, ImageAcquirer]: The Harvester object and the ImageAcquirer object.

    """
    # Create a Harvester object:
    h = Harvester()
    # Load a GenTL Producer; you can load many more if you want to:
    h.add_file(cti)
    # Enumerate the available devices that GenTL Producers can handle:
    h.update()

    # Select a target device and create an ImageAcquire object that
    # controls the device:
    ia = h.create(0)

    # Configure the target device; it looks very small but this is just
    # for demonstration:
    # ia.remote_device.node_map.Width.value = 1456
    # ia.remote_device.node_map.Height.value = 1088
    # ia.remote_device.node_map.PixelFormat.symbolics
    # ia.remote_device.node_map.PixelFormat.value = 'Mono8'

    # ia.remote_device.node_map.ChunkSelector.value = 'ExposureTime'
    # ia.remote_device.node_map.ExposureTime.set_value(100000.0)

    # Allow the ImageAcquire object to start image acquisition:
    # print(dir(ia.device.node_map))
    ia.device.node_map.Height.value = height
    ia.device.node_map.Width.value = width
    ia.start()
    nodes = ia.remote_device.node_map
    nodes.AcquisitionFrameRate.value = fps
    nodes.BlackLevel.value = blacklevel
    return h, ia


def setup_webcam(idx: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(idx)
    return cap


def returnHarvesterCameras(
    cti: str = r"C:\Program Files\Active Silicon\FireBird CXP\GenTL\win64\TLActiveSilicon.cti",
) -> list[dict]:
    h = Harvester()
    h.add_file(cti)
    # Enumerate the available devices that GenTL Producers can handle:
    h.update()
    devices = [cam.property_dict for cam in h.device_info_list]
    return devices


def returnCameraIndices() -> list[int]:
    # checks the first 10 indices for opencv detectable cameras.
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
    ia.stop()
    ia.destroy()
    h.reset()


def get_webcam_image(cap: cv2.VideoCapture) -> np.ndarray:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).T
    return gray


def get_random_image(_: None) -> np.ndarray:
    return np.random.randint(0, 255, (1920, 1080), dtype=np.uint8)
