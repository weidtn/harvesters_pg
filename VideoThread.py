#!/usr/bin/env python3
from PyQt6 import QtCore
import cam
import numpy as np
import cv2
from typing import Callable
import time
from harvesters.core import Harvester, ImageAcquirer


class VideoThread(QtCore.QThread):
    """Worker thread for capturing video frames.

    Args:
        source (str): The source of the video feed. It can be either "Webcam {index}" or "random"

    Attributes:
        change_pixmap_signal (QtCore.pyqtSignal): Signal to emit the frame to the main thread
        _run_flag (bool): Flag to control the execution of the thread
        source (str): The source of the video feed
        image_functions (dict): Dictionary of functions to get video frames
        image_func (Callable): The function to get video frames
        cap (cv2.VideoCapture | ImageAcquirer | None): The video capture
    """

    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)
    acquisition_finished_signal = QtCore.pyqtSignal()

    def __init__(self, source) -> None:
        super().__init__()
        self._run_flag: bool = True
        self.source: str = source
        self.image_functions = {
            "Webcam": cam.get_webcam_image,
            "random": cam.get_random_image,
            "SVS-Vistek": cam.get_harvester_image,
        }
        self.image_func: Callable = self.image_functions[self.source.split(" ")[0]]
        self.cap: cv2.VideoCapture | ImageAcquirer | None = None
        self.harvester: Harvester | None = None
        self.set_cap()

    def set_cap(self) -> None:
        if self.source == "random":
            self.cap = None
        elif "Webcam" in self.source:
            self.cap = cam.setup_webcam(int(self.source.split(" ")[1]))
        else:
            # TODO: Manage the cti files better
            cti = cam.find_common_cti_files()[0]
            self.harvester, self.cap = cam.setup_cam(cti=cti)

    def __del__(self) -> None:
        if type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        elif type(self.cap) is None:
            pass
        else:
            if self.harvester is not None:
                self.harvester.reset()
                self.harvester = None
            if self.cap is not None:
                self.cap.stop()
                self.cap.destroy()

    def run(self) -> None:
        """Capture video frames and emit the frame to the main thread."""

        while self._run_flag:
            try:
                frame = self.image_func(self.cap)
                self.change_pixmap_signal.emit(frame)
                time.sleep(1/30) # 30 fps
            except Exception as e:
                print(e)


    def stop(self) -> None:
        self._run_flag = False

    def resume(self) -> None:
        self._run_flag = True

