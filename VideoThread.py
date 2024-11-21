#!/usr/bin/env python3
from PyQt6 import QtCore
import cam
import numpy as np
import cv2
from typing import Callable
import time


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
        cap (cv2.VideoCapture | None): The video capture
    """

    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, source) -> None:
        super().__init__()
        self._run_flag = True
        self.source: str = source
        self.image_functions = {
            "Webcam": cam.get_webcam_image,
            "random": cam.get_random_image,
        }
        self.image_func: Callable = self.image_functions[self.source.split(" ")[0]]
        self.cap: cv2.VideoCapture | None = (
            cam.setup_webcam(int(self.source.split(" ")[1]))
            if self.source != "random"
            else None
        )

    def __del__(self) -> None:
        if type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        elif type(self.cap) is None:
            pass
        # TODO: add harvester

    def run(self) -> None:
        """Capture video frames and emit the frame to the main thread."""

        while self._run_flag:
            frame = self.image_func(self.cap)
            self.change_pixmap_signal.emit(frame)
            time.sleep(0.03)  # Add a small delay to control frame rate

    def stop(self) -> None:
        self._run_flag = False
        self.wait()

    def resume(self) -> None:
        self._run_flag = True
