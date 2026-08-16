#!/usr/bin/env python3
import time
from typing import Callable

import cv2
import numpy as np
from harvesters.core import Harvester, ImageAcquirer
from PyQt6 import QtCore
import logging

import cam


class VideoThread(QtCore.QThread):
    """Worker thread for capturing video frames.

    Args:
        source (str): The source of the video feed. It can be either "Webcam {index}", "random", or any Harvester camera.

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

    def __init__(self, source, ctis) -> None:
        super().__init__()
        self.source: str = source
        self._run_flag: bool = False
        self.image_functions = {
            "Webcam": cam.get_webcam_image,
            "random": cam.get_random_image,
        }
        try:
            self.image_func = self.image_functions[self.source.split(" ")[0]]
        except KeyError:
            self.image_func = cam.get_harvester_image
        self.cap: cv2.VideoCapture | ImageAcquirer | None = None
        self.harvester: Harvester | None = None
        self.set_cap(ctis)

    def set_cap(self, ctis) -> None:
        if self.source == "random":
            self.cap = None
        elif "Webcam" in self.source:
            self.cap = cam.setup_webcam(int(self.source.split(" ")[1]))
        else:
            self.harvester, self.cap = cam.setup_cam(cti=ctis[0])

    def _stop_and_join(self) -> None:
        """Signal the run loop to exit and wait for the worker thread to finish.

        Unblocks a fetch() waiting on the camera first, then joins. Only after
        the worker has actually exited is it safe to stop/destroy the camera,
        otherwise teardown races the driver's DMA buffers and can hang the machine.
        """
        self._run_flag = False
        if not self.isRunning():
            return
        # Unblock a fetch() that is waiting for a frame, then join.
        if type(self.cap) is ImageAcquirer:
            try:
                self.cap.stop()
            except Exception as e:
                logging.debug(f"Error stopping camera while joining thread: {e}")
        deadline = time.monotonic() + 10.0
        while self.isRunning() and time.monotonic() < deadline:
            self.wait(100)
        if self.isRunning():
            logging.warning("VideoThread worker did not stop within 10s")

    def __del__(self) -> None:
        self._stop_and_join()
        if type(self.cap) is cv2.VideoCapture:
            self.cap.release()
        elif type(self.cap) is None:
            logging.debug("Cap was none")
        else:
            # Only teardown the camera once the worker has actually exited, so we
            # never destroy DMA buffers while the producer is still streaming.
            if self.cap is not None and not self.isRunning():
                self.cap.stop()
                self.cap.destroy()
            if self.harvester is not None:
                self.harvester.reset()
                self.harvester = None

    def run(self) -> None:
        """Capture video frames and emit the frame to the main thread."""

        while self._run_flag:
            try:
                frame = self.image_func(self.cap)
                if frame is not None:
                    self.change_pixmap_signal.emit(frame)
                time.sleep(1/30) # 30 fps
            except Exception as e:
                print(e)


    def stop(self) -> None:
        self._stop_and_join()

    def _start(self) -> None:
        self._run_flag = True
        if type(self.cap) is ImageAcquirer:
            self.cap.start()
        super().start()
