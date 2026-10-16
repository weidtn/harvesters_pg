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
            if not ctis:
                raise RuntimeError(
                    "No GenTL producer (*.cti) available; set "
                    "GENICAM_GENTL64_PATH or pick one via "
                    "File > Select GenTL Producer."
                )
            # Hand over every known producer and select by display name. Using
            # ctis[0] with device index 0 ignored the source the user picked,
            # so a second camera - or one behind a second producer - silently
            # opened the wrong device, possibly one already held elsewhere.
            self.harvester, self.cap = cam.setup_cam(
                cti=ctis, search_key={"display_name": self.source}
            )

    def _stop_and_join(self) -> None:
        """Signal the run loop to exit and wait for the worker thread to finish.

        ia.stop() does NOT reliably unblock a fetch() that is already blocked
        waiting for a frame in another thread: harvesters' fetch() wait loop
        never checks is_acquiring(), so a worker stuck between frames can stay
        blocked well past stop() returning. We still call stop() (it's the
        best available nudge and is a no-op if the camera isn't acquiring),
        but the join below is what actually determines whether the worker
        exited - callers must not assume teardown is safe just because this
        method returned.
        """
        self._run_flag = False
        # Stop acquisition regardless of whether the worker thread ever
        # started: setup_cam() leaves a freshly connected Harvester camera
        # acquiring even before Start is pressed.
        if type(self.cap) is ImageAcquirer:
            try:
                self.cap.stop()
            except Exception as e:
                logging.debug(f"Error stopping camera while joining thread: {e}")
        if not self.isRunning():
            return
        deadline = time.monotonic() + 10.0
        while self.isRunning() and time.monotonic() < deadline:
            self.wait(100)
        if self.isRunning():
            logging.warning("VideoThread worker did not stop within 10s")

    def close(self) -> None:
        """Stop the worker and release the camera. Safe to call repeatedly.

        This teardown used to live in __del__, which handed its timing to
        CPython's refcounting and cyclic GC: the device could stay exclusively
        opened long after the UI said "disconnected", so a quick reconnect ran
        into a camera the producer still held. Callers release the camera by
        calling this explicitly (see MainWindow.disconnect_video).
        """
        logging.debug("close(): stopping acquisition and joining the worker")
        self._stop_and_join()
        if self.isRunning():
            # Free nothing the stuck worker may still be inside. Resetting the
            # harvester unloads the GenTL producer out from under an in-flight
            # fetch(); releasing a VideoCapture frees buffers out from under an
            # in-flight read(). Either one can crash the process or leave the
            # device wedged. Leak the handles and log loudly instead.
            logging.warning(
                "VideoThread worker still running at teardown; leaking the "
                "camera handles instead of releasing them out from under the "
                "stuck call."
            )
            return
        cap, self.cap = self.cap, None
        # Drop the reference without resetting: the Harvester is shared for the
        # whole session (see cam.get_harvester), so unloading its producers per
        # disconnect is exactly the churn we are avoiding. MainWindow.closeEvent
        # releases it via cam.release_harvester().
        self.harvester = None
        if cap is None:
            logging.debug("Cap was none")
        elif type(cap) is cv2.VideoCapture:
            cap.release()
        else:
            cam.teardown_cam(None, cap)

    def __del__(self) -> None:
        # Backstop only. Finalizers run at unpredictable times and swallow
        # exceptions, so nothing may depend on this; close() is the real
        # teardown and has already run on the normal path, which leaves this a
        # cheap no-op.
        try:
            self.close()
        except Exception as e:
            logging.debug(f"Error during VideoThread finalization: {e}")

    def run(self) -> None:
        """Capture video frames and emit the frame to the main thread."""

        while self._run_flag:
            try:
                # Acquire the shared device lock so node-map writes from the GUI
                # thread never race this fetch (GenICam producers aren't
                # thread-safe; concurrent access can corrupt/hang the driver).
                with cam.IA_LOCK:
                    frame = self.image_func(self.cap)
                if frame is not None:
                    self.change_pixmap_signal.emit(frame)
            except Exception as e:
                logging.warning(f"Error while capturing a frame: {e}")
            # Outside the try on purpose. With the sleep inside it, any error
            # from image_func skipped the pacing and turned this into a tight
            # loop: it pegged a core, flooded the log, and cycled IA_LOCK fast
            # enough to starve the GUI thread of it.
            time.sleep(1 / 30)  # 30 fps


    def stop(self) -> None:
        self._stop_and_join()

    def _start(self) -> bool:
        """Start acquisition and the worker thread.

        Returns:
            bool: False if the previous worker is still running, in which case
                nothing was started. stop() can time out without the worker
                actually exiting (see _stop_and_join), and calling start() on
                the ImageAcquirer that worker is still fetching from means two
                concurrent state changes on one device - precisely the producer
                corruption IA_LOCK exists to prevent.
        """
        if self.isRunning():
            logging.warning(
                "The worker from the previous run has not stopped; refusing to "
                "start acquisition a second time."
            )
            return False
        self._run_flag = True
        if type(self.cap) is ImageAcquirer:
            self.cap.start()
        super().start()
        return True
