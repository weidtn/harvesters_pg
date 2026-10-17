#!/usr/bin/env python3
import logging
import time

import cv2
import numpy as np
from harvesters.core import Harvester, ImageAcquirer
from PyQt6 import QtCore

import cam

# Workers that did not stop when asked: a call inside the producer (or the
# webcam driver) never returned. Each keeps its camera handles until it has
# really exited - freeing them under the stuck call is a use-after-free - and
# this set keeps the object alive until its finished signal releases them.
_STUCK_WORKERS: set["VideoThread"] = set()


def has_stuck_workers() -> bool:
    """Whether a previous capture worker is still stuck inside its camera.

    While one is, nothing may reset or re-enumerate the shared Harvester:
    Harvester.update()/reset() destroy every ImageAcquirer it created and
    unload the producer, under the stuck call.
    """
    return any(worker.isRunning() for worker in _STUCK_WORKERS)


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

    # Upper bound for stop() to wait for the worker. A healthy worker is gone
    # within one cam.FETCH_SLICE_S slice plus one frame conversion, because
    # every wait inside the device checks _run_flag between slices; this bound
    # only comes into play when a driver call itself hangs.
    JOIN_TIMEOUT_S = 10.0

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
            self.image_func = self._harvester_frame
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
            # start=False: acquisition starts with live view or a recording.
            # Starting and warming up here only to stop again right after the
            # connect was a full announce/start/kill-stop/revoke cycle on the
            # frame grabber per connect, for nothing.
            self.harvester, self.cap = cam.setup_cam(
                cti=ctis, search_key={"display_name": self.source}, start=False
            )

    def _harvester_frame(self, ia: ImageAcquirer) -> np.ndarray | None:
        # Give up waiting for a frame as soon as a stop is requested.
        return cam.get_harvester_image(ia, keep_waiting=self._should_run)

    def _should_run(self) -> bool:
        return self._run_flag

    def _stop_and_join(self) -> bool:
        """Signal the run loop to exit, wait for it, then stop acquisition.

        The order matters. harvesters' stop() revokes every announced buffer,
        so all fetched buffers have to be back with the ImageAcquirer before it
        runs. Stopping first (while the worker was still converting a frame)
        made the worker's buffer.queue() hit a revoked buffer and crash the
        producer with a SIGSEGV in queue_buffer - the same crash 32472c3 once
        fixed by joining first. The worker returns its buffer before it exits,
        and the stop itself takes IA_LOCK so nothing else can be inside the
        device either.

        Returns:
            bool: False if the worker is still running - stuck inside a call
                that did not return. The camera is then left as it is, because
                stopping it under that call is exactly the crash above.
        """
        self._run_flag = False
        if QtCore.QThread.currentThread() is self:
            # On the worker itself (e.g. __del__ after run() returned): it
            # cannot wait for itself. QThread.wait() returns at once with a
            # warning, which turned the join below into a 10 s busy loop.
            return False
        if self.isRunning():
            deadline = time.monotonic() + self.JOIN_TIMEOUT_S
            while self.isRunning() and time.monotonic() < deadline:
                self.wait(100)
            if self.isRunning():
                logging.warning(
                    f"VideoThread worker did not stop within {self.JOIN_TIMEOUT_S:.0f}s; "
                    "not stopping the camera under the stuck call"
                )
                return False
        # Stop even if the worker never ran: a camera started by
        # setup_cam(start=True) or left acquiring must not keep streaming.
        if isinstance(self.cap, ImageAcquirer):
            with cam.IA_LOCK:
                try:
                    self.cap.stop()
                except Exception as e:
                    logging.debug(f"Error stopping camera after joining the worker: {e}")
        return True

    def close(self) -> None:
        """Stop the worker and release the camera. Safe to call repeatedly.

        This teardown used to live in __del__, which handed its timing to
        CPython's refcounting and cyclic GC: the device could stay exclusively
        opened long after the UI said "disconnected", so a quick reconnect ran
        into a camera the producer still held. Callers release the camera by
        calling this explicitly (see MainWindow.disconnect_video).
        """
        logging.debug("close(): joining the worker and stopping acquisition")
        self._stop_and_join()
        if self.isRunning():
            # Free nothing the stuck worker may still be inside. Resetting the
            # harvester unloads the GenTL producer out from under an in-flight
            # fetch(); releasing a VideoCapture frees buffers out from under an
            # in-flight read(). Either one can crash the process or leave the
            # device wedged. Keep the handles until the worker really exits;
            # its finished signal then releases them on the GUI thread.
            if QtCore.QThread.currentThread() is not self and self not in _STUCK_WORKERS:
                logging.warning(
                    "VideoThread worker still running at teardown; keeping the "
                    "camera handles until it exits instead of releasing them "
                    "out from under the stuck call."
                )
                _STUCK_WORKERS.add(self)
                self.finished.connect(self._release_after_stuck_exit)
            return
        cap, self.cap = self.cap, None
        # Drop the reference without resetting: the Harvester is shared for the
        # whole session (see cam.get_harvester), so unloading its producers per
        # disconnect is exactly the churn we are avoiding. MainWindow.closeEvent
        # releases it via cam.release_harvester().
        self.harvester = None
        if cap is None:
            logging.debug("Cap was none")
        elif isinstance(cap, cv2.VideoCapture):
            cap.release()
        else:
            cam.teardown_cam(None, cap)

    def _release_after_stuck_exit(self) -> None:
        """finished-slot of a worker that close() had to leave running."""
        _STUCK_WORKERS.discard(self)
        logging.info("The stuck VideoThread worker has exited; releasing its camera now")
        self.close()

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
                # No IA_LOCK around the whole call: cam.get_harvester_image
                # takes it per wait slice, so GUI-thread node writes interleave
                # with a slow or stalled camera instead of queueing behind the
                # entire wait for a frame.
                frame = self.image_func(self.cap)
                # A frame that completes after a stop was requested is dropped,
                # so no stale or black frame lands in the view after Pause.
                if frame is not None and self._run_flag:
                    self.change_pixmap_signal.emit(frame)
            except Exception as e:
                logging.warning(f"Error while capturing a frame: {e}")
            # Outside the try on purpose. With the sleep inside it, any error
            # from image_func skipped the pacing and turned this into a tight
            # loop: it pegged a core, flooded the log, and cycled IA_LOCK fast
            # enough to starve the GUI thread of it.
            if self._run_flag:
                time.sleep(1 / 30)  # 30 fps

    def stop(self) -> bool:
        return self._stop_and_join()

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
        if isinstance(self.cap, ImageAcquirer):
            try:
                with cam.IA_LOCK:
                    self.cap.start()
            except Exception:
                self._run_flag = False
                raise
        super().start()
        return True
