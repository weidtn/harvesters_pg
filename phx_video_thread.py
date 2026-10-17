"""Worker thread for the raw-PHX (Active Silicon C SDK) acquisition backend.

Mirrors VideoThread.py's threading contract - bounded fetch, explicit
close()/stop(), never leaving acquisition running behind a dead worker - but
never touches GenTL: a phx_cam.PhxAcquirer is the only thing it drives. See
phx_cam.py's module docstring for why that split exists.
"""

import logging
import time

import numpy as np
from PyQt6 import QtCore

import phx_cam


class PhxVideoThread(QtCore.QThread):
    """Captures frames from a PhxAcquirer and emits them to the GUI thread.

    Attributes:
        cap (phx_cam.PhxAcquirer | None): The open raw-PHX board, or None
            after close(). Named ``cap`` (not ``pa``) for parity with
            VideoThread/ui.py call sites that check ``videothread.cap``.
    """

    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, pa: phx_cam.PhxAcquirer) -> None:
        super().__init__()
        self.cap: phx_cam.PhxAcquirer | None = pa
        self._run_flag: bool = False

    def _stop_and_join(self) -> None:
        """Signal the run loop to exit and wait for the worker to finish.

        Same caution as VideoThread._stop_and_join(): stopping the board
        does not guarantee a fetch already blocked inside try_fetch()
        unblocks immediately (it is bounded by FETCH_TIMEOUT_S, so it always
        returns eventually, but "eventually" can be up to that long). The
        join below, not stop() returning, is what determines whether the
        worker actually exited.
        """
        self._run_flag = False
        if self.cap is not None:
            try:
                self.cap.stop()
            except Exception as e:
                logging.debug(f"Error stopping raw-PHX acquisition while joining thread: {e}")
        if not self.isRunning():
            return
        deadline = time.monotonic() + 10.0
        while self.isRunning() and time.monotonic() < deadline:
            self.wait(100)
        if self.isRunning():
            logging.warning("PhxVideoThread worker did not stop within 10s")

    def stop(self) -> None:
        self._stop_and_join()

    def _start(self) -> bool:
        """Start acquisition and the worker thread.

        Returns:
            bool: False if the previous worker has not stopped yet, in which
                case nothing was started - see _stop_and_join().
        """
        if self.isRunning():
            logging.warning(
                "The worker from the previous run has not stopped; refusing to "
                "start acquisition a second time."
            )
            return False
        self._run_flag = True
        if self.cap is not None:
            self.cap.start()
        super().start()
        return True

    def run(self) -> None:
        """Capture frames and emit them to the main thread."""
        while self._run_flag:
            try:
                if self.cap is not None:
                    frame = phx_cam.get_image(self.cap)
                    if frame is not None:
                        self.change_pixmap_signal.emit(frame)
            except Exception as e:
                logging.warning(f"Error while capturing a frame: {e}")
            # Outside the try on purpose - see VideoThread.run()'s identical
            # comment: pacing must not be skipped by a repeatedly-erroring
            # source, or this turns into a tight loop that starves the GUI.
            time.sleep(1 / 30)

    def close(self) -> None:
        """Stop the worker and release the board. Safe to call repeatedly."""
        logging.debug("close(): stopping acquisition and joining the worker")
        self._stop_and_join()
        if self.isRunning():
            logging.warning(
                "PhxVideoThread worker still running at teardown; leaking the "
                "board handle instead of destroying it out from under the "
                "stuck call."
            )
            return
        cap, self.cap = self.cap, None
        if cap is not None:
            cap.destroy()

    def __del__(self) -> None:
        # Backstop only, same rationale as VideoThread.__del__: close() is
        # the real teardown and normally already ran.
        try:
            self.close()
        except Exception as e:
            logging.debug(f"Error during PhxVideoThread finalization: {e}")
