#!/usr/bin/env python3
"""Raw-PHX camera acquisition GUI.

A from-scratch alternative to ui.py/harvesters_pg's MainWindow that never
loads a GenTL producer while streaming. See phx_cam.py's module docstring
for the full hardware/investigation rationale.

This program reuses harvesters_pg's Qt layout (MainWindow.py, generated
from MainWindow.ui - pure widgets, no camera logic) and its saving/export
code (saving.py, importexport.py - pure numpy/h5py/cv2, no GenTL
dependency), plus cam.py's extract_timestamp()/extract_frame_counter()/
get_max_exposure() helpers (pure functions, no GenICam access). Every
acquisition and settings code path is new: there is no generic
"walk every GenICam node and expose it as an editable field" settings panel
here. That panel is what let a user reach the raw CoaXPress link-training
register that crashed the whole machine on 2026-09-21 (see the git history
of ui.py and ~/ASL-RMA-evidence-FBD-4XCXP12.md on sweetums).

Exposure, gain, and black level are fixed at connect time only - the
spinboxes are disabled while connected. This used to be live-editable
(PhxMainWindow._reconfigure_camera() implements it, correctly, and is still
here), but a 2026-09-21 stress test proved that class of operation - close
the board, one GenICam round trip, reopen - can crash the machine on its
very first cycle (stress_phx.py: dead by cycle 1, looping nothing but
connect/disconnect). A per-field "changed" handler that fires this on
ordinary use is not an acceptable default here. _reconfigure_camera() is
left wired to nothing; see its docstring before reconnecting it. Resolution
is connect-time-only regardless.

CONNECT/DISCONNECT ITSELF IS NOT FIXED (measured 2026-09-21): stress_ui.py
vs stress_phx.py, both looping plain connect/disconnect with no streaming
in between, crashed the machine on cycle 5 (ui.py) and cycle 1 (this
program) respectively - this program was NOT safer for that specific
workload, because connect_video() still runs one real GenICam round trip
per connect either way. What this program actually buys you, demonstrated
repeatedly crash-free in the same session, is staying connected: live
view and recording ran clean for extended periods. Do not loop
connect/disconnect, and do not wire settings changes back to
_reconfigure_camera() while connected, for any reason (including testing)
without expecting a hard reboot.

Run with: uv run python phx_ui.py
"""

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import numpy as np
from PyQt6 import QtWidgets

import cam
import importexport
import phx_cam
from MainWindow import Ui_MainWindow
from phx_video_thread import PhxVideoThread

DEFAULT_CTI = "/usr/local/activesilicon/lib64/libTLActiveSilicon.cti"


class PhxMainWindow(QtWidgets.QMainWindow):
    """Main window for the raw-PHX acquisition program.

    Same layout and workflow as harvesters_pg's MainWindow (connect, live
    view, record N frames, save) minus the parts that reach GenTL after
    connect: no generic settings panel, no webcam/random source (this
    program exists specifically to avoid the GenTL path, so a GenTL
    producer picker and non-CXP sources are dead weight here). Exposure,
    gain, resolution, and black level are all fixed at connect time only -
    see the module docstring for why live reconfigure is implemented
    (_reconfigure_camera()) but deliberately not wired to any UI signal.

    Attributes:
        ui (Ui_MainWindow): Reused Qt layout from harvesters_pg.
        videothread (PhxVideoThread | None): The worker thread capturing
            frames via the raw PHX board, or None while disconnected.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.Image = self.ui.ImageView
        self.Image.ui.menuBtn.setHidden(True)
        self.Image.ui.roiBtn.setHidden(True)
        self.Image.getView().setAspectLocked(True)
        self.Image.getView().setMenuEnabled(False)
        self.Image.discreteTimeLine = True

        self.videothread: PhxVideoThread | None = None
        self.is_running: bool = False
        self.recording: np.ndarray = np.array([])
        self.timestamps: list = []
        self.frame_counters: list = []
        self.filename = "."

        self._setup_ui_components()

    def _setup_ui_components(self) -> None:
        # Exactly one source: the FireBird frame grabber via raw PHX. No
        # webcam/random source, no GenTL producer picker.
        self.ui.cb_source.addItem("FireBird (raw PHX)")
        self.ui.cb_source.setDisabled(True)
        # No generic per-node settings panel in this program - see module
        # docstring.
        self.ui.pb_settings.setHidden(True)
        self.ui.menuSettings.menuAction().setVisible(False)
        self.ui.actionSelect_GenTLProducer.setVisible(False)

        self.ui.pb_connect.clicked.connect(self.connect_video)
        self.ui.pb_live.clicked.connect(self.toggle_video)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.pb_acquire.clicked.connect(self.on_btn_acquire_clicked)
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_save.clicked.connect(self.save)
        self.ui.sb_frames.editingFinished.connect(self.on_frames_changed)
        self.on_frames_changed()  # run once to set up the seconds display
        self.ui.sb_seconds.editingFinished.connect(self.on_seconds_changed)
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.pb_save_frame.clicked.connect(self.save_current_frame)

        self.ui.pb_live.setDisabled(True)
        self.ui.lbl_status.setStyleSheet("color: red;")
        self.ui.recording.setHidden(True)

        self.ui.ImageView.timeLine.setPen(width=10)  # bigger cursor in the timeline
        self.ui.pb_play.clicked.connect(self.on_play_clicked)
        self.ui.sb_goto_frame.editingFinished.connect(
            lambda: self.Image.setCurrentIndex(self.ui.sb_goto_frame.value())
        )
        self.ui.ImageView.getImageItem().sigImageChanged.connect(
            lambda: self.ui.lbl_current_frame.setText(
                f"Current frame: {self.ui.ImageView.currentIndex}"
            )
        )
        self.ui.cb_histogram.checkStateChanged.connect(self.on_histogram_checked)

        # Exposure/gain/black level are editable both before connecting
        # (values to apply on the next connect) and while connected (each
        # editingFinished/sliderReleased triggers a bounded reconfigure
        # cycle - see _reconfigure_camera() and the module docstring).
        # sb_analog_gain and hsli_black_level deliberately do NOT use
        # valueChanged for the reconfigure trigger (unlike ui.py's live
        # per-tick updates): that would fire a full board close/reopen on
        # every spinner click or drag tick, defeating the point of bounding
        # this to explicit user actions.
        self.ui.sb_exposure.setMaximum(cam.get_max_exposure(self.ui.sb_fps.value()))
        self.ui.sb_fps.editingFinished.connect(
            lambda: self.ui.sb_exposure.setMaximum(cam.get_max_exposure(self.ui.sb_fps.value()))
        )
        self.ui.hsli_black_level.valueChanged.connect(self.on_blacklevel_display_changed)

    def closeEvent(self, event) -> None:
        """Orderly stop the board before the window closes."""
        if self.videothread:
            self.disconnect_video()
        event.accept()

    def connect_video(self) -> None:
        """Configure the camera once via GenICam, then open it via raw PHX."""
        if self.videothread:
            return
        logging.info("Connecting to FireBird (raw PHX)")
        try:
            fps = self.ui.sb_fps.value()
            exposure = self.ui.sb_exposure.value()
            blacklevel = self.ui.hsli_black_level.value()
            cfg = cam.load_resolution_config()
            width = cfg.get("width") if isinstance(cfg.get("width"), int) else None
            height = cfg.get("height") if isinstance(cfg.get("height"), int) else None
            actual_width, actual_height, pixel_format = phx_cam.configure_camera_via_genicam(
                DEFAULT_CTI, fps=fps, exposure=exposure, blacklevel=blacklevel, width=width, height=height,
            )
            pa = phx_cam.open_board(actual_width, actual_height, pixel_format)
        except Exception as e:
            # An exception escaping a Qt slot reaches qFatal and aborts the
            # process - report it and stay disconnected instead (same
            # concern ui.py's connect_video() guards against).
            logging.exception("Could not connect to the FireBird")
            QtWidgets.QMessageBox.critical(
                self, "Could not connect", f"Connecting to the FireBird failed:\n\n{e}"
            )
            return
        self.videothread = PhxVideoThread(pa)
        self.videothread.change_pixmap_signal.connect(self.update_image)
        self.ui.pb_live.setDisabled(False)
        self.ui.pb_acquire.setDisabled(False)
        self.ui.pb_connect.setText("Disconnect")
        self.ui.pb_connect.clicked.disconnect()
        self.ui.pb_connect.clicked.connect(self.disconnect_video)
        self.ui.cb_source.setDisabled(True)
        # Settings are fixed for the lifetime of this connection - see
        # _reconfigure_camera()'s docstring for why this is disabled rather
        # than wired up: measured 2026-09-21, the same class of operation
        # (close the board, one GenICam round trip, reopen) crashed the
        # machine on the very first cycle in a dedicated test. A per-field
        # "changed" handler firing this on ordinary use is not an acceptable
        # default on this hardware.
        self.ui.settings.setDisabled(True)
        self.pause_video()  # sets status to orange without starting the live view
        logging.info(f"Connected to FireBird (raw PHX), {actual_width}x{actual_height} {pixel_format}")

    def disconnect_video(self) -> None:
        """Stop the worker, release the board, and reset the UI."""
        if not self.videothread:
            return
        logging.info("Disconnecting from FireBird (raw PHX)")
        self.videothread.close()
        self.videothread = None
        logging.info("Disconnected from FireBird (raw PHX)")
        self.is_running = False
        self.ui.pb_live.setDisabled(True)
        self.ui.pb_connect.setText("Connect")
        self.ui.pb_connect.clicked.disconnect()
        self.ui.pb_connect.clicked.connect(self.connect_video)
        self.ui.cb_source.setDisabled(True)
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_live.setText("Start Live View")
        self.ui.lbl_status.setStyleSheet("color: red;")
        self.ui.settings.setDisabled(False)

    def start_video(self) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")
        self.ui.pb_save.setDisabled(True)
        if not self.videothread._start():
            QtWidgets.QMessageBox.warning(
                self,
                "Camera busy",
                "The previous live view has not stopped yet, so it was not "
                "restarted. Try again; if it keeps happening, disconnect and "
                "reconnect the camera.",
            )
            return
        self.is_running = True
        self.ui.pb_live.setText("Pause")
        self.ui.lbl_status.setStyleSheet("color: green;")

    def pause_video(self) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")
        self.videothread.stop()
        self.is_running = False
        self.ui.pb_live.setText("Start Live View")
        self.ui.lbl_status.setStyleSheet("color: orange;")

    def toggle_video(self) -> None:
        if not self.is_running:
            self.start_video()
        else:
            self.pause_video()

    def _reconfigure_camera(self, **overrides) -> bool:
        """Re-apply camera settings via a close/reconfigure/reopen cycle.

        Correct and tested (see the 2026-09-21 live-reconfigure test run
        before the connect/disconnect stress test) - but deliberately NOT
        wired to any UI signal. See the module docstring: the same class of
        operation this performs crashed the machine on its first cycle in a
        dedicated stress test. Do not reconnect this to sb_fps/sb_exposure/
        sb_analog_gain/hsli_black_level without accepting that risk on every
        single settings change, not just eventually under repeated use.

        Returns:
            bool: False if the reconnect itself failed, in which case the
                board is left disconnected (same as a failed connect_video()
                - never half-open).
        """
        if not self.videothread:
            return False
        was_running = self.is_running
        fps = overrides.get("fps", self.ui.sb_fps.value())
        exposure = overrides.get("exposure", self.ui.sb_exposure.value())
        gain = overrides.get("gain", self.ui.sb_analog_gain.value())
        blacklevel = overrides.get("blacklevel", self.ui.hsli_black_level.value())
        logging.info(
            f"Reconfiguring: fps={fps} exposure={exposure} gain={gain} blacklevel={blacklevel}"
        )
        self.ui.pb_live.setDisabled(True)
        self.ui.pb_acquire.setDisabled(True)
        self.videothread.close()  # stops the worker, PHX_Close/PHX_Destroy the board
        self.videothread = None
        try:
            cfg = cam.load_resolution_config()
            width = cfg.get("width") if isinstance(cfg.get("width"), int) else None
            height = cfg.get("height") if isinstance(cfg.get("height"), int) else None
            actual_width, actual_height, pixel_format = phx_cam.configure_camera_via_genicam(
                DEFAULT_CTI, fps=fps, exposure=exposure, blacklevel=blacklevel, gain=gain,
                width=width, height=height,
            )
            pa = phx_cam.open_board(actual_width, actual_height, pixel_format)
        except Exception as e:
            logging.exception("Reconfigure failed; disconnected")
            QtWidgets.QMessageBox.critical(
                self, "Reconfigure failed", f"Could not reapply settings:\n\n{e}"
            )
            self.ui.pb_connect.setText("Connect")
            self.ui.pb_connect.clicked.disconnect()
            self.ui.pb_connect.clicked.connect(self.connect_video)
            self.ui.cb_source.setDisabled(True)
            self.ui.pb_acquire.setDisabled(True)
            self.ui.lbl_status.setStyleSheet("color: red;")
            return False
        self.videothread = PhxVideoThread(pa)
        self.videothread.change_pixmap_signal.connect(self.update_image)
        self.ui.pb_live.setDisabled(False)
        self.ui.pb_acquire.setDisabled(False)
        if was_running:
            self.start_video()
        else:
            self.pause_video()
        logging.info(f"Reconfigured: {actual_width}x{actual_height} {pixel_format}")
        return True

    def on_blacklevel_display_changed(self) -> None:
        """Cheap, local-only feedback while dragging the slider - no camera I/O."""
        self.ui.lbl_black_level_current.setText(str(self.ui.hsli_black_level.value()))

    def update_image(self, frame: np.ndarray) -> None:
        # Frames are stored/exported in the camera's natural (h, w)
        # orientation but shown transposed, same as ui.py's update_image().
        disp = frame.T if frame.ndim == 2 else frame.swapaxes(1, 2)
        if self.ui.cb_autolevel.isChecked():
            levels = (disp.T[1::4, ::4].min(), disp.T[1::4, ::4].max())
        else:
            levels = (0, 255)
        self.Image.setImage(disp, autoRange=False, autoHistogramRange=False, levels=levels)
        self.ui.lbl_img_minmax.setText(
            f"Pixelvalue Min: {disp.T[1::4,::4].min()} - Max: {disp.T[1::4,::4].max()}"
        )
        if not self.ui.pb_save_frame.isEnabled():
            self.ui.pb_save_frame.setDisabled(False)
        if frame.ndim == 3:
            self.ui.recording.setHidden(False)
            self.ui.sb_goto_frame.setMaximum(frame.shape[0] - 1)
            self.ui.lbl_goto_frame_max.setText(f"/ {frame.shape[0] - 1}")
        else:
            self.ui.recording.setHidden(True)

    def on_btn_acquire_clicked(self) -> None:
        self.pause_video()
        if self.videothread is not None and self.videothread.isRunning():
            logging.warning(
                "Video thread did not stop in time; aborting recording to "
                "avoid a concurrent-access race on the board."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Camera busy",
                "Live view did not stop in time, so recording was cancelled "
                "to avoid accessing the board from two threads at once. "
                "Try again; if it keeps happening, disconnect and reconnect "
                "the camera.",
            )
            return
        self.Image.clear()
        self.ui.pb_save.setDisabled(True)
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_connect.setDisabled(True)
        self.ui.pb_live.setDisabled(True)
        self.recording = np.array([])
        n_frames = int(self.ui.sb_frames.text())
        try:
            self.acquire(n_frames, self.ui.progressBar)
        except Exception as e:
            logging.exception("Recording failed")
            QtWidgets.QMessageBox.critical(
                self, "Recording failed", f"The recording was aborted:\n\n{e}"
            )
            return
        finally:
            self.ui.pb_acquire.setDisabled(False)
            self.ui.pb_connect.setDisabled(False)
            self.ui.pb_live.setDisabled(False)
        self.ui.pb_save.setDisabled(False)
        self.update_image(self.recording)

    def acquire(self, n_frames: int, progressbar) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")

        preview_state = [0.0]

        def preview(frame: np.ndarray) -> None:
            now = time.monotonic()
            if preview_state[0] == 0.0 or now - preview_state[0] >= 0.033:
                preview_state[0] = now
                self.update_image(frame)
            QtWidgets.QApplication.processEvents()

        imgs = phx_cam.record_frames(self.videothread.cap, n_frames, progressbar, preview=preview)
        self.recording = imgs
        self.timestamps = [cam.extract_timestamp(img) / 25_000_000 for img in imgs]
        self.frame_counters = [cam.extract_frame_counter(img) for img in imgs]
        self.filename = "."

    def save(self) -> None:
        """Save the recorded frames to a hdf5/png/jpg file."""
        filename, filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "HDF5 Files (*.h5);;Images (*.png);;Images (*.jpg)"
        )
        if not filename:
            return
        metadata: dict = {
            "timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "fps": self.ui.sb_fps.value(),
            "exposure": self.ui.sb_exposure.value(),
            "gain": self.ui.sb_analog_gain.value(),
            "blacklevel": self.ui.hsli_black_level.value(),
            "timestamps": self.timestamps,
            "frame_counters": self.frame_counters,
        }
        if filter == "HDF5 Files (*.h5)":
            if ".h5" not in filename:
                filename += ".h5"
            importexport.save_frames_to_h5(
                self.recording, filename, metadata=metadata, progressbar=self.ui.progressBar
            )
        elif filter == "Images (*.png)":
            if ".png" not in filename:
                filename += ".png"
            importexport.save_frames_to_png(self.recording, filename, progressbar=self.ui.progressBar)
        elif filter == "Images (*.jpg)":
            if ".jpg" not in filename:
                filename += ".jpg"
            importexport.save_frames_to_jpg(self.recording, filename, progressbar=self.ui.progressBar)
        logging.info(f"Saved recording to {filename}")
        self.filename = filename

    def on_frames_changed(self) -> None:
        n_frames = self.ui.sb_frames.value()
        fps = self.ui.sb_fps.value()
        self.ui.sb_seconds.setValue(n_frames / fps)

    def on_seconds_changed(self) -> None:
        seconds = self.ui.sb_seconds.value()
        fps = self.ui.sb_fps.value()
        self.ui.sb_frames.setValue(int(seconds * fps))

    def open_file(self) -> None:
        """Open a file dialog to load a recorded h5 file for review."""
        if self.videothread:
            self.pause_video()
            self.Image.clear()
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if not filename:
            return
        if filter == "HDF5 Files (*.h5)":
            self.recording, self.metadata = importexport.load_frames_from_h5(
                filename, progressbar=self.ui.progressBar
            )
        self.update_image(self.recording)
        self.filename = filename
        self.ui.pb_save.setDisabled(False)

    def save_current_frame(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Image", "", "Image (*.png);;Image (*.jpg)"
        )
        if filename != "" and self.Image.image is not None:
            if ".png" not in filename:
                filename += ".png"
            if ".jpg" not in filename:
                filename += ".jpg"
            self.Image.getImageItem().save(filename)

    def on_play_clicked(self) -> None:
        self.Image.togglePause()
        pb = self.ui.pb_play
        pb.setText("Pause" if pb.text() == "Play" else "Play")

    def on_histogram_checked(self) -> None:
        state = self.ui.cb_histogram.isChecked()
        self.ui.ImageView.getHistogramWidget().setHidden(not state)


def configure_logging(verbose: int = 0, log_file: str = "phx_pg.log") -> None:
    """Send logs to the console and to *log_file* (flushed every record)."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    fmt = "%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logging.getLogger().addHandler(handler)
    logging.info(f"Logging to {Path(log_file).resolve()} at level {logging.getLevelName(level)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="count", default=0)
    parser.add_argument(
        "--log-file", default="phx_pg.log", help="where to write the log (default: phx_pg.log)"
    )
    args = parser.parse_args()
    configure_logging(args.verbose, args.log_file)

    app = QtWidgets.QApplication(sys.argv)
    window = PhxMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
