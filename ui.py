"""This is the ViewModel in the MVVM pattern.

All user interaction with the ui is handled in here and communicated to the modle (cam.py)
"""

import os

import sys
from pathlib import Path
from PyQt6 import QtWidgets, QtCore
import numpy as np
import datetime
import time
import uuid

os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cam
import camera_profiles
import importexport
import logging
from MainWindow import Ui_MainWindow
from settings import Ui_SettingsWindow
from VideoThread import VideoThread
import argparse
from genicam.genapi import (
    IInteger,
    IEnumeration,
    IBoolean,
    IFloat,
    IString,
    AccessException,
    ICommand,
)


class SettingsWindow(QtWidgets.QMainWindow):
    """Settings window to change camera settings.

    The settings window is created in Qt Designer and converted using pyuic6.
    The settings window is populated with the camera settings
    so the user can change the settings.

    Attributes:
        ui (Ui_SettingsWindow): The UI object created using Qt Designer
        mainwindow (MainWindow): The main window object
        settings (dict): A dictionary containing the camera settings
    """

    def __init__(self, mainwindow) -> None:
        """Initialize the settings window.

        Args:
            mainwindow (MainWindow): The main window object
        """
        super(SettingsWindow, self).__init__()
        self.ui = Ui_SettingsWindow()
        self.ui.setupUi(self)
        self.mainwindow = mainwindow
        self.settings = {}
        self.timer = QtCore.QTimer(self)
        self._timer_connected: bool = False

    def fill_settings(self):
        """Fill the settings window' with the camera settings."""
        # Drop whatever the previous camera left behind first. This used to be
        # purely additive: each connect appended another full set of rows, kept
        # the dead camera's widgets in self.settings, and connected
        # update_fields to the same long-lived timer again. After N
        # connect/disconnect cycles a single 500 ms tick ran N complete
        # node-map polls, and the GUI never got idle time back.
        self.teardown()
        nodemap = cam.get_harvester_nodemap(self.mainwindow.videothread.cap)
        for node in dir(nodemap):
            self.settings[node] = {
                "label": QtWidgets.QLabel(text=node),
                "widget": None,
                "button": QtWidgets.QPushButton(text="Apply"),
            }
            try:
                if type(getattr(nodemap, node)) is IInteger:
                    widget = QtWidgets.QSpinBox(readOnly=True)
                    try:
                        widget.setMinimum(getattr(nodemap, node).min)
                        widget.setMaximum(getattr(nodemap, node).max)
                        self.settings[node]["widget"] = widget
                        widget.valueChanged.connect(
                            lambda v, node=node: self.update_node(node, v)
                        )
                    except OverflowError:
                        widget = QtWidgets.QLineEdit(readOnly=True)
                        widget.setText(str(getattr(nodemap, node).value))
                    except AttributeError:
                        logging.debug(f"AttributeError for {node}")
                    widget.wheelEvent = lambda e: None
                    self.settings[node]["widget"] = widget
                elif type(getattr(nodemap, node)) is IFloat:
                    widget = QtWidgets.QDoubleSpinBox()
                    widget.valueChanged.connect(
                        lambda v, node=node: self.update_node(node, v)
                    )
                    self.settings[node]["widget"] = widget
                elif type(getattr(nodemap, node)) is IEnumeration:
                    widget = QtWidgets.QComboBox(enabled=False)
                    widget.addItems(getattr(nodemap, node)._get_symbolics())
                    widget.currentTextChanged.connect(
                        lambda value, node=node: self.update_node(node, value)
                    )
                    self.settings[node]["widget"] = widget
                elif type(getattr(nodemap, node)) is IBoolean:
                    widget = QtWidgets.QCheckBox(enabled=False)
                    widget.stateChanged.connect(
                        lambda value, node=node: self.update_node(node, bool(value))
                    )
                    self.settings[node]["widget"] = widget
                elif type(getattr(nodemap, node)) is IString:
                    widget = QtWidgets.QLineEdit()
                    widget.editingFinished.connect(
                        lambda: self.update_node(node, widget.text())
                    )
                    self.settings[node]["widget"] = widget
                elif type(getattr(nodemap, node)) is ICommand:
                    widget = QtWidgets.QPushButton(text="Execute")
                    widget.pressed.connect(lambda node=node: self.execute_command(node))
                    self.settings[node]["widget"] = widget
                else:
                    logging.debug(
                        f"Type {type(getattr(nodemap, node))} not supported for {node}"
                    )
                    self.settings.pop(node)
                    continue
            except AccessException:
                self.settings.pop(node)
                continue
            except Exception as e:
                logging.debug(f"Exception {e} for {node}")
                self.settings.pop(node)
                continue
        for field in self.settings:
            self.ui.formLayout.addRow(
                self.settings[field]["label"],
                self.settings[field]["widget"],
            )
        # The poll starts from showEvent, not here. Starting it on connect
        # meant the app read the whole node map over GenCP twice a second from
        # the moment a camera was connected - forever, whether or not anyone
        # had this panel open. Continuous register traffic for zero user value.
        if self.isVisible():
            self.start_polling()

    def start_polling(self) -> None:
        """Poll the node map every 500 ms, and refresh once immediately."""
        if not self.settings:
            return  # nothing built yet: no camera connected
        if not self._timer_connected:
            self.timer.timeout.connect(self.update_fields)
            self._timer_connected = True
        self.update_fields()  # show current values now, not in 500 ms
        self.timer.setInterval(500)
        self.timer.start()

    def showEvent(self, event) -> None:
        """Start polling when the panel appears on screen."""
        super().showEvent(event)
        self.start_polling()

    def hideEvent(self, event) -> None:
        """Stop polling as soon as the panel is off screen."""
        super().hideEvent(event)
        self.timer.stop()

    def teardown(self) -> None:
        """Stop polling and drop every widget built for the current camera.

        Called on disconnect and before repopulating, so no poll outlives the
        node map it reads and the panel cannot accumulate duplicate rows.
        """
        self.timer.stop()
        if self._timer_connected:
            self.timer.timeout.disconnect(self.update_fields)
            self._timer_connected = False
        while self.ui.formLayout.rowCount():
            self.ui.formLayout.removeRow(0)
        self.settings = {}

    def execute_command(self, node):
        """Execute a command on the camera.

        Args:
            node (str): The command to execute
        """
        logging.debug(f"Executing command {node}")
        nodemap = cam.get_harvester_nodemap(self.mainwindow.videothread.cap)
        with cam.IA_LOCK:
            getattr(nodemap, node).execute()
        self.update_fields()

    def update_fields(self):
        """Update the fields with the current values from the camera.

        This function is called when the settings window is opened
        or when the settings are changed in the main window.
        """
        try:
            nodemap = cam.get_harvester_nodemap(self.mainwindow.videothread.cap)
        except AttributeError:
            logging.debug("AttributeError: nodemap not found, camera not connected")
            return
        # Serialize node_map reads against the worker's fetch() so the poll never
        # accesses the producer concurrently with a live fetch (not thread-safe).
        # Never *wait* for the lock, though: the worker holds it for a whole
        # fetch, so blocking here froze the GUI thread for as long as the camera
        # took to deliver - and forever when the fetch itself hung. A skipped
        # poll only means the fields refresh 500 ms later, which is invisible.
        if not cam.IA_LOCK.acquire(blocking=False):
            logging.debug("Camera busy; skipping this settings poll")
            return
        try:
            for node in self.settings:
                # Access_modes possible: NA (1), ?, RO (3), RW (4)
                widget = self.settings[node]["widget"]
                widget.blockSignals(True)
                try:
                    access_mode = getattr(nodemap, node).get_access_mode()
                    if type(widget) is QtWidgets.QSpinBox:
                        widget.setValue(getattr(nodemap, node).value)
                        widget.setReadOnly(
                            False
                        ) if access_mode == 4 else widget.setReadOnly(True)
                    elif type(widget) is QtWidgets.QDoubleSpinBox:
                        widget.setValue(getattr(nodemap, node).value)
                        widget.setReadOnly(
                            False
                        ) if access_mode == 4 else widget.setReadOnly(True)
                    elif type(widget) is QtWidgets.QComboBox:
                        widget.setCurrentText(str(getattr(nodemap, node).value))
                        widget.setEnabled(True) if access_mode == 4 else widget.setEnabled(
                            False
                        )
                    elif type(widget) is QtWidgets.QCheckBox:
                        widget.setChecked(getattr(nodemap, node).value)
                        widget.setEnabled(True) if access_mode == 4 else widget.setEnabled(
                            False
                        )
                    elif type(widget) is QtWidgets.QLineEdit:
                        widget.setText(str(getattr(nodemap, node).value))
                        widget.setReadOnly(
                            False
                        ) if access_mode == 4 else widget.setReadOnly(True)
                except AccessException:
                    continue
                finally:
                    # In a finally, not a trailing statement: the
                    # AccessException path above used to `continue` straight
                    # past it, leaving that widget with its signals blocked for
                    # the rest of the session.
                    widget.blockSignals(False)
        finally:
            cam.IA_LOCK.release()

    def update_node(self, node, value):
        """Update the camera settings based on the settings window."""
        logging.debug(f"Setting {node} to {value}")
        profile = self.mainwindow.profile
        if node == profile.frame_rate_node:
            self.mainwindow.ui.sb_fps.setValue(value)
            self.mainwindow.on_fps_changed()
        elif node == profile.exposure_node:
            self.mainwindow.ui.sb_exposure.setValue(value)
            self.mainwindow.on_exposure_changed()
        elif node == profile.gain_node:
            self.mainwindow.ui.sb_analog_gain.setValue(int(value))
            self.mainwindow.on_analog_gain_changed()
        elif node == "BlackLevel":
            self.mainwindow.ui.hsli_black_level.setValue(value)
            self.mainwindow.on_blacklevel_changed()
        cam.set_attribute(self.mainwindow.videothread.cap, node, value)
        self.update_fields()


class MainWindow(QtWidgets.QMainWindow):
    """Main window for the application.

    The UI is created using Qt Designer and converted to python using pyuic6.
    The VideoThread class is used to capture video frames and emit them to the main thread.
    The video feed is displayed using pyqtgraph's ImageItem.
    The user can select the video source from a combobox and connect to the source using the "Connect" button.
    The "Start"/"Stop" button starts and pauses the video feed.
    The "Record" button can be used to record a given number of frames into a hdf5 file.

    Attributes:
        ui (Ui_MainWindow): The UI object created using Qt Designer
        videothread (VideoThread | None): The worker thread to capture video frames
        is_running (bool): Flag to control the video feed
        source (str): The video source
        cam_indices (list[int]): List of available camera indices for opencv
    """

    def __init__(self, *args, **kwargs) -> None:
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # Quirks of the connected camera family (node names); replaced with
        # the detected profile in setup_settings() once a Harvester camera
        # connects. The settings window reads it from here as well.
        self.profile = camera_profiles.SFNC_DEFAULT

        self.ui.settings_window = SettingsWindow(self)

        # Setup and add the ImageItem for displaying the video feed
        self.Image = self.ui.ImageView
        self.Image.ui.menuBtn.setHidden(True)
        self.Image.ui.roiBtn.setHidden(True)
        self.Image.getView().setAspectLocked(True)
        # Disable menu
        self.Image.getView().setMenuEnabled(False)
        self.Image.discreteTimeLine = True

        # Initialize variables
        self.videothread: None | VideoThread = None
        self.is_running: bool = False
        self.source: str = "random"
        self.recording: np.ndarray = np.array([])
        # Identity of self.recording: every file it is saved to carries both.
        self.recording_id: str | None = None
        self.recording_timestamp: str | None = None  # start, %Y%m%d%H%M%S
        self.ctis: list[str] = []
        self.filename = "."

        self.setup_ui_components()

    def setup_ui_components(self) -> None:
        """Set up UI components and connect signals."""
        # Connect the buttons to their respective functions
        self.ui.pb_connect.clicked.connect(self.connect_video)
        self.ui.pb_live.clicked.connect(self.toggle_video)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.pb_acquire.clicked.connect(self.on_btn_acquire_clicked)
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_save.clicked.connect(self.save)
        self.ui.sb_frames.editingFinished.connect(self.on_frames_changed)
        self.on_frames_changed()  # run once to setup the seconds display:
        self.ui.sb_seconds.editingFinished.connect(self.on_seconds_changed)
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionSelect_GenTLProducer.triggered.connect(self.find_cti_dialog)
        self.ui.pb_settings.clicked.connect(self.on_settings_pressed)
        self.ui.pb_save_frame.clicked.connect(self.save_current_frame)

        # Disable certain elements initially
        self.ui.pb_live.setDisabled(True)
        self.ui.lbl_status.setStyleSheet("color: red;")
        self.ui.recording.setHidden(True)

        self.ui.settings.setHidden(True)
        self.ui.sb_fps.editingFinished.connect(self.on_fps_changed)
        self.ui.sb_exposure.editingFinished.connect(self.on_exposure_changed)
        self.ui.sb_analog_gain.valueChanged.connect(self.on_analog_gain_changed)
        self.ui.hsli_black_level.valueChanged.connect(self.on_blacklevel_changed)

        # Auto exposure / auto gain toggles, appended on the right of their
        # rows. Hidden until a camera that has the respective auto feature is
        # connected (see setup_auto_boxes).
        self.ui.cb_auto_exposure = QtWidgets.QCheckBox(text="Auto")
        self.ui.cb_auto_exposure.setVisible(False)
        self.ui.hLay_exposure.addWidget(self.ui.cb_auto_exposure)
        self.ui.cb_auto_exposure.toggled.connect(self.on_auto_exposure_toggled)
        self.ui.cb_auto_gain = QtWidgets.QCheckBox(text="Auto")
        self.ui.cb_auto_gain.setVisible(False)
        self.ui.hLay_analog_gain.addWidget(self.ui.cb_auto_gain)
        self.ui.cb_auto_gain.toggled.connect(self.on_auto_gain_toggled)
        self.ui.cb_histogram.checkStateChanged.connect(self.on_histogram_checked)
        self.ui.ImageView.timeLine.setPen(
            width=10
        )  # Make the cursor in timeline bigger
        self.ui.pb_play.clicked.connect(self.on_play_clicked)
        self.ui.sb_goto_frame.editingFinished.connect(
            lambda: self.Image.setCurrentIndex(self.ui.sb_goto_frame.value())
        )
        self.ui.ImageView.getImageItem().sigImageChanged.connect(
            lambda: self.ui.lbl_current_frame.setText(
                f"Current frame: {self.ui.ImageView.currentIndex}"
            )
        )

        # Set tooltips for the labels
        self.ui.lbl_fps_minmax.setToolTip(self.ui.sb_fps.toolTip())
        self.ui.lbl_fps.setToolTip(self.ui.sb_fps.toolTip())

        self.ui.lbl_exposure.setToolTip(self.ui.sb_exposure.toolTip())
        self.ui.lbl_exposure_minmax.setToolTip(self.ui.sb_exposure.toolTip())

        self.ui.lbl_analog_gain_minmax_2.setToolTip(self.ui.sb_analog_gain.toolTip())
        self.ui.lbl_analog_gain.setToolTip(self.ui.sb_analog_gain.toolTip())

        self.ui.lbl_black_level.setToolTip(self.ui.hsli_black_level.toolTip())
        self.ui.lbl_black_level_current.setToolTip(self.ui.hsli_black_level.toolTip())
        self.ui.lbl_black_level_min.setToolTip(self.ui.hsli_black_level.toolTip())
        self.ui.lbl_black_level_max.setToolTip(self.ui.hsli_black_level.toolTip())

        # Resolution (AOI) row in the settings panel. Width/Height are only
        # writable while acquisition is stopped on many producers, so the Apply
        # button restarts the camera through cam.set_resolution and persists
        # the chosen size for the next connect.
        self.ui.lbl_resolution = QtWidgets.QLabel(text="Image Size [px]")
        self.ui.sb_res_width = QtWidgets.QSpinBox()
        self.ui.sb_res_width.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.ui.lbl_res_x = QtWidgets.QLabel(text="×")
        self.ui.sb_res_height = QtWidgets.QSpinBox()
        self.ui.sb_res_height.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.ui.pb_res_apply = QtWidgets.QPushButton(text="Apply")
        self.ui.pb_res_apply.setDisabled(True)  # until a Harvester camera connects
        self.ui.hLay_resolution = QtWidgets.QHBoxLayout()
        for widget in (
            self.ui.lbl_resolution,
            self.ui.sb_res_width,
            self.ui.lbl_res_x,
            self.ui.sb_res_height,
            self.ui.pb_res_apply,
        ):
            self.ui.hLay_resolution.addWidget(widget)
        self.ui.vLay_settings.addLayout(self.ui.hLay_resolution)
        self.ui.pb_res_apply.clicked.connect(self.on_resolution_apply)

        self.populate_video_sources()

    def populate_video_sources(self) -> None:
        """Populate combobox with available video sources.

        Refuses to run while a camera is connected. Enumerating builds a second
        Harvester over the same GenTL producer and resets it when done, which
        unloads the producer library - with a device open through it, that
        destroys handles still in use. Probing the OpenCV indices reopens a
        webcam that may currently be streaming, for the same reason.
        """
        if self.videothread is not None:
            logging.warning(
                "Not rescanning video sources while a camera is connected: "
                "re-enumerating would reset the GenTL producer in use."
            )
            QtWidgets.QMessageBox.information(
                self,
                "Camera connected",
                "Disconnect the current camera before rescanning for sources.",
            )
            return

        previous = self.ui.cb_source.currentText()
        # Rebuild from scratch. This runs again whenever a .cti is added, and
        # appending duplicated every entry that was already there.
        self.ui.cb_source.clear()

        # Add OpenCV camera indices
        cam_indices: list[int] = cam.get_camera_indices()
        [self.ui.cb_source.addItem(f"Webcam {i}") for i in cam_indices]

        # Add Harvester cameras. Deduplicated: add_file() rejects a producer
        # that is already loaded, so a repeated scan used to break enumeration
        # outright.
        for cti in cam.load_cti_envvar():
            if cti not in self.ctis:
                self.ctis.append(cti)
        harvester_cams: list[dict] = cam.return_harvester_cameras(self.ctis)
        [
            self.ui.cb_source.addItem(f"{cam['display_name']}")
            for cam in harvester_cams
            if cam["model"] != "Unknown Model"
        ]
        if self.ui.cb_source.findText("random") == -1:
            self.ui.cb_source.addItem("random")
        if previous:
            index = self.ui.cb_source.findText(previous)
            if index != -1:
                self.ui.cb_source.setCurrentIndex(index)

    def closeEvent(self, event) -> None:
        """Orderly stop the camera before the window closes.

        Stops acquisition and joins the worker before Python teardown, so the
        GenTL producer is never destroyed while still streaming (a whole-system
        freeze cause).
        """
        if self.videothread:
            self.disconnect_video()
        else:
            # No worker, but the settings panel may still hold a running poll.
            self.ui.settings_window.teardown()
        self.ui.settings_window.close()
        # Only here: this unloads the GenTL producers, which disconnect_video
        # deliberately leaves alone.
        cam.release_harvester()
        event.accept()

    def connect_video(self) -> None:
        """Read the video source and start the video thread."""
        if not self.videothread:
            self.source = self.ui.cb_source.currentText()
            logging.info(f"Connecting to {self.source!r}")
            try:
                videothread = VideoThread(self.source, self.ctis)
            except Exception as e:
                # An exception escaping a Qt slot reaches qFatal and aborts the
                # process - possibly with the camera half-open. Report it and
                # stay disconnected instead; setup_cam has already released
                # whatever it managed to open.
                logging.exception(f"Could not connect to {self.source}")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Could not connect",
                    f"Connecting to {self.source!r} failed:\n\n{e}",
                )
                return
            self.videothread = videothread
            self.videothread.change_pixmap_signal.connect(self.update_image)
            self.ui.pb_live.setDisabled(False)
            self.ui.pb_acquire.setDisabled(False)
            self.ui.pb_connect.setText("Disconnect")
            self.ui.pb_connect.clicked.disconnect()
            self.ui.pb_connect.clicked.connect(self.disconnect_video)
            self.ui.cb_source.setDisabled(True)
            # Stop acquisition before reading the node map. setup_cam leaves
            # the camera streaming, so the settings read-out and the saved-AOI
            # re-apply below used to run against a live device whose buffers
            # nobody was draining. pause_video() also sets the status to orange.
            self.pause_video()
            try:
                self.on_source_connected()
            except Exception as e:
                # Same qFatal concern as above, but here the camera is already
                # open: keep the connection, drop the settings panel, and tell
                # the user rather than aborting the process.
                logging.exception("Could not populate the camera settings")
                self.ui.settings_window.teardown()
                self.ui.settings.setHidden(True)
                self.ui.pb_settings.setDisabled(True)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Settings unavailable",
                    f"The camera connected, but its settings could not be "
                    f"read:\n\n{e}",
                )
            logging.info(f"Connected to {self.source!r}")

    def disconnect_video(self) -> None:
        """Stop the video thread, release the camera, and reset the UI."""
        if self.videothread:
            # Stop polling the node map before the device goes away, then
            # release it explicitly. Dropping the reference and leaving the
            # teardown to __del__ kept the camera opened for an unpredictable
            # stretch, so a quick reconnect hit a device the producer still
            # held.
            logging.info(f"Disconnecting from {self.source!r}")
            self.ui.settings_window.teardown()
            self.videothread.close()
            self.videothread = None
            logging.info(f"Disconnected from {self.source!r}")
            self.is_running = False
            self.ui.pb_live.setDisabled(True)
            self.ui.pb_connect.setText("Connect")
            self.ui.pb_connect.clicked.disconnect()
            self.ui.pb_connect.clicked.connect(self.connect_video)
            self.ui.cb_source.setDisabled(False)
            self.ui.pb_acquire.setDisabled(True)
            self.ui.pb_live.setText("Start")
            self.ui.lbl_status.setStyleSheet("color: red;")
            self.ui.settings.setHidden(True)
            # Reset the per-camera row extensions until the next connect.
            for checkbox, spinbox in (
                (self.ui.cb_auto_exposure, self.ui.sb_exposure),
                (self.ui.cb_auto_gain, self.ui.sb_analog_gain),
            ):
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.setVisible(False)
                checkbox.blockSignals(False)
                spinbox.setEnabled(True)
            self.profile = camera_profiles.SFNC_DEFAULT
            self.ui.pb_settings.setDisabled(True)

    def start_video(self) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")
        self.ui.pb_save.setDisabled(True)
        # starts camera acquisition and the worker thread once
        if not self.videothread._start():
            # The previous worker never exited (a pause that timed out). Do not
            # pretend the live view restarted: a second start() on the same
            # device while the old fetch() is still in flight is exactly the
            # race this guards against.
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
        self.ui.pb_live.setText("Start")
        self.ui.lbl_status.setStyleSheet("color: orange;")

    def toggle_video(self) -> None:
        if not self.is_running:
            self.start_video()
        else:
            self.pause_video()

    def update_image(self, frame: np.ndarray) -> None:
        # Frames are stored/exported in the camera's natural (h, w) orientation
        # (that is what makes the fast recording path linear), but the app has
        # always shown them transposed on screen. Transpose at display time only,
        # so the on-screen picture is unchanged while the hot path stays fast.
        disp = frame.T if frame.ndim == 2 else frame.swapaxes(1, 2)
        if self.ui.cb_autolevel.isChecked():
            levels = (disp.T[1::4, ::4].min(), disp.T[1::4, ::4].max())
        else:
            levels = (0, 255)
        self.Image.setImage(
            disp, autoRange=False, autoHistogramRange=False, levels=levels
        )
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
            # pause_video() -> VideoThread.stop() can time out (10s) without
            # actually stopping the worker: ia.stop() does not reliably
            # unblock a fetch() already blocked inside the device (see
            # VideoThread._stop_and_join's docstring). Recording now would
            # start a second, concurrent start()/fetch()/stop() sequence on
            # the same ImageAcquirer the stuck worker is still using, so
            # refuse instead of silently racing it.
            logging.warning(
                "Video thread did not stop in time; aborting recording to "
                "avoid a concurrent-access race on the camera."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Camera busy",
                "Live view did not stop in time, so recording was cancelled "
                "to avoid accessing the camera from two threads at once. "
                "Try again; if it keeps happening, disconnect and reconnect "
                "the camera.",
            )
            return
        self.Image.clear()
        self.ui.pb_save.setDisabled(True)
        # Disable actions that touch the camera / re-enter the event loop while
        # recording: the loop below pumps events (so preview frames repaint), and
        # a re-entered connect/start/acquire would race fetch() on the device.
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_connect.setDisabled(True)
        self.ui.pb_live.setDisabled(True)
        # A mid-recording AOI change would violate the preallocated (n, h, w)
        # buffer; hold the Apply button (and the camera keep-alive) until done.
        self.ui.pb_res_apply.setDisabled(True)
        # The recording loop pumps the event loop after every frame while
        # holding IA_LOCK. IA_LOCK is reentrant, so the settings poll would
        # re-acquire it on this very thread and read the node map from inside
        # the fetch loop - a lock cannot protect against the GUI thread
        # re-entering itself. Suspend the poll, and the panel's own widgets,
        # for the duration instead.
        settings_window = self.ui.settings_window
        was_polling = settings_window.timer.isActive()
        settings_window.timer.stop()
        settings_window.setEnabled(False)
        self.recording = np.array([])
        n_frames = int(self.ui.sb_frames.text())
        try:
            self.acquire(n_frames, self.ui.progressBar)
        except Exception as e:
            # A fetch timeout aborts the recording; letting it escape the slot
            # would take the process down with qFatal.
            logging.exception("Recording failed")
            QtWidgets.QMessageBox.critical(
                self, "Recording failed", f"The recording was aborted:\n\n{e}"
            )
            return
        finally:
            self.ui.pb_acquire.setDisabled(False)
            self.ui.pb_connect.setDisabled(False)
            self.ui.pb_live.setDisabled(False)
            self.ui.pb_res_apply.setDisabled(False)
            settings_window.setEnabled(True)
            if was_polling:
                settings_window.timer.start()
        self.ui.pb_save.setDisabled(False)
        self.update_image(self.recording)

    def toggle_settings(self) -> None:
        if self.ui.settings.isHidden():
            self.ui.settings.setHidden(False)
        else:
            self.ui.settings.setHidden(True)

    def acquire(
        self, n_frames: int, progressbar: None | QtWidgets.QProgressBar
    ) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")

        # Mutable holder for the throttle timestamp, so the first frame is
        # always shown immediately (nonlocal: closure writes through it).
        preview_state = [0.0]

        def preview(frame: np.ndarray) -> None:
            """Show the frame being recorded and let the UI repaint.

            Recording runs on the GUI thread, so repainting needs an explicit
            event-loop pump after each frame, otherwise nothing renders until
            the loop finishes. setImage on a full-res frame takes ~1.5 ms, which
            alone would cap recording at ~600 fps, so image updates are throttled
            to ~30 fps while the event pump still runs every frame.
            """
            now = time.monotonic()
            if preview_state[0] == 0.0 or now - preview_state[0] >= 0.033:
                preview_state[0] = now
                self.update_image(frame)
            QtWidgets.QApplication.processEvents()

        started = datetime.datetime.now()
        if self.source == "random":
            imgs = cam.record_random_frames(n_frames, progressbar, preview=preview)
        elif "Webcam" in self.source:
            imgs = cam.record_webcam_frames(
                self.videothread.cap, n_frames, progressbar, preview=preview
            )
        else:
            imgs = cam.record_harvester_frames(
                self.videothread.cap, n_frames, progressbar, preview=preview
            )
        self.recording = imgs
        self.recording_id = str(uuid.uuid4())
        self.recording_timestamp = started.strftime("%Y%m%d%H%M%S")
        self.timestamps = [cam.extract_timestamp(img) / 25_000_000 for img in imgs]
        self.frame_counters = [cam.extract_frame_counter(img) for img in imgs]
        self.filename = "."

    def save(self) -> None:
        """Save the recorded frames to a hdf5 file."""
        # Get filename with a file dialog
        filename, filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "HDF5 Files (*.h5);;Images (*.png);;Images (*.jpg)",
        )
        if self.recording_id is None:  # loaded from a file saved without an id
            self.recording_id = str(uuid.uuid4())
        metadata: dict = {
            "recording_id": self.recording_id,
            "timestamp": self.recording_timestamp
            or datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        }
        # TODO: Better handling, especially if the video was loaded from a file
        if type(self.videothread.cap) is cam.VideoCapture:
            metadata["fps"] = cam.get_webcam_fps(self.videothread.cap)
        elif self.source == "random":
            metadata["fps"] = 30
        elif type(self.videothread.cap) is cam.ImageAcquirer:
            nodemap = cam.get_all_attributes_harvesters(self.videothread.cap)
            [metadata.update({k: v}) for k, v in nodemap.items()]
            metadata["fps"] = self.ui.sb_fps.value()
            metadata["timestamps"] = self.timestamps
            metadata["frame_counters"] = self.frame_counters
        if filter == "HDF5 Files (*.h5)":
            if ".h5" not in filename:
                filename += ".h5"
            importexport.save_frames_to_h5(
                self.recording,
                filename,
                metadata=metadata,
                progressbar=self.ui.progressBar,
            )
        elif filter == "Images (*.png)":
            if ".png" not in filename:
                filename += ".png"
            importexport.save_frames_to_png(
                self.recording, filename, progressbar=self.ui.progressBar
            )
        elif filter == "Images (*.jpg)":
            if ".jpg" not in filename:
                filename += ".jpg"
            importexport.save_frames_to_jpg(
                self.recording, filename, progressbar=self.ui.progressBar
            )
        logging.info(f"Saved recording to {filename}")
        self.filename = filename

    def on_frames_changed(self) -> None:
        """Update the progress bar maximum value."""
        n_frames = self.ui.sb_frames.value()
        fps = self.ui.sb_fps.value()
        self.ui.sb_seconds.setValue(n_frames / fps)

    def on_seconds_changed(self) -> None:
        """Update the number of frames based on the seconds."""
        seconds = self.ui.sb_seconds.value()
        fps = self.ui.sb_fps.value()
        self.ui.sb_frames.setValue(int(seconds * fps))

    def on_source_connected(self) -> None:
        """Update the source based on the combobox."""
        self.source = self.ui.cb_source.currentText()
        if "random" in self.source or "Webcam" in self.source:
            self.ui.settings.setHidden(True)
        else: # Harvester camera
            self.setup_settings()  # get minmax values for settings
            self.setup_resolution_settings()  # populate + re-apply saved image size
            self.ui.pb_settings.setDisabled(False)
            self.ui.settings.setHidden(False)
            self.ui.settings_window.fill_settings()

    def on_fps_changed(self) -> None:
        """Update the fps based on the settings."""
        cam.set_attribute(
            self.videothread.cap,
            self.profile.frame_rate_node,
            self.ui.sb_fps.value(),
        )
        self.ui.lbl_exposure_minmax.setText(
            f"(2 - {cam.get_max_exposure(self.ui.sb_fps.value())})"
        )
        self.on_exposure_changed()
        self.ui.sb_exposure.setMaximum(cam.get_max_exposure(self.ui.sb_fps.value()))
        self.on_frames_changed()

    def on_exposure_changed(self) -> None:
        """Update the exposure based on the settings."""
        if self.ui.cb_auto_exposure.isChecked():
            return  # camera is in auto mode; manual value not pushed
        logging.debug(f"Setting exposure to {self.ui.sb_exposure.value()}")
        exposure = cam.cap_at_max_exposure(
            self.ui.sb_exposure.value(), self.ui.sb_fps.value()
        )
        self.ui.sb_exposure.setValue(exposure)
        cam.set_attribute(self.videothread.cap, self.profile.exposure_node, exposure)

    def on_analog_gain_changed(self) -> None:
        """Update the gain based on the settings."""
        logging.debug(f"Setting analog gain to {self.ui.sb_analog_gain.value()}")
        cam.set_attribute(
            self.videothread.cap, self.profile.gain_node, self.ui.sb_analog_gain.value()
        )

    def on_blacklevel_changed(self) -> None:
        """Update the blacklevel based on the settings."""
        logging.debug(f"Setting black level to {self.ui.hsli_black_level.value()}")
        self.ui.lbl_black_level_current.setText(str(self.ui.hsli_black_level.value()))
        cam.set_attribute(
            self.videothread.cap, "BlackLevel", self.ui.hsli_black_level.value()
        )

    def setup_settings(self) -> None:
        ia = self.videothread.cap
        node_map = ia.remote_device.node_map
        self.profile = camera_profiles.detect_profile(node_map)
        logging.info(f"Connected camera uses profile: {self.profile.key}")
        # Some cameras (e.g. TLSimu) report no frame-rate maximum anywhere;
        # keep the UI's default cap in that case instead of crashing.
        max_fps = camera_profiles.max_frame_rate(node_map, self.profile)
        if max_fps is None:
            logging.warning("Camera reports no frame rate maximum, keeping UI default")
            max_fps = self.ui.sb_fps.maximum()
        else:
            max_fps = int(max_fps)
        self.ui.sb_fps.setMaximum(max_fps)
        self.ui.lbl_fps_minmax.setText(f"(1 - {max_fps})")
        self.on_fps_changed()
        self.setup_auto_boxes()

    def setup_auto_boxes(self) -> None:
        """Show the Auto checkboxes for the features this camera can automate.

        Runs after connecting a Harvester camera: a checkbox appears on the
        right of its row only when the connected camera exposes the respective
        auto node; its initial state mirrors the camera's current mode.
        """
        node_map = self.videothread.cap.remote_device.node_map
        for checkbox, spinbox, node_name in (
            (
                self.ui.cb_auto_exposure,
                self.ui.sb_exposure,
                self.profile.auto_exposure_node,
            ),
            (
                self.ui.cb_auto_gain,
                self.ui.sb_analog_gain,
                self.profile.auto_gain_node,
            ),
        ):
            value = None
            if node_name is not None:
                with cam.IA_LOCK:
                    try:
                        value = str(getattr(node_map, node_name).value)
                    except Exception as e:
                        logging.debug(f"{node_name} not available: {e}")
            checkbox.blockSignals(True)
            checkbox.setVisible(value is not None)
            checkbox.setChecked(value == camera_profiles.AUTO_ON)
            checkbox.blockSignals(False)
            spinbox.setEnabled(not checkbox.isChecked())
        self.setup_gain_range(node_map)

    def setup_gain_range(self, node_map) -> None:
        """Clamp the gain spinbox (and its label) to what the camera allows."""
        try:
            with cam.IA_LOCK:
                gain = getattr(node_map, self.profile.gain_node)
                lo, hi = float(gain.min), float(gain.max)
        except Exception as e:
            logging.debug(f"Gain range not available, keeping defaults: {e}")
            return
        self.ui.sb_analog_gain.setMinimum(int(lo))
        self.ui.sb_analog_gain.setMaximum(int(hi))
        self.ui.lbl_analog_gain_minmax_2.setText(f"({lo:g}-{hi:g})")

    def on_auto_exposure_toggled(self, checked: bool) -> None:
        """Switch the camera between auto and manual exposure."""
        if not self.videothread or type(self.videothread.cap) is not cam.ImageAcquirer:
            return
        if self.profile.auto_exposure_node is None:
            return
        cam.set_attribute(
            self.videothread.cap,
            self.profile.auto_exposure_node,
            camera_profiles.AUTO_ON if checked else camera_profiles.AUTO_OFF,
        )
        self.ui.sb_exposure.setEnabled(not checked)
        if not checked:
            self.on_exposure_changed()  # push the manual value again

    def on_auto_gain_toggled(self, checked: bool) -> None:
        """Switch the camera between auto and manual gain."""
        if not self.videothread or type(self.videothread.cap) is not cam.ImageAcquirer:
            return
        if self.profile.auto_gain_node is None:
            return
        cam.set_attribute(
            self.videothread.cap,
            self.profile.auto_gain_node,
            camera_profiles.AUTO_ON if checked else camera_profiles.AUTO_OFF,
        )
        self.ui.sb_analog_gain.setEnabled(not checked)
        if not checked:
            self.on_analog_gain_changed()  # push the manual value again

    def setup_resolution_settings(self) -> None:
        """Populate the resolution row and re-apply a saved image size.

        Runs after connecting a Harvester camera. Reads the Width/Height node
        ranges, applies any resolution persisted by an earlier session, and
        syncs the spinboxes to what the camera actually reports.
        """
        ia = self.videothread.cap
        node_map = ia.remote_device.node_map
        for sb, node_name in (
            (self.ui.sb_res_width, "Width"),
            (self.ui.sb_res_height, "Height"),
        ):
            try:
                node = getattr(node_map, node_name)
                sb.setMinimum(int(node.min))
                sb.setMaximum(int(node.max))
                sb.setDisabled(False)
            except Exception as e:
                logging.warning(f"Resolution {node_name} not available: {e}")
                sb.setDisabled(True)
        cfg = cam.load_resolution_config()
        if isinstance(cfg.get("width"), int) and isinstance(cfg.get("height"), int):
            w, h, ox, oy = cam.set_resolution(
                ia,
                width=cfg["width"],
                height=cfg["height"],
                offset_x=cfg.get("offset_x"),
                offset_y=cfg.get("offset_y"),
            )
            self.ui.sb_res_width.setValue(w)
            self.ui.sb_res_height.setValue(h)
            logging.info(f"Reapplied saved resolution {w}x{h}")
        else:
            try:
                self.ui.sb_res_width.setValue(int(node_map.Width.value))
                self.ui.sb_res_height.setValue(int(node_map.Height.value))
            except Exception as e:
                logging.warning(f"Could not read resolution: {e}")
        self.ui.pb_res_apply.setDisabled(False)

    def on_resolution_apply(self) -> None:
        """Apply the Width x Height from the settings panel to the camera.

        The camera is stopped for the AOI writes and restarted if it was
        acquiring; the result is persisted so the chosen image size survives
        reconnect/restart. Live view, recordings, and saved files all follow
        the applied size.
        """
        if not self.videothread or type(self.videothread.cap) is not cam.ImageAcquirer:
            logging.debug("Resolution apply requires a Harvester camera")
            return
        width = self.ui.sb_res_width.value()
        height = self.ui.sb_res_height.value()
        w, h, ox, oy = cam.set_resolution(
            self.videothread.cap, width=width, height=height
        )
        self.ui.sb_res_width.setValue(w)
        self.ui.sb_res_height.setValue(h)
        cam.save_resolution_config(
            {"width": w, "height": h, "offset_x": ox, "offset_y": oy}
        )
        logging.info(f"Resolution set to {w}x{h}")

    def on_settings_pressed(self) -> None:
        logging.debug("Settings pressed")
        self.ui.settings_window.show()

    def open_file(self) -> None:
        """Open a file dialog to load a video file."""
        if self.videothread:
            self.pause_video()
            self.Image.clear()
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if filter == "HDF5 Files (*.h5)":
            self.recording, self.metadata = importexport.load_frames_from_h5(
                filename, progressbar=self.ui.progressBar
            )
            self.recording_id = self.metadata.get("recording_id")
            self.recording_timestamp = self.metadata.get("timestamp")
        self.update_image(self.recording)
        self.filename = filename
        self.ui.pb_save.setDisabled(False)

    def find_cti_dialog(self) -> None:
        """Open a dialog to find the GenTL Producer file."""
        if self.videothread is not None:
            # populate_video_sources() below would reset the producer the
            # connected camera runs through; say so before opening the dialog.
            QtWidgets.QMessageBox.information(
                self,
                "Camera connected",
                "Disconnect the current camera before adding a GenTL producer.",
            )
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GenTL Producer", "", "GenTL Producer (*.cti)"
        )
        if filename and filename not in self.ctis:
            self.ctis.append(filename)
        self.populate_video_sources()

    def on_histogram_checked(self) -> None:
        state = self.ui.cb_histogram.isChecked()
        self.ui.ImageView.getHistogramWidget().setHidden(not state)

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
        if pb.text() == "Play":
            pb.setText("Pause")
        else:
            pb.setText("Play")


def configure_logging(verbose: int = 0, log_file: str = "harvesters_pg.log") -> None:
    """Send logs to the console and to *log_file*.

    This used to live in the ``if __name__ == "__main__"`` block, so the
    installed ``harvesters_pg`` console script - which calls main() directly -
    ran with no logging configured at all and wrote an empty log file. Every
    warning about a stuck worker, a leaked camera handle or a fetch timeout
    was silently discarded, which is exactly the evidence needed to diagnose a
    freeze. It is set up in main() now, so both entry points get it.

    The thread name is in the format on purpose: when the app hangs, the last
    line tells you whether the GUI thread or the capture worker got there last.
    FileHandler flushes on every record, so nothing is lost to a hard freeze.
    """
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
    # Add argument to set the log level by using -v or -vv or -vvv
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="count",
        default=0,
    )
    parser.add_argument(
        "--log-file",
        default="harvesters_pg.log",
        help="where to write the log (default: harvesters_pg.log in the working directory)",
    )
    args = parser.parse_args()
    configure_logging(args.verbose, args.log_file)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
