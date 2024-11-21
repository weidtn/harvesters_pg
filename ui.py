#!/usr/bin/env python3
import sys

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets

import cam
from MainWindow import Ui_MainWindow
from VideoThread import VideoThread

class MainWindow(QtWidgets.QMainWindow):
    """Main window for the application.

    The UI is created using Qt Designer and converted to python using pyuic6.
    The VideoThread class is used to capture video frames and emit them to the main thread.
    The video feed is displayed using pyqtgraph's ImageItem.
    The user can select the video source from a combobox and connect to the source using the "Connect" button.
    The "Start"/"Stop" button starts and pauses the video feed.
    The "Record" button is disabled for now, but will be used to record a given number of frames into a hdf5 file.

    Attributes:
        ui (Ui_MainWindow): The UI object created using Qt Designer
        image_item (pg.ImageItem): The ImageItem to display the video feed
        videothread (VideoThread | None): The worker thread to capture video frames
        is_running (bool): Flag to control the video feed
        source (str): The video source
        cam_indices (list[int]): List of available camera indices for opencv
    """

    def __init__(self, *args, **kwargs) -> None:
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Setup and add the ImageItem for displaying the video feed
        self.image_item = pg.ImageItem()
        self.ui.graphicsView.addItem(self.image_item)
        self.videothread: None | VideoThread = None

        # Connect the buttons to their respective functions
        self.ui.pb_connect.clicked.connect(self.connect_video)
        self.ui.pb_live.clicked.connect(self.toggle_video)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.pb_live.setDisabled(True)
        self.ui.lbl_status.setStyleSheet("color: red;")

        # Initialize the video source and set the default source to "random"
        self.is_running: bool = False
        self.source: str = "random"
        self.ui.cb_source.addItem("random")

        # Detect all available cameras for opencv:
        self.cam_indices: list[int] = cam.returnCameraIndices()
        # Add the available opencv video sources to the combobox
        [self.ui.cb_source.addItem(f"Webcam {i}") for i in self.cam_indices]

        # Detect all available cameras for harvesters:
        self.harvester_cams: list[dict] = cam.returnHarvesterCameras()
        [self.ui.cb_source.addItem(f"{cam['display_name']}") for cam in self.harvester_cams if cam["model"] != "Unknown Model"]

        # TODO:
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_settings.setDisabled(True)
        self.ui.le_analog_gain.setDisabled(True)
        self.ui.le_exposure.setDisabled(True)
        self.ui.le_fps.setDisabled(True)
        self.ui.le_frames.setDisabled(True)
        self.ui.le_seconds.setDisabled(True)
        self.ui.le_black_level.setDisabled(True)
        self.ui.progressBar.setDisabled(True)

    def connect_video(self) -> None:
        """Read the video source and start the video thread"""
        if not self.videothread:
            self.source = self.ui.cb_source.currentText()
            self.videothread = VideoThread(self.source)
            self.videothread.change_pixmap_signal.connect(self.update_image)
            self.ui.pb_live.setDisabled(False)
            self.ui.pb_connect.setText("Disconnect")
            self.ui.pb_connect.clicked.disconnect()
            self.ui.pb_connect.clicked.connect(self.disconnect_video)
            self.ui.cb_source.setDisabled(True)

    def disconnect_video(self) -> None:
        """Stop the video thread and reset the UI"""
        if self.videothread:
            self.videothread.stop()
            self.videothread = None
            self.is_running = False
            self.ui.pb_live.setDisabled(True)
            self.ui.pb_connect.setText("Connect")
            self.ui.pb_connect.clicked.disconnect()
            self.ui.pb_connect.clicked.connect(self.connect_video)
            self.ui.cb_source.setDisabled(False)
            self.ui.pb_live.setText("Start")
            self.ui.lbl_status.setStyleSheet("color: red;")

    def start_video(self) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")
        self.videothread.start()
        self.videothread.resume()
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
        self.image_item.setImage(frame)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
