"""This is the ViewModel in the MVVM pattern.

All user interaction with the ui is handled in here and communicated to the modle (cam.py)
"""
import os
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
import sys
from PyQt6 import QtWidgets
import numpy as np
import datetime

import cam
from MainWindow import Ui_MainWindow
from VideoThread import VideoThread
import importexport


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

        # Setup and add the ImageItem for displaying the video feed
        self.ui.ImageView.getHistogramWidget().setHidden(True)
        self.ui.ImageView.ui.menuBtn.setHidden(True)
        self.ui.ImageView.ui.roiBtn.setHidden(True)
        self.ui.ImageView.getView().setAspectLocked(True)
        self.Image = self.ui.ImageView
        # Disable menu
        self.ui.ImageView.getView().setMenuEnabled(False)

        # Initialize variables
        self.videothread: None | VideoThread = None
        self.is_running: bool = False
        self.source: str = "random"
        self.recording: np.ndarray = np.array([])
        self.ctis: list[str] = []

        self.setup_ui_components()

    def setup_ui_components(self) -> None:
        """Setup UI components and connect signals."""
        # Connect the buttons to their respective functions
        self.ui.pb_connect.clicked.connect(self.connect_video)
        self.ui.pb_live.clicked.connect(self.toggle_video)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.pb_acquire.clicked.connect(self.on_btn_acquire_clicked)
        self.ui.pb_acquire.setDisabled(True)
        self.ui.pb_save.clicked.connect(self.save)
        self.ui.sb_frames.editingFinished.connect(self.on_frames_changed)
        self.on_frames_changed() # run once to setup the seconds display:
        self.ui.sb_seconds.editingFinished.connect(self.on_seconds_changed)
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionSelect_GenTLProducer.triggered.connect(self.find_cti_dialog)

        # Disable certain elements initially
        self.ui.pb_live.setDisabled(True)
        self.ui.lbl_status.setStyleSheet("color: red;")

        self.ui.settings.setHidden(True)
        self.ui.sb_fps.editingFinished.connect(self.on_fps_changed)
        self.ui.sb_exposure.editingFinished.connect(self.on_exposure_changed)
        self.ui.sb_analog_gain.editingFinished.connect(self.on_analog_gain_changed)
        self.ui.sb_black_level.editingFinished.connect(self.on_blacklevel_changed)

        self.populate_video_sources()

    def populate_video_sources(self) -> None:
        """Populate combobox with available video sources."""

        # Add OpenCV camera indices
        cam_indices: list[int] = cam.get_camera_indices()
        [self.ui.cb_source.addItem(f"Webcam {i}") for i in cam_indices]

        # Add Harvester cameras
        self.ctis.extend(cam.find_common_cti_files())
        harvester_cams: list[dict] = cam.return_harvester_cameras(self.ctis)
        [
            self.ui.cb_source.addItem(f"{cam['display_name']}")
            for cam in harvester_cams
            if cam["model"] != "Unknown Model"
        ]
        self.ui.cb_source.addItem("random")

    def connect_video(self) -> None:
        """Read the video source and start the video thread"""
        if not self.videothread:
            self.source = self.ui.cb_source.currentText()
            self.videothread = VideoThread(self.source)
            self.videothread.change_pixmap_signal.connect(self.update_image)
            self.ui.pb_live.setDisabled(False)
            self.ui.pb_acquire.setDisabled(False)
            self.ui.pb_connect.setText("Disconnect")
            self.ui.pb_connect.clicked.disconnect()
            self.ui.pb_connect.clicked.connect(self.disconnect_video)
            self.ui.cb_source.setDisabled(True)
            self.on_source_connected()

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
            self.ui.pb_acquire.setDisabled(True)
            self.ui.pb_live.setText("Start")
            self.ui.lbl_status.setStyleSheet("color: red;")
            self.ui.settings.setHidden(True)

    def start_video(self) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")
        self.ui.pb_save.setDisabled(True)
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
        self.Image.setImage(frame, autoRange=False, autoHistogramRange=False)

    def on_btn_acquire_clicked(self) -> None:
        self.ui.pb_save.setDisabled(True)
        n_frames = int(self.ui.sb_frames.text())
        self.pause_video()
        self.acquire(n_frames, self.ui.progressBar)
        self.ui.pb_save.setDisabled(False)
        self.update_image(self.recording)

    def toggle_settings(self) -> None:
        if self.ui.settings.isHidden():
            self.ui.settings.setHidden(False)
        else:
            self.ui.settings.setHidden(True)

    def acquire(self,n_frames: int, progressbar: None | QtWidgets.QProgressBar) -> None:
        if not self.videothread:
            raise ValueError("Video thread not started")
        if self.source == "random":
            imgs = cam.record_random_frames(n_frames, progressbar)
        elif "Webcam" in self.source:
            imgs = cam.record_webcam_frames(self.videothread.cap, n_frames, progressbar)
        else:
            imgs = cam.record_harvester_frames(self.videothread.cap, n_frames, progressbar)
        self.recording = imgs


    def save(self) -> None:
        """Save the recorded frames to a hdf5 file."""
        # Get filename with a file dialog
        filename, filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save File", "", "HDF5 Files (*.h5);;Images (*.png);;All Files (*)"
        )
        metadata: dict = {"timestamp": datetime.datetime.now().strftime("%Y%m%d%H%M%S")}
        if type(self.videothread.cap) is cam.VideoCapture:
            metadata["fps"] = cam.get_webcam_fps(self.videothread.cap)
        elif self.source == "random":
            metadata["fps"] = 30
        elif type(self.videothread.cap) is cam.ImageAcquirer:
            nodemap = cam.get_all_attributes_harvesters(self.videothread.cap)
            [metadata.update({k: v}) for k, v in nodemap.items()]
            metadata["fps"] = self.ui.sb_fps.value()
        self.ui.progressBar.setMaximum(0)
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setFormat("Saving...")
        if filter == "HDF5 Files (*.h5)":
            if ".h5" not in filename:
                filename += ".h5"
            importexport.save_frames_to_h5(self.recording, filename, metadata=metadata)
        elif filter == "Images (*.png)":
            if ".png" not in filename:
                filename += ".png"
            importexport.save_frames_to_png(self.recording, filename)
        self.ui.progressBar.setMaximum(1)
        self.ui.progressBar.setValue(1)
        self.ui.progressBar.setFormat("%p%")
        print(f"Saved recording to {filename}")

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
        if "random" in self.source:
            self.ui.settings.setHidden(True)
        if "Webcam" in self.source:
            self.ui.settings.setHidden(True)
        if "SVS-Vistek" in self.source:
            self.setup_settings() # get minmax values for settings
            self.ui.settings.setHidden(False)

    def on_fps_changed(self)-> None:
        """Update the fps based on the settings."""
        self.videothread.cap.remote_device.node_map.AcquisitionFrameRate.value = self.ui.sb_fps.value()
        self.on_frames_changed()

    def on_exposure_changed(self)-> None:
        """Update the exposure based on the settings."""
        self.videothread.cap.remote_device.node_map.ExposureTime.value = self.ui.sb_exposure.value()

    def on_analog_gain_changed(self)-> None:
        """Update the gain based on the settings."""
        self.videothread.cap.remote_device.node_map.AnalogGain.value = f"x{self.ui.sb_analog_gain.value()}"

    def on_blacklevel_changed(self)-> None:
        """Update the blacklevel based on the settings."""
        self.videothread.cap.remote_device.node_map.BlackLevel.value = self.ui.sb_black_level.value()


    def setup_settings(self) -> None:
        ia = self.videothread.cap
        node_map = ia.remote_device.node_map
        self.ui.sb_fps.setMaximum(node_map.AcquisitionFrameRateMax.value)


    def open_file(self) -> None:
        """Open a file dialog to load a video file."""
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open File", "", "HDF5 Files (*.h5);;All Files (*)"
        )
        if filter == "HDF5 Files (*.h5)":
            self.recording = importexport.load_frames_from_h5(filename)
        self.update_image(self.recording)

    def find_cti_dialog(self)-> None:
        """ Open a dialog to find the GenTL Producer file."""
        filename, filter = QtWidgets.QFileDialog.getOpenFileName(self, "Select GenTL Producer", "", "GenTL Producer (*.cti)")
        if filename:
            self.ctis.append(filename)
            self.populate_video_sources()

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
