#!/usr/bin/env python3
"""Contract tests for the recording-speed + node-guard fixes.

Frames are stored and displayed in the camera's natural (height, width)
orientation throughout the app (single exceptions: none). These tests pin the
new observable contracts:
  * info-field extraction reads the top-left pixels of the (h, w) frame,
  * h5 save/load round-trips preserve (n, h, w) without any transpose and
    the recording identity (recording_id, timestamp) for re-saving,
  * PNG export writes the upright (h, w) frame,
  * setup_cam tolerates cameras without AcquisitionFrameRate/Max or other
    optional GenICam nodes (e.g. the bundled TLSimu producer),
  * record_harvester_frames yields an owned, C-contiguous (n, h, w) stack.

Run:  .venv/bin/python -m unittest test_cam -v
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import cam
import importexport


class FrameOrientationTests(unittest.TestCase):
    """Non-hardware contracts (pure numpy / filesystem)."""

    def test_extract_frame_counter_reads_top_left_of_hw(self):
        img = np.zeros((64, 48), dtype=np.uint8)  # (h, w)
        img.flat[0:4] = (0x01, 0x02, 0x03, 0xAB)
        self.assertEqual(cam.extract_frame_counter(img), 0x00030201)

    def test_extract_timestamp_reads_pixels_4_7_of_hw(self):
        img = np.zeros((64, 48), dtype=np.uint8)
        img.flat[4:8] = (0x01, 0x02, 0x03, 0x04)
        # little endian: (4<<24) | (3<<16) | (2<<8) | 1
        self.assertEqual(cam.extract_timestamp(img), 0x04030201)

    def test_h5_roundtrip_preserves_nhw(self):
        frames = np.ones((5, 40, 30), dtype=np.uint8)  # (n, h, w)
        for i in range(5):
            frames[i] = (i * 11 + np.arange(40 * 30)).reshape(40, 30) % 256
        out = Path(__file__).parent / "test_frames_roundtrip.h5"
        importexport.save_frames_to_h5(frames, str(out), complevel=0)
        try:
            loaded, _ = importexport.load_frames_from_h5(str(out))
        finally:
            out.unlink(missing_ok=True)
        self.assertEqual(loaded.shape, (5, 40, 30))
        np.testing.assert_array_equal(loaded, frames)

    def test_h5_roundtrip_keeps_recording_identity(self):
        # A loaded recording that is saved again must keep its id and start time.
        identity = {
            "recording_id": "1b4e28ba-2fa1-11d2-883f-0016d3cca427",
            "timestamp": "20260415144348",
        }
        out = Path(__file__).parent / "test_identity_roundtrip.h5"
        importexport.save_frames_to_h5(
            np.zeros((2, 4, 3), dtype=np.uint8), str(out), metadata=dict(identity)
        )
        try:
            _, metadata = importexport.load_frames_from_h5(str(out))
        finally:
            out.unlink(missing_ok=True)
        self.assertEqual({k: metadata.get(k) for k in identity}, identity)

    def test_png_export_is_upright_hw(self):
        frame = np.zeros((40, 30), dtype=np.uint8)
        frame[7, 13] = 255  # asymmetric marker: row 7, col 13
        base = Path(__file__).parent / "test_frames_export"
        importexport.save_frames_to_png(frame, str(base))
        try:
            import cv2

            back = cv2.imread(str(base / "test_frames_export.png"), cv2.IMREAD_GRAYSCALE)
        finally:
            (base / "test_frames_export.png").unlink(missing_ok=True)
            base.rmdir()
        self.assertEqual(back.shape, (40, 30))  # not transposed
        self.assertEqual(back[7, 13], 255)

    def test_snap_to_increment(self):
        class FakeNode:
            min = 8
            max = 4096
            inc = 4

        node = FakeNode()
        self.assertEqual(cam._snap_to_increment(node, 160), 160)  # already on grid
        self.assertEqual(cam._snap_to_increment(node, 98), 96)  # snap down
        self.assertEqual(cam._snap_to_increment(node, 102), 100)
        self.assertEqual(cam._snap_to_increment(node, 2), 8)  # clamp to min
        self.assertEqual(cam._snap_to_increment(node, 99999), 4096)  # clamp to max

    def test_resolution_config_is_kept_per_camera(self):
        original = cam.RESOLUTION_CONFIG_PATH
        cam.RESOLUTION_CONFIG_PATH = Path(__file__).parent / "test_res_config.json"
        try:
            cam.save_resolution_config(
                "Mikrotron/EoSens/1", {"width": 160, "height": 120, "offset_x": None, "offset_y": 4}
            )
            cam.save_resolution_config("Allied/Mako/2", {"width": 640, "height": 480})
            self.assertEqual(
                cam.load_resolution_config("Mikrotron/EoSens/1"),
                {"width": 160, "height": 120, "offset_y": 4},
            )
            self.assertEqual(
                cam.load_resolution_config("Allied/Mako/2"), {"width": 640, "height": 480}
            )
            # A camera that never saved one must not inherit another camera's AOI.
            self.assertEqual(cam.load_resolution_config("Other/Model/3"), {})
        finally:
            cam.RESOLUTION_CONFIG_PATH.unlink(missing_ok=True)
            cam.RESOLUTION_CONFIG_PATH = original


class CtiDiscoveryTests(unittest.TestCase):
    """GENICAM_GENTL64_PATH parsing and well-known-location fallback.

    Allied Vision's Install_GenTL_Path.sh registers colon-separated paths;
    Vimba X tarballs also work unregistered via the well-known globs.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="cti_test_"))
        self._orig_well_known = cam._WELL_KNOWN_GENTL_DIRS
        cam._WELL_KNOWN_GENTL_DIRS = []
        os.environ.pop("GENICAM_GENTL64_PATH", None)

    def tearDown(self):
        os.environ.pop("GENICAM_GENTL64_PATH", None)
        cam._WELL_KNOWN_GENTL_DIRS = self._orig_well_known
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_cti_dir(self, name: str) -> str:
        d = self._tmp / name
        d.mkdir()
        (d / f"{name}.cti").write_bytes(b"")
        return str(d)

    def test_single_dir_in_env_var(self):
        d = self._make_cti_dir("gentl_a")
        os.environ["GENICAM_GENTL64_PATH"] = d
        self.assertEqual(cam.load_cti_envvar(), [os.path.join(d, "gentl_a.cti")])

    def test_colon_separated_env_var_entries(self):
        a = self._make_cti_dir("gentl_a")
        b = self._make_cti_dir("gentl_b")
        os.environ["GENICAM_GENTL64_PATH"] = ":".join([a, "", b, ""])
        self.assertEqual(
            cam.load_cti_envvar(),
            [os.path.join(a, "gentl_a.cti"), os.path.join(b, "gentl_b.cti")],
        )

    def test_repeated_dir_reported_once(self):
        d = self._make_cti_dir("gentl_a")
        os.environ["GENICAM_GENTL64_PATH"] = f"{d}:{d}"
        self.assertEqual(cam.load_cti_envvar(), [os.path.join(d, "gentl_a.cti")])

    def test_well_known_locations_fallback_glob(self):
        root = self._tmp / "opt"
        cti_dir = root / "alliedvision" / "VimbaX_2026-2" / "cti"
        cti_dir.mkdir(parents=True)
        (cti_dir / "VimbaGigETL.cti").write_bytes(b"")
        cam._WELL_KNOWN_GENTL_DIRS = [str(root / "alliedvision" / "VimbaX*" / "cti")]
        self.assertEqual(
            cam.load_cti_envvar(), [str(cti_dir / "VimbaGigETL.cti")]
        )

    def test_env_var_takes_precedence_over_well_known(self):
        d = self._make_cti_dir("gentl_first")
        os.environ["GENICAM_GENTL64_PATH"] = d
        cam._WELL_KNOWN_GENTL_DIRS = [str(self._tmp / "nowhere")]
        self.assertEqual(cam.load_cti_envvar(), [os.path.join(d, "gentl_first.cti")])


class _FakeComponent:
    """Minimal stand-in for harvesters' Component2DImage (read-only attrs)."""

    def __init__(self, fmt, data, width, height):
        self.data_format = fmt
        self.data = data
        self.width = width
        self.height = height


class _FakeBuffer:
    """Stand-in for a fetched harvesters Buffer: a context manager whose exit
    is the queue() that hands it back to the producer."""

    def __init__(self, component, on_exit=None):
        self.payload = SimpleNamespace(components=[component])
        self._on_exit = on_exit

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._on_exit is not None:
            self._on_exit()


def _mono8_buffer(value, width=2, height=2, on_exit=None):
    data = np.full(width * height, value, np.uint8)
    return _FakeBuffer(_FakeComponent("Mono8", data, width, height), on_exit=on_exit)


def _stalled_fetch(timeout):
    """What try_fetch does on a stream that delivers nothing: wait, then None."""
    time.sleep(timeout)
    return None


class PixelFormatTests(unittest.TestCase):
    """Every delivered pixel format must become an owned uint8 (h, w).

    harvesters expands packed formats (Mono12Packed, Bayer*10Packed, ...) to
    natural uint8/uint16 arrays before we see them, so these tests feed the
    same already-expanded payloads a real delivery would provide.
    """

    def test_mono8_is_owned_identity(self):
        data = np.arange(12, dtype=np.uint8)
        img = cam.grayscale8_from_bytes("Mono8", data, 4, 3)
        self.assertEqual(img.shape, (3, 4))
        self.assertEqual(img.dtype, np.uint8)
        np.testing.assert_array_equal(img, data.reshape(3, 4))
        self.assertFalse(np.shares_memory(img, data))  # owned copy
        self.assertTrue(img.flags["C_CONTIGUOUS"])

    def test_mono_bitshift(self):
        for fmt, bits, values in (
            ("Mono10", 10, np.array([0, 1023, 512, 1], np.uint16)),
            ("Mono12", 12, np.array([4095, 2048, 17], np.uint16)),
            ("Mono14", 14, np.array([16383, 8192], np.uint16)),
            ("Mono16", 16, np.array([0xFFFF, 0x8000, 0x1234], np.uint16)),
        ):
            self.assertEqual(
                cam.grayscale8_from_bytes(fmt, values, len(values), 1).reshape(-1).tolist(),
                (values >> (bits - 8)).tolist(),
                fmt,
            )

    def test_packed_mono_bitshift(self):
        # Delivered payloads are pre-expanded to uint16, one value per pixel.
        for fmt in ("Mono10Packed", "Mono10p"):
            values = np.array([1023, 512, 0], np.uint16)
            self.assertEqual(
                cam.grayscale8_from_bytes(fmt, values, 3, 1).reshape(-1).tolist(),
                (values >> 2).tolist(),
            )
        for fmt in ("Mono12Packed", "Mono12p"):
            values = np.array([4095, 2048, 1], np.uint16)
            self.assertEqual(
                cam.grayscale8_from_bytes(fmt, values, 3, 1).reshape(-1).tolist(),
                (values >> 4).tolist(),
            )

    def test_bayer8_demosaic_matches_cv2(self):
        import cv2

        mosaic = (np.arange(8 * 8) * 7) % 256
        mosaic = mosaic.reshape(8, 8).astype(np.uint8)
        # GenICam names the top-left 2x2 cell; OpenCV's matching codes are the
        # explicit four-letter ones (its short names are shifted by one).
        for pattern, code in (
            ("BayerGR8", cv2.COLOR_BayerGRBG2GRAY),
            ("BayerRG8", cv2.COLOR_BayerRGGB2GRAY),
            ("BayerGB8", cv2.COLOR_BayerGBRG2GRAY),
            ("BayerBG8", cv2.COLOR_BayerBGGR2GRAY),
        ):
            self.assertEqual(
                cam.grayscale8_from_bytes(pattern, mosaic, 8, 8).tolist(),
                cv2.cvtColor(mosaic, code).tolist(),
                pattern,
            )

    def test_bayer_red_scene_is_weighted_as_red(self):
        # BayerRG8: R at even rows/even columns, B at odd rows/odd columns.
        red = np.zeros((8, 8), np.uint8)
        red[0::2, 0::2] = 255
        blue = np.zeros((8, 8), np.uint8)
        blue[1::2, 1::2] = 255
        red_luma = cam.grayscale8_from_bytes("BayerRG8", red, 8, 8).mean()
        blue_luma = cam.grayscale8_from_bytes("BayerRG8", blue, 8, 8).mean()
        # Luminance weighs red (0.299) far above blue (0.114). With OpenCV's
        # shifted short names the two were swapped.
        self.assertGreater(red_luma, 2 * blue_luma)

    def test_bayer10_16bit_demosaic(self):
        import cv2

        mosaic16 = (np.arange(8 * 8).reshape(8, 8) * 512).astype(np.uint16)
        mosaic8 = (mosaic16 >> 2).astype(np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("BayerRG10", mosaic16, 8, 8).tolist(),
            cv2.cvtColor(mosaic8, cv2.COLOR_BayerRGGB2GRAY).tolist(),
        )
        mosaic16b = (np.arange(8 * 8).reshape(8, 8) * 32).astype(np.uint16)
        self.assertEqual(
            cam.grayscale8_from_bytes("BayerBG12Packed", mosaic16b, 8, 8).tolist(),
            cv2.cvtColor((mosaic16b >> 4).astype(np.uint8), cv2.COLOR_BayerBGGR2GRAY).tolist(),
        )

    def test_rgb_and_rgba_match_cv2(self):
        import cv2

        rgb = np.arange(3 * 4 * 2, dtype=np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("RGB8", rgb, 4, 2).tolist(),
            cv2.cvtColor(rgb.reshape(2, 4, 3), cv2.COLOR_RGB2GRAY).tolist(),
        )
        bgr = np.arange(3 * 4 * 2, dtype=np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("BGR8", bgr, 4, 2).tolist(),
            cv2.cvtColor(bgr.reshape(2, 4, 3), cv2.COLOR_BGR2GRAY).tolist(),
        )
        rgba = np.arange(4 * 4 * 2, dtype=np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("RGBA8", rgba, 4, 2).tolist(),
            cv2.cvtColor(rgba.reshape(2, 4, 4), cv2.COLOR_RGBA2GRAY).tolist(),
        )

    def test_yuv_luma_extraction(self):
        # UYVY rows: U,Y0,V,Y1 per pixel pair -> luma at odd bytes.
        row = np.array([0, 10, 1, 20, 2, 30, 3, 40], np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("YUV422Packed", row, 4, 1).tolist(),
            [[10, 20, 30, 40]],
        )
        # Y,U,V per pixel -> luma is the first byte.
        yuv444 = np.array([1, 9, 9, 2, 9, 9, 3, 9, 9], np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("YUV444Packed", yuv444, 3, 1).tolist(),
            [[1, 2, 3]],
        )
        # U,Y0,Y1,V,Y2,Y3 per 4-pixel group -> luma at bytes 1,2,4,5.
        yuv411 = np.tile([5, 10, 20, 30, 40, 50], 6).astype(np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("YUV411Packed", yuv411, 12, 2).tolist(),
            (np.tile([10, 20, 40, 50], 6)).reshape(2, 12).tolist(),
        )

    def test_pfnc_yuv_names_extract_luma(self):
        # Current cameras report PFNC names; these used to degrade to black.
        yuyv = np.array([10, 0, 20, 1, 30, 2, 40, 3], np.uint8)  # Y0 U Y1 V
        uyvy = np.array([0, 10, 1, 20, 2, 30, 3, 40], np.uint8)  # U Y0 V Y1
        for fmt, data in (
            ("YUV422_8", yuyv),
            ("YCbCr422_8", yuyv),
            ("YUV422_8_UYVY", uyvy),
            ("YCbCr422_8_CbYCrY", uyvy),
        ):
            self.assertEqual(
                cam.grayscale8_from_bytes(fmt, data, 4, 1).tolist(), [[10, 20, 30, 40]], fmt
            )
        cbycr = np.array([9, 1, 9, 9, 2, 9, 9, 3, 9], np.uint8)  # Cb Y Cr per pixel
        self.assertEqual(
            cam.grayscale8_from_bytes("YCbCr8_CbYCr", cbycr, 3, 1).tolist(), [[1, 2, 3]]
        )
        uyyvyy = np.tile([5, 10, 20, 30, 40, 50], 2).astype(np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("YUV411_8_UYYVYY", uyyvyy, 8, 1).tolist(),
            [[10, 20, 40, 50, 10, 20, 40, 50]],
        )

    def test_unsupported_format_mismatch_raises(self):
        # A 3-element payload can never be a 2x1 frame; must raise, not guess.
        with self.assertRaises(ValueError):
            cam.grayscale8_from_bytes("Bogus24", np.arange(3, dtype=np.uint8), 2, 1)
        with self.assertRaises(ValueError):
            cam.grayscale8_from_bytes("Coord3D_A32f", np.arange(4, dtype=np.float32), 2, 2)

    def test_component_wrapper_black_fallback_and_valid_path(self):
        cam._warned_formats.clear()
        bad = _FakeComponent("Killer32", np.arange(5, dtype=np.uint8), 4, 3)
        img = cam.component_to_grayscale8(bad)
        self.assertEqual(img.shape, (3, 4))
        self.assertEqual(img.tolist(), np.zeros((3, 4), np.uint8).tolist())
        self.assertIn("Killer32", cam._warned_formats)
        good = _FakeComponent("Mono8", np.arange(12, dtype=np.uint8) % 4, 4, 3)
        np.testing.assert_array_equal(
            cam.component_to_grayscale8(good), (np.arange(12) % 4).reshape(3, 4)
        )


# All TLSimu hardware interactions live in ONE test to avoid the producer's
# flaky in-process re-enumeration.
class TlSimuTests(unittest.TestCase):
    """Hardware-backed contracts via the bundled TLSimu GenTL producer."""

    @classmethod
    def setUpClass(cls):
        import genicam

        cls.cti = Path(genicam.__file__).parent / "TLSimu.cti"
        if not cls.cti.exists():
            raise unittest.SkipTest(f"TLSimu.cti not found: {cls.cti}")

    def tearDown(self):
        # Don't leak a loaded producer into the next test.
        cam.release_harvester()

    def test_setup_cam_guards_missing_nodes_and_records(self):
        h, ia = cam.setup_cam(cti=str(self.cti), fps=5)  # must not raise
        try:
            self.assertIsNotNone(ia)
            # setup_cam leaves the camera started, so the live path can fetch
            # right away (fetch while stopped would block forever).
            live = cam.get_harvester_image(ia)
            self.assertEqual(live.shape, (ia.remote_device.node_map.Height.value,
                                          ia.remote_device.node_map.Width.value))
            self.assertTrue(live.flags["OWNDATA"])
            # AOI change WHILE acquiring: Width/Height are not writable during
            # acquisition on TLSimu, so set_resolution must stop -> write ->
            # start; the live path then delivers the new size.
            w0 = int(ia.remote_device.node_map.Width.value)
            h0 = int(ia.remote_device.node_map.Height.value)
            aw, ah, _, _ = cam.set_resolution(ia, width=64, height=48)
            self.assertEqual((aw, ah), (64, 48))  # 64/48 are on the inc=4 grid
            live2 = cam.get_harvester_image(ia)
            self.assertEqual(live2.shape, (ah, aw))
            # record at the new size: owned, C-contiguous (n, h, w) stack.
            frames = cam.record_harvester_frames(ia, 2, None)
            self.assertEqual(frames.shape, (2, 48, 64))
            self.assertTrue(frames.flags["C_CONTIGUOUS"])
            self.assertTrue(frames.flags["OWNDATA"])
            self.assertFalse(np.isfortran(frames[0]))
            # After record the camera is stopped; an off-grid request still
            # snaps onto the increment grid and restores cleanly (no fetch).
            cam.set_resolution(ia, width=66, height=46)
            self.assertEqual(int(ia.remote_device.node_map.Width.value), 64)
            self.assertEqual(int(ia.remote_device.node_map.Height.value), 44)
            cam.set_resolution(ia, width=w0, height=h0)
        finally:
            cam.disconnect_cam(h, ia)

    def test_reconnect_cycles_release_the_device(self):
        """Connect/disconnect repeatedly on a device that is not index 0.

        setup_cam() used to ignore the picked source (always create(0)) and
        leave teardown to VideoThread.__del__, so a device could still be
        exclusively opened when the next connect ran - the software-only
        "camera is wedged until I replug it" symptom. The cycle must also reuse
        one producer load rather than unloading and reloading each time.
        """
        h = cam.get_harvester([str(self.cti)])
        h.update()
        names = [device.display_name for device in h.device_info_list]
        self.assertGreater(len(names), 1, "TLSimu should expose several devices")

        target = names[-1]  # deliberately not index 0
        harvesters_seen = set()
        for attempt in range(3):
            harvester, ia = cam.setup_cam(
                cti=[str(self.cti)], search_key={"display_name": target}, fps=5
            )
            harvesters_seen.add(id(harvester))
            try:
                self.assertIsNotNone(ia, f"connect {attempt} produced no camera")
                frame = cam.get_harvester_image(ia)
                self.assertEqual(frame.dtype, np.uint8)
                self.assertEqual(frame.ndim, 2)
            finally:
                # Exactly what a disconnect does now: release the device, leave
                # the producers loaded.
                cam.teardown_cam(None, ia)
        self.assertEqual(len(harvesters_seen), 1, "producer reloaded per cycle")

    def test_setup_cam_releases_the_device_on_an_unknown_search_key(self):
        # create() raises for a key that matches nothing; the harvester and the
        # producer it loaded must still be released, or the .cti stays loaded.
        with self.assertRaises(Exception):
            cam.setup_cam(
                cti=str(self.cti), search_key={"display_name": "no such camera"}
            )
        # The producer is free again, so a normal connect still works.
        h, ia = cam.setup_cam(cti=str(self.cti), fps=5)
        cam.teardown_cam(h, ia)


_SHARED_WINDOW = None


def _shared_main_window():
    """One MainWindow, reused by every UI test in this module.

    pyqtgraph's ViewBox cleanup segfaults during garbage collection once
    several MainWindow instances have been created and dropped in the same
    process, so the tests share a single window and reset the state they touch
    instead of building a new one each time.
    """
    global _SHARED_WINDOW
    if _SHARED_WINDOW is None:
        import ui

        _SHARED_WINDOW = ui.MainWindow()
    return _SHARED_WINDOW


def _reset_main_window(window):
    """Return the shared window to a disconnected, idle state."""
    window.videothread = None
    window.is_running = False
    window.is_recording = False
    window.recording = np.array([])
    window.recording_metadata = {}
    window.ui.settings_window.teardown()
    for widget in (window.ui.pb_connect, window.ui.cb_source):
        widget.setEnabled(True)
    window.ui.actionSelect_GenTLProducer.setEnabled(True)


class UiRaceGuardTests(unittest.TestCase):
    """Regression test for the record-vs-live-view race fixed 2026-08-28.

    VideoThread.stop() can time out (10s) without actually stopping the
    worker: ia.stop() does not reliably unblock a fetch() already blocked
    inside the device (see VideoThread._stop_and_join's docstring). Before
    the fix, on_btn_acquire_clicked() proceeded into record_harvester_frames()
    regardless, driving a second start()/fetch()/stop() sequence on the same
    ImageAcquirer a still-running worker might still be using. It must now
    refuse instead.
    """

    @classmethod
    def setUpClass(cls):
        from PyQt6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_acquire_aborts_when_video_thread_still_running(self):
        from unittest.mock import MagicMock, patch

        import ui

        window = _shared_main_window()
        try:
            fake_thread = MagicMock()
            fake_thread.isRunning.return_value = True
            window.videothread = fake_thread

            with (
                patch.object(window, "pause_video") as mock_pause,
                patch.object(window, "acquire") as mock_acquire,
                patch("ui.QtWidgets.QMessageBox.warning") as mock_box,
            ):
                window.on_btn_acquire_clicked()

            mock_pause.assert_called_once()
            mock_acquire.assert_not_called()  # the whole point of the guard
            mock_box.assert_called_once()
        finally:
            _reset_main_window(window)

    def test_acquire_proceeds_when_video_thread_stopped(self):
        from unittest.mock import MagicMock, patch

        import ui

        window = _shared_main_window()
        try:
            fake_thread = MagicMock()
            fake_thread.isRunning.return_value = False
            window.videothread = fake_thread

            with (
                patch.object(window, "pause_video") as mock_pause,
                patch.object(window, "acquire") as mock_acquire,
                # on_btn_acquire_clicked() displays self.recording after
                # acquire() returns; mocking acquire() leaves it as the
                # placeholder np.array([]) it's reset to just before the
                # call, which update_image() (unrelated to this guard) can't
                # display. Not exercised here.
                patch.object(window, "update_image"),
            ):
                window.on_btn_acquire_clicked()

            mock_pause.assert_called_once()
            mock_acquire.assert_called_once()  # unaffected by the new guard
        finally:
            _reset_main_window(window)


# ---------------------------------------------------------------------------
# Connect / disconnect freeze + wedge regressions.
#
# Everything below pins a way the app could hang the machine when a camera was
# connected or disconnected in software (no cable replug involved).
# ---------------------------------------------------------------------------


class BoundedFetchTests(unittest.TestCase):
    """Every buffer fetch must be bounded.

    ImageAcquirer.fetch() defaults to timeout=0, which switches its timeout
    check off and spins in `while not buffer` forever. warmup_cam() runs that
    on the GUI thread during connect, so a camera that never delivered - still
    waiting for a trigger, a packet-size mismatch, a device wedged by an
    earlier crash - froze the whole application with no way to cancel. Every
    call site now goes through try_fetch(timeout=...).
    """

    def test_warmup_cam_gives_up_instead_of_blocking(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        ia.try_fetch.return_value = None
        with self.assertRaises(TimeoutError):
            cam.warmup_cam(ia, frames=3, timeout=0.01)
        ia.try_fetch.assert_called_with(timeout=0.01)
        ia.fetch.assert_not_called()  # the unbounded call must not be used

    def test_warmup_cam_queues_every_buffer_it_got(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        buffers = [MagicMock() for _ in range(3)]
        ia.try_fetch.side_effect = buffers
        cam.warmup_cam(ia, frames=3, timeout=0.01)
        for buffer in buffers:
            buffer.queue.assert_called_once()

    def test_live_fetch_falls_back_to_a_black_frame(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        ia.try_fetch.return_value = None
        ia.remote_device.node_map.Height.value = 6
        ia.remote_device.node_map.Width.value = 5
        img = cam.get_harvester_image(ia, timeout=0.05)
        self.assertEqual(img.shape, (6, 5))
        self.assertEqual(int(img.max()), 0)
        ia.fetch.assert_not_called()

    def test_recording_fetch_aborts_instead_of_blocking(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        ia.try_fetch.return_value = None
        out = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(TimeoutError):
            cam._harvester_frame_into(ia, out, timeout=0.05)
        ia.fetch.assert_not_called()

    def test_record_stops_the_camera_when_a_fetch_times_out(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        ia.try_fetch.return_value = None
        ia.remote_device.node_map.Height.value = 4
        ia.remote_device.node_map.Width.value = 4
        with self.assertRaises(TimeoutError), patch.object(cam, "FETCH_TIMEOUT_S", 0.05):
            cam.record_harvester_frames(ia, 2, None)
        # A camera left acquiring keeps filling buffers nobody drains, and the
        # next start() then runs against an already-started device.
        ia.stop.assert_called_once()

    def test_live_fetch_waits_through_incomplete_buffers(self):
        # harvesters requeues an incomplete buffer and returns None at once.
        # That is one lost frame, not a stalled camera: no black frame.
        ia = MagicMock()
        ia.try_fetch.side_effect = [None, None, _mono8_buffer(42)]
        img = cam.get_harvester_image(ia, timeout=5)
        self.assertEqual(img.tolist(), [[42, 42], [42, 42]])

    def test_recording_carries_on_after_an_incomplete_buffer(self):
        ia = MagicMock()
        ia.remote_device.node_map.Height.value = 2
        ia.remote_device.node_map.Width.value = 2
        deliveries = iter([_mono8_buffer(0), None, _mono8_buffer(1), _mono8_buffer(2)])
        ia.try_fetch.side_effect = lambda timeout: next(deliveries, None)
        recording = cam.record_harvester_frames(ia, 3, None)
        # A single dropped frame used to abort the whole recording with a
        # "no frame within 5.0s" error.
        self.assertEqual([int(frame[0, 0]) for frame in recording], [0, 1, 2])

    def test_interrupted_recording_keeps_what_was_recorded(self):
        ia = MagicMock()
        ia.remote_device.node_map.Height.value = 2
        ia.remote_device.node_map.Width.value = 2
        deliveries = iter([_mono8_buffer(7), _mono8_buffer(8)])  # then the camera stops
        ia.try_fetch.side_effect = lambda timeout: next(deliveries, None)
        with patch.object(cam, "FETCH_TIMEOUT_S", 0.05):
            with self.assertRaises(cam.RecordingInterrupted) as caught:
                cam.record_harvester_frames(ia, 5, None)
        self.assertEqual([int(frame[0, 0]) for frame in caught.exception.frames], [7, 8])
        ia.stop.assert_called_once()

    def test_oversized_recording_is_refused_before_it_fills_the_ram(self):
        # 1000 full-HD frames are ~2 GB. Under Linux overcommit the allocation
        # itself succeeds, and the machine swaps itself into a freeze while the
        # frames are written - so it has to be refused up front.
        with patch.object(cam, "available_memory_bytes", return_value=1024**3):
            with self.assertRaises(MemoryError):
                cam.record_random_frames(1000, None)

    def test_info_fields_are_only_decoded_where_the_camera_writes_them(self):
        ia = MagicMock()
        nodes = ia.remote_device.node_map
        nodes.InfoFieldFrameCounterEnable.value = True
        nodes.InfoFieldTimeStampEnable.value = True
        nodes.PixelFormat.value = "Mono8"
        self.assertTrue(cam.info_fields_enabled(ia))
        nodes.PixelFormat.value = "Mono12"  # the field bytes are bit-shifted away
        self.assertFalse(cam.info_fields_enabled(ia))
        # Cameras without the Mikrotron nodes: the first pixels are image content.
        ia.remote_device.node_map = SimpleNamespace(PixelFormat=SimpleNamespace(value="Mono8"))
        self.assertFalse(cam.info_fields_enabled(ia))


class CameraReleaseTests(unittest.TestCase):
    """A camera must never stay open because something failed on the way in."""

    def setUp(self):
        # The shared Harvester is module state; poke it directly rather than
        # calling release_harvester(), which would reset() a mock.
        cam._SHARED_HARVESTER = None
        cam._SHARED_HARVESTER_CTIS = ()

    tearDown = setUp

    def test_setup_cam_releases_the_device_when_configuration_fails(self):
        from unittest.mock import MagicMock, patch

        h, ia = MagicMock(), MagicMock()
        h.create.return_value = ia
        with (
            patch("cam.Harvester", return_value=h),
            patch("cam._configure_cam", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaises(RuntimeError):
                cam.setup_cam(cti="fake.cti")
        # Dropping the ImageAcquirer without destroy() keeps the device
        # exclusively opened, so the *next* connect blocks on a device the
        # producer still considers in use.
        ia.destroy.assert_called_once()
        # ...but the producers stay loaded for the retry: unloading them per
        # attempt is the churn the shared Harvester exists to avoid.
        h.reset.assert_not_called()

    def test_setup_cam_opens_the_requested_device(self):
        from unittest.mock import MagicMock, patch

        h = MagicMock()
        key = {"display_name": "Cam B"}
        with (
            patch("cam.Harvester", return_value=h),
            patch("cam._configure_cam"),
            patch("cam.warmup_cam"),
        ):
            cam.setup_cam(cti=["a.cti", "b.cti", "a.cti"], search_key=key)
        # Was hardcoded to create(0), which silently opened the wrong camera.
        h.create.assert_called_once_with(key)
        # Duplicates dropped: add_file() rejects an already-loaded producer.
        self.assertEqual(
            [call.args[0] for call in h.add_file.call_args_list], ["a.cti", "b.cti"]
        )

    def test_harvester_is_shared_across_connect_cycles(self):
        from unittest.mock import MagicMock, patch

        created = []

        def make_harvester():
            harvester = MagicMock()
            harvester.device_info_list = []
            created.append(harvester)
            return harvester

        with (
            patch("cam.Harvester", side_effect=make_harvester),
            patch("cam._configure_cam"),
            patch("cam.warmup_cam"),
        ):
            h1, ia1 = cam.setup_cam(cti="a.cti")
            cam.teardown_cam(None, ia1)  # what a disconnect does now
            h2, _ = cam.setup_cam(cti="a.cti")
        self.assertIs(h1, h2)
        # One producer load for both connects. Building a Harvester per connect
        # unloaded and reloaded the GenTL producer every cycle.
        self.assertEqual(len(created), 1)
        h1.reset.assert_not_called()
        ia1.destroy.assert_called_once()

    def test_release_harvester_resets_and_forgets_it(self):
        from unittest.mock import MagicMock, patch

        first, second = MagicMock(), MagicMock()
        with patch("cam.Harvester", side_effect=[first, second]):
            self.assertIs(cam.get_harvester(["a.cti"]), first)
            self.assertIs(cam.get_harvester(["a.cti"]), first)  # reused
            cam.release_harvester()
            first.reset.assert_called_once()
            # Rebuilt afterwards, not handed out already-reset.
            self.assertIs(cam.get_harvester(["a.cti"]), second)

    def test_changing_the_producer_set_rebuilds_the_harvester(self):
        from unittest.mock import MagicMock, patch

        first, second = MagicMock(), MagicMock()
        with patch("cam.Harvester", side_effect=[first, second]):
            self.assertIs(cam.get_harvester(["a.cti"]), first)
            # add_file() cannot be undone selectively, so a new .cti means a
            # new Harvester - and the old one must be released.
            self.assertIs(cam.get_harvester(["a.cti", "b.cti"]), second)
        first.reset.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in second.add_file.call_args_list],
            ["a.cti", "b.cti"],
        )

    def test_connect_skips_enumeration_for_an_already_known_device(self):
        from unittest.mock import MagicMock, patch

        device = MagicMock()
        device.display_name = "Cam A"
        h = MagicMock()
        h.device_info_list = [device]
        with (
            patch("cam.Harvester", return_value=h),
            patch("cam._configure_cam"),
            patch("cam.warmup_cam"),
        ):
            cam.setup_cam(cti="a.cti", search_key={"display_name": "Cam A"})
        # h.update() is a full re-enumeration, discovery traffic on every
        # interface. It used to run on every connect even though the startup
        # scan had just enumerated the very same device.
        h.update.assert_not_called()

    def test_connect_enumerates_for_an_unknown_device(self):
        from unittest.mock import MagicMock, patch

        h = MagicMock()
        h.device_info_list = []
        with (
            patch("cam.Harvester", return_value=h),
            patch("cam._configure_cam"),
            patch("cam.warmup_cam"),
        ):
            cam.setup_cam(cti="a.cti", search_key={"display_name": "Cam A"})
        h.update.assert_called_once()

    def test_teardown_cam_continues_after_a_failing_step(self):
        from unittest.mock import MagicMock

        h, ia = MagicMock(), MagicMock()
        ia.stop.side_effect = RuntimeError("not acquiring")
        ia.destroy.side_effect = RuntimeError("nope")
        cam.teardown_cam(h, ia)
        ia.destroy.assert_called_once()
        h.reset.assert_called_once()  # must not be skipped by the failures above

    def test_camera_index_probe_releases_every_handle(self):
        from unittest.mock import MagicMock, patch

        probes = []

        def fake_capture(index):
            probe = MagicMock()
            probe.read.return_value = (index == 0, None)
            probes.append(probe)
            return probe

        with patch("cam.VideoCapture", side_effect=fake_capture):
            self.assertEqual(cam.get_camera_indices(), [0])
        self.assertEqual(len(probes), 10)
        for probe in probes:
            # release() used to sit inside the `if`, leaking the nine handles
            # of the indices that did not answer - on every rescan.
            probe.release.assert_called_once()

    def test_connect_rescans_once_when_the_cached_device_cannot_be_opened(self):
        device = MagicMock()
        device.display_name = "Cam A"
        h, ia = MagicMock(), MagicMock()
        h.device_info_list = [device]
        h.create.side_effect = [RuntimeError("stale device handle"), ia]
        with (
            patch("cam.Harvester", return_value=h),
            patch("cam._configure_cam"),
            patch("cam.warmup_cam"),
        ):
            _, opened = cam.setup_cam(cti="a.cti", search_key={"display_name": "Cam A"})
        # A camera power-cycled since the startup scan failed every Connect
        # from the cached list until something re-enumerated.
        self.assertIs(opened, ia)
        h.update.assert_called_once()

    def test_connect_opens_a_camera_two_producers_both_list(self):
        first, second = MagicMock(), MagicMock()
        first.display_name = second.display_name = "Cam A"
        h = MagicMock()
        h.device_info_list = [first, second]
        with (
            patch("cam.Harvester", return_value=h),
            patch("cam._configure_cam"),
            patch("cam.warmup_cam"),
        ):
            cam.setup_cam(cti=["vimba.cti", "vimbax.cti"], search_key={"display_name": "Cam A"})
        # harvesters' create(dict) refuses with "multiple devices found".
        h.create.assert_called_once_with(first)

    def test_saved_offsets_survive_a_camera_that_starts_at_full_size(self):
        width_max, height_max = 2336, 1728
        state = {"Width": width_max, "Height": height_max, "OffsetX": 0, "OffsetY": 0}

        class CoupledNode:
            """SFNC AOI node: an offset and its size share the sensor."""

            inc = 1

            def __init__(self, name):
                self.name = name

            @property
            def min(self):
                return 0 if self.name.startswith("Offset") else 16

            @property
            def max(self):
                return {
                    "Width": width_max - state["OffsetX"],
                    "Height": height_max - state["OffsetY"],
                    "OffsetX": width_max - state["Width"],
                    "OffsetY": height_max - state["Height"],
                }[self.name]

            @property
            def value(self):
                return state[self.name]

            @value.setter
            def value(self, value):
                if not self.min <= value <= self.max:
                    raise ValueError(f"{self.name}={value} out of range")
                state[self.name] = value

        ia = MagicMock()
        ia.is_acquiring.return_value = False
        ia.remote_device.node_map = SimpleNamespace(**{name: CoupledNode(name) for name in state})
        applied = cam.set_resolution(ia, width=1024, height=1024, offset_x=656, offset_y=352)
        # Offsets written first were clamped to 0 by the full-size AOI.
        self.assertEqual(applied, (1024, 1024, 656, 352))


class VideoThreadTeardownTests(unittest.TestCase):
    """Nothing may be freed out from under a worker that is still inside it."""

    @classmethod
    def setUpClass(cls):
        from PyQt6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    @staticmethod
    def _bare_thread():
        """A VideoThread without the camera setup its __init__ would do."""
        from PyQt6 import QtCore

        from VideoThread import VideoThread

        class _Bare(VideoThread):
            def __init__(self):
                QtCore.QThread.__init__(self)
                self.source = "test"
                self._run_flag = False
                self.cap = None
                self.harvester = None
                self.image_func = lambda cap: None

        return _Bare()

    def test_close_keeps_a_webcam_handle_while_the_worker_runs(self):
        from unittest.mock import patch

        import VideoThread as videothread_module

        class FakeCapture:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        thread = self._bare_thread()
        cap = FakeCapture()
        thread.cap = cap
        with (
            patch.object(thread, "_stop_and_join"),
            patch.object(thread, "isRunning", return_value=True),
            patch.object(videothread_module.cv2, "VideoCapture", FakeCapture),
        ):
            thread.close()
        # release() under an in-flight read() is a use-after-free in OpenCV /
        # V4L2. The isRunning() guard used to cover only the harvester branch.
        self.assertFalse(cap.released)
        self.assertIs(thread.cap, cap)  # leaked on purpose, not freed
        thread.cap = None

    def test_close_releases_a_webcam_once_the_worker_exited(self):
        from unittest.mock import patch

        import VideoThread as videothread_module

        class FakeCapture:
            def __init__(self):
                self.released = False

            def release(self):
                self.released = True

        thread = self._bare_thread()
        cap = FakeCapture()
        thread.cap = cap
        with (
            patch.object(thread, "_stop_and_join"),
            patch.object(videothread_module.cv2, "VideoCapture", FakeCapture),
        ):
            thread.close()
        self.assertTrue(cap.released)
        self.assertIsNone(thread.cap)

    def test_close_tears_the_harvester_camera_down_exactly_once(self):
        from unittest.mock import patch

        thread = self._bare_thread()
        thread.cap = object()  # neither None nor a VideoCapture
        thread.harvester = object()
        with patch.object(thread, "_stop_and_join"), patch("cam.teardown_cam") as td:
            thread.close()
            thread.close()  # idempotent: a second disconnect must be harmless
        td.assert_called_once()
        self.assertIsNone(thread.cap)
        self.assertIsNone(thread.harvester)

    def test_close_leaks_the_camera_while_the_worker_is_stuck(self):
        from unittest.mock import patch

        thread = self._bare_thread()
        thread.cap = object()
        with (
            patch.object(thread, "_stop_and_join"),
            patch.object(thread, "isRunning", return_value=True),
            patch("cam.teardown_cam") as td,
        ):
            thread.close()
        # h.reset() unloads the GenTL producer; doing that under an in-flight
        # fetch() can crash the process or wedge the camera.
        td.assert_not_called()
        thread.cap = None

    def test_start_refuses_over_a_worker_that_never_stopped(self):
        from unittest.mock import patch

        thread = self._bare_thread()
        with patch.object(thread, "isRunning", return_value=True):
            self.assertFalse(thread._start())
        # A second start() on the device the stuck worker is still fetching
        # from is two concurrent state changes on one GenTL device.
        self.assertFalse(thread._run_flag)

    def test_start_and_stop_round_trip(self):
        thread = self._bare_thread()
        self.assertTrue(thread._start())
        try:
            self.assertTrue(thread._run_flag)
        finally:
            thread.stop()
        self.assertFalse(thread.isRunning())

    def test_run_keeps_pacing_when_the_source_keeps_failing(self):
        import threading
        import time

        thread = self._bare_thread()
        calls = []

        def boom(cap):
            calls.append(time.monotonic())
            raise RuntimeError("camera gone")

        thread.image_func = boom
        thread._run_flag = True
        worker = threading.Thread(target=thread.run)
        worker.start()
        try:
            time.sleep(0.3)
        finally:
            thread._run_flag = False
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        # ~9 iterations at 30 Hz. With the sleep inside the try (the bug), an
        # erroring source spun thousands of times: it pegged a core and cycled
        # IA_LOCK fast enough to starve the GUI thread of it.
        self.assertGreater(len(calls), 1)
        self.assertLess(len(calls), 30)

    def _harvester_thread(self, ia):
        thread = self._bare_thread()
        thread.cap = ia
        thread.image_func = thread._harvester_frame
        return thread

    def test_stop_waits_for_the_worker_to_hand_its_buffer_back(self):
        """harvesters' stop() revokes every announced buffer. A buffer the
        worker was still holding crashed the producer when it was queued
        afterwards (SIGSEGV in queue_buffer)."""
        from harvesters.core import ImageAcquirer

        events = []
        holding, stopped = threading.Event(), threading.Event()

        def hold_then_queue():
            holding.set()
            stopped.wait(0.3)  # a slow frame conversion: still holding it
            events.append("queue")

        ia = MagicMock(spec=ImageAcquirer)
        ia.try_fetch.side_effect = lambda timeout: _mono8_buffer(1, on_exit=hold_then_queue)
        ia.stop.side_effect = lambda: (events.append("stop"), stopped.set())
        thread = self._harvester_thread(ia)
        self.assertTrue(thread._start())
        try:
            self.assertTrue(holding.wait(5))
        finally:
            thread.stop()
        self.assertFalse(thread.isRunning())
        self.assertEqual(events[-1], "stop")  # nothing queued after the stop

    def test_stop_is_prompt_while_the_camera_delivers_nothing(self):
        from harvesters.core import ImageAcquirer

        ia = MagicMock(spec=ImageAcquirer)
        ia.try_fetch.side_effect = _stalled_fetch
        thread = self._harvester_thread(ia)
        self.assertTrue(thread._start())
        time.sleep(0.3)
        started = time.monotonic()
        thread.stop()
        # Pause/Disconnect sat out the whole 5 s fetch timeout on the GUI thread.
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(thread.isRunning())

    def test_camera_writes_are_not_held_up_by_a_stalled_camera(self):
        from harvesters.core import ImageAcquirer

        ia = MagicMock(spec=ImageAcquirer)
        ia.try_fetch.side_effect = _stalled_fetch
        thread = self._harvester_thread(ia)
        self.assertTrue(thread._start())
        try:
            time.sleep(0.2)
            started = time.monotonic()
            cam.set_attribute(ia, "Gain", 3)  # one step of a GUI control
            elapsed = time.monotonic() - started
        finally:
            thread.stop()
        # The worker held IA_LOCK across its whole 5 s wait for a frame.
        self.assertLess(elapsed, 1.0)

    def test_stop_on_the_worker_itself_returns_at_once(self):
        # What __del__ -> close() does when the last reference is dropped on
        # the worker thread: QThread.wait() on itself returns immediately, so
        # the join loop spun (and flooded stderr) for the full 10 s.
        thread = self._bare_thread()
        durations = []

        def stop_from_inside(cap):
            started = time.monotonic()
            thread._stop_and_join()
            durations.append(time.monotonic() - started)

        thread.image_func = stop_from_inside
        self.assertTrue(thread._start())
        self.assertTrue(thread.wait(5000))
        self.assertEqual(len(durations), 1)
        self.assertLess(durations[0], 1.0)

    def test_a_stuck_worker_keeps_its_camera_until_it_exits(self):
        import VideoThread as videothread_module

        inside, release = threading.Event(), threading.Event()

        def hanging_driver_call(cap):
            inside.set()
            release.wait(10)

        thread = self._bare_thread()
        thread.cap = object()
        thread.image_func = hanging_driver_call
        thread.JOIN_TIMEOUT_S = 0.2
        self.assertTrue(thread._start())
        self.assertTrue(inside.wait(5))
        with patch("cam.teardown_cam") as teardown:
            thread.close()
            # Releasing (or resetting the producer) under the hung call is a
            # use-after-free; and the UI must know the camera is still taken.
            teardown.assert_not_called()
            self.assertTrue(videothread_module.has_stuck_workers())
            release.set()
            deadline = time.monotonic() + 5
            while not teardown.called and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.01)
            # Released once the worker is really gone.
            teardown.assert_called_once()
        self.assertFalse(videothread_module.has_stuck_workers())


class SettingsPanelTests(unittest.TestCase):
    """The 500 ms node-map poll must not outlive or multiply across cameras."""

    @classmethod
    def setUpClass(cls):
        from PyQt6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_refilling_does_not_stack_pollers_or_rows(self):
        from unittest.mock import MagicMock, patch

        import ui

        window = _shared_main_window()
        try:
            panel = window.ui.settings_window
            window.videothread = MagicMock()
            with patch("ui.cam.get_harvester_nodemap", return_value=MagicMock()):
                panel.fill_settings()
                rows_after_one = panel.ui.formLayout.rowCount()
                panel.fill_settings()
            # Every connect used to append another full set of rows and connect
            # update_fields to the same timer again, so after N cycles one tick
            # ran N complete node-map polls and the GUI never went idle again.
            self.assertEqual(panel.ui.formLayout.rowCount(), rows_after_one)
            # And connecting must not start a poll while the panel is hidden.
            self.assertFalse(panel.timer.isActive())
            self.assertEqual(panel.timer.receivers(panel.timer.timeout), 0)

            panel.teardown()
            self.assertFalse(panel.timer.isActive())
            self.assertEqual(panel.settings, {})
            self.assertEqual(panel.ui.formLayout.rowCount(), 0)
        finally:
            _reset_main_window(window)

    def test_poll_runs_only_while_the_panel_is_on_screen(self):
        from unittest.mock import MagicMock, patch

        window = _shared_main_window()
        panel = window.ui.settings_window
        try:
            window.videothread = MagicMock()
            # fill_settings finds no usable nodes behind a mock node map, so
            # stand one in - start_polling has nothing to poll otherwise.
            panel.settings = {"SomeNode": {"widget": MagicMock()}}
            with patch("ui.cam.get_harvester_nodemap", return_value=MagicMock()):
                panel.show()
                self.assertTrue(panel.timer.isActive())
                self.assertEqual(panel.timer.receivers(panel.timer.timeout), 1)
                panel.hide()
                self.assertFalse(panel.timer.isActive())
                # Reopening must not stack a second connection.
                panel.show()
                panel.hide()
                self.assertEqual(panel.timer.receivers(panel.timer.timeout), 1)
        finally:
            panel.hide()
            _reset_main_window(window)

    def test_update_fields_never_waits_for_the_device_lock(self):
        import threading
        import time
        from unittest.mock import MagicMock, patch

        import ui

        window = _shared_main_window()
        holding = threading.Event()
        release = threading.Event()

        def hold_the_device():
            with cam.IA_LOCK:
                holding.set()
                release.wait(10)

        holder = threading.Thread(target=hold_the_device)
        holder.start()
        try:
            self.assertTrue(holding.wait(5))
            panel = window.ui.settings_window
            window.videothread = MagicMock()
            widget = MagicMock()
            panel.settings = {"SomeNode": {"widget": widget}}

            started = time.monotonic()
            with patch("ui.cam.get_harvester_nodemap", return_value=MagicMock()):
                panel.update_fields()
            elapsed = time.monotonic() - started
        finally:
            release.set()
            holder.join(timeout=10)
            _reset_main_window(window)
        # The worker holds IA_LOCK for a whole fetch; blocking on it here froze
        # the GUI thread for as long as the camera took - forever if it hung.
        self.assertLess(elapsed, 1.0)
        widget.blockSignals.assert_not_called()  # poll skipped, not performed

    def _fill_with(self, panel, nodemap, **node_types):
        """fill_settings() over a fake node map; node_types patches ui.IString etc."""
        patches = [patch(f"ui.{name}", fake) for name, fake in node_types.items()]
        patches.append(patch("ui.cam.get_harvester_nodemap", return_value=nodemap))
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        panel.fill_settings()

    def test_string_edits_go_to_their_own_node(self):
        class FakeString:
            value = ""

        window = _shared_main_window()
        panel = window.ui.settings_window
        try:
            window.videothread = MagicMock()
            nodemap = SimpleNamespace(DeviceLabel=FakeString(), DeviceUserID=FakeString())
            self._fill_with(panel, nodemap, IString=FakeString)
            with patch.object(panel, "update_node") as update_node:
                widget = panel.settings["DeviceLabel"]["widget"]
                widget.setText("rig A")
                widget.editingFinished.emit()
            # The lambda used to read the loop variables when the edit finished:
            # the last node of the loop, with the last widget's text.
            update_node.assert_called_once_with("DeviceLabel", "rig A")
        finally:
            _reset_main_window(window)

    def test_float_rows_show_large_values_and_ignore_the_wheel(self):
        from PyQt6 import QtCore, QtGui, QtWidgets

        class FakeFloat:
            value = 999.5
            min = 0.0
            max = 1e6

            def get_access_mode(self):
                return 4  # RW

        class FakeEnum:
            value = "Off"

            def _get_symbolics(self):
                return ("Off", "On")

            def get_access_mode(self):
                return 4

        window = _shared_main_window()
        panel = window.ui.settings_window
        try:
            window.videothread = MagicMock()
            nodemap = SimpleNamespace(ExposureTime=FakeFloat(), TriggerMode=FakeEnum())
            self._fill_with(panel, nodemap, IFloat=FakeFloat, IEnumeration=FakeEnum)
            panel.update_fields()
            exposure = panel.settings["ExposureTime"]["widget"]
            trigger = panel.settings["TriggerMode"]["widget"]
            # Qt's default 0..99.99 range showed 999.5 as 99.99.
            self.assertEqual(exposure.value(), 999.5)
            with patch.object(panel, "update_node") as update_node:
                for widget in (exposure, trigger):
                    notch = QtGui.QWheelEvent(
                        QtCore.QPointF(5, 5),
                        QtCore.QPointF(5, 5),
                        QtCore.QPoint(0, 0),
                        QtCore.QPoint(0, -120),
                        QtCore.Qt.MouseButton.NoButton,
                        QtCore.Qt.KeyboardModifier.NoModifier,
                        QtCore.Qt.ScrollPhase.NoScrollPhase,
                        False,
                    )
                    QtWidgets.QApplication.sendEvent(widget, notch)
            # Scrolling the panel over a row wrote the camera (TriggerMode=On:
            # no frame ever arrived afterwards).
            update_node.assert_not_called()
            self.assertEqual(trigger.currentText(), "Off")
        finally:
            _reset_main_window(window)

    def test_acquisition_and_reset_commands_are_not_offered(self):
        class FakeCommand:
            pass

        window = _shared_main_window()
        panel = window.ui.settings_window
        try:
            window.videothread = MagicMock()
            nodemap = SimpleNamespace(
                AcquisitionStart=FakeCommand(),
                AcquisitionStop=FakeCommand(),
                DeviceReset=FakeCommand(),
                TriggerSoftware=FakeCommand(),
            )
            self._fill_with(panel, nodemap, ICommand=FakeCommand)
            # Run behind harvesters' back they leave it believing the camera is
            # stopped while it streams (or reset it under open handles).
            self.assertEqual(list(panel.settings), ["TriggerSoftware"])
        finally:
            _reset_main_window(window)

    def test_unsafe_cxp_link_nodes_are_never_exposed(self):
        """ConnectionConfig/CxpLinkConfiguration must never reach the panel.

        Writing either makes the frame grabber retrain the physical CoaXPress
        link. On the rig that produced a fatal PCIe fault on the grabber's
        root port - a whole-machine hard lockup, not a Python exception -
        captured in harvesters_pg.log at 2026-09-21 15:13:48 ("Setting
        ConnectionConfig to CONNECTION4SPEED12500", nothing logged after).
        fill_settings() must skip them before reading them at all.
        """
        accessed = []
        listed = ["ConnectionConfig", "CxpLinkConfiguration", "ExposureTime"]

        class FakeNodeMap:
            ConnectionConfig = "unsafe"
            CxpLinkConfiguration = "unsafe"
            ExposureTime = "safe"

            def __dir__(self):
                return list(listed)

            def __getattribute__(self, name):
                if name in listed:
                    accessed.append(name)
                return object.__getattribute__(self, name)

        window = _shared_main_window()
        panel = window.ui.settings_window
        try:
            window.videothread = MagicMock()
            self._fill_with(panel, FakeNodeMap())
            self.assertNotIn("ConnectionConfig", panel.settings)
            self.assertNotIn("CxpLinkConfiguration", panel.settings)
            # Not merely unexposed: never read, so the panel issues no GenCP
            # transaction at all for them.
            self.assertNotIn("ConnectionConfig", accessed)
            self.assertNotIn("CxpLinkConfiguration", accessed)
            # A normal node on the same node map is still read.
            self.assertIn("ExposureTime", accessed)
        finally:
            _reset_main_window(window)

    def test_connect_defers_reading_the_node_map_until_the_panel_opens(self):
        window = _shared_main_window()
        panel = window.ui.settings_window
        window.ui.cb_source.addItem("Cam X")
        try:
            window.ui.cb_source.setCurrentText("Cam X")
            window.videothread = MagicMock()
            with (
                patch.object(window, "setup_settings"),
                patch.object(window, "setup_resolution_settings"),
                patch("ui.cam.get_harvester_nodemap", return_value=SimpleNamespace()) as nodemap,
            ):
                window.on_source_connected()
                # Every connect walked the whole node map on the GUI thread
                # for a panel that is usually never opened.
                nodemap.assert_not_called()
                panel.show()
                nodemap.assert_called_once()
        finally:
            panel.hide()
            window.ui.cb_source.removeItem(window.ui.cb_source.findText("Cam X"))
            _reset_main_window(window)


class ConnectDisconnectTests(unittest.TestCase):
    """MainWindow's connect/disconnect path."""

    @classmethod
    def setUpClass(cls):
        from PyQt6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_disconnect_releases_the_camera_explicitly(self):
        from unittest.mock import MagicMock

        import ui

        window = _shared_main_window()
        try:
            worker = MagicMock()
            worker.isRunning.return_value = False  # it stopped, as close() ensures
            window.videothread = worker
            window.disconnect_video()
            # Teardown used to wait for __del__, so the device stayed open for
            # an unpredictable stretch and a quick reconnect wedged.
            worker.close.assert_called_once()
            self.assertIsNone(window.videothread)
            self.assertFalse(window.ui.settings_window.timer.isActive())
        finally:
            _reset_main_window(window)

    def test_connect_reports_a_failure_instead_of_aborting(self):
        from unittest.mock import patch

        import ui

        window = _shared_main_window()
        try:
            with (
                patch("ui.VideoThread", side_effect=RuntimeError("no camera")),
                patch("ui.QtWidgets.QMessageBox.critical") as box,
            ):
                window.connect_video()
            # An exception escaping a Qt slot reaches qFatal and kills the
            # process, possibly with the camera half-open.
            box.assert_called_once()
            self.assertIsNone(window.videothread)
            self.assertTrue(window.ui.cb_source.isEnabled())
        finally:
            _reset_main_window(window)

    def test_rescan_refuses_while_a_camera_is_connected(self):
        from unittest.mock import MagicMock, patch

        import ui

        window = _shared_main_window()
        try:
            window.videothread = MagicMock()
            with (
                patch("ui.cam.return_harvester_cameras") as enumerate_cameras,
                patch("ui.cam.get_camera_indices") as probe_webcams,
                patch("ui.QtWidgets.QMessageBox.information") as box,
            ):
                window.populate_video_sources()
            # Enumerating builds a second Harvester over the same producer and
            # resets it when done, unloading the library the open device uses.
            enumerate_cameras.assert_not_called()
            probe_webcams.assert_not_called()
            box.assert_called_once()
        finally:
            _reset_main_window(window)

    def test_rescan_does_not_duplicate_sources_or_producers(self):
        from unittest.mock import patch

        import ui

        window = _shared_main_window()
        original_ctis = list(window.ctis)
        try:
            with (
                patch("ui.cam.get_camera_indices", return_value=[0]),
                patch("ui.cam.load_cti_envvar", return_value=["/x/a.cti"]),
                patch(
                    "ui.cam.return_harvester_cameras",
                    return_value=[{"display_name": "Cam A", "model": "A"}],
                ),
            ):
                window.ctis = []
                window.populate_video_sources()
                first = [
                    window.ui.cb_source.itemText(i)
                    for i in range(window.ui.cb_source.count())
                ]
                window.populate_video_sources()
                second = [
                    window.ui.cb_source.itemText(i)
                    for i in range(window.ui.cb_source.count())
                ]
            self.assertEqual(first, ["Webcam 0", "Cam A", "random"])
            self.assertEqual(second, first)
            # add_file() rejects a producer that is already loaded, so a
            # duplicated cti list broke enumeration outright.
            self.assertEqual(window.ctis, ["/x/a.cti"])
        finally:
            window.ctis = original_ctis
            _reset_main_window(window)

    def test_start_refuses_when_the_worker_did_not_stop(self):
        from unittest.mock import MagicMock, patch

        import ui

        window = _shared_main_window()
        try:
            worker = MagicMock()
            worker._start.return_value = False
            window.videothread = worker
            with patch("ui.QtWidgets.QMessageBox.warning") as box:
                window.start_video()
            box.assert_called_once()
            self.assertFalse(window.is_running)
        finally:
            _reset_main_window(window)

    def test_clicks_queued_during_a_blocking_connect_are_dropped(self):
        from PyQt6 import QtCore, QtGui, QtWidgets

        window = _shared_main_window()
        button = window.ui.pb_connect

        def blocking_connect():
            # The user clicks the button again while the GUI thread is busy;
            # by the time the click is delivered the button means Disconnect.
            center = QtCore.QPointF(button.rect().center())
            for kind in (
                QtCore.QEvent.Type.MouseButtonPress,
                QtCore.QEvent.Type.MouseButtonRelease,
            ):
                QtWidgets.QApplication.postEvent(
                    button,
                    QtGui.QMouseEvent(
                        kind,
                        center,
                        center,
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.MouseButton.LeftButton,
                        QtCore.Qt.KeyboardModifier.NoModifier,
                    ),
                )
            window.videothread = MagicMock()
            button.clicked.disconnect()
            button.clicked.connect(window.disconnect_video)

        try:
            with (
                patch.object(window, "_connect_video", side_effect=blocking_connect),
                patch.object(window, "disconnect_video") as disconnect,
            ):
                window.connect_video()
                self.app.processEvents()
            disconnect.assert_not_called()
        finally:
            button.clicked.disconnect()
            button.clicked.connect(window.connect_video)
            _reset_main_window(window)

    def test_connect_stays_locked_until_a_stuck_worker_lets_go(self):
        window = _shared_main_window()
        try:
            worker = MagicMock()
            worker.isRunning.return_value = True  # close() had to leave it running
            window.videothread = worker
            window.disconnect_video()
            # Connecting or rescanning now would re-enumerate or reset the
            # producer under the stuck call, or block on the device it holds.
            self.assertFalse(window.ui.pb_connect.isEnabled())
            self.assertFalse(window.ui.actionSelect_GenTLProducer.isEnabled())
            worker.finished.connect.call_args.args[0]()  # the worker exits
            self.assertTrue(window.ui.pb_connect.isEnabled())
            self.assertTrue(window.ui.actionSelect_GenTLProducer.isEnabled())
        finally:
            _reset_main_window(window)


class RecordingAndSavingTests(unittest.TestCase):
    """A recording must survive the UI around it until it is saved."""

    @classmethod
    def setUpClass(cls):
        from PyQt6 import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.window = _shared_main_window()
        self.tmp = Path(tempfile.mkdtemp(prefix="hpg_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        _reset_main_window(self.window)

    def _save_as(self, path, file_filter="HDF5 Files (*.h5)"):
        with patch(
            "ui.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(path), file_filter)
        ):
            self.window.save()

    def test_recording_can_be_saved_after_disconnecting(self):
        window = self.window
        window.videothread = None  # recorded, then disconnected
        window.recording = np.zeros((2, 4, 4), np.uint8)
        window.recording_id = "rec-1"
        window.recording_timestamp = "20260101120000"
        window.recording_metadata = {"fps": 123, "ExposureTime": 5.0, "timestamps": [0.0, 0.5]}
        out = self.tmp / "rec.h5"
        self._save_as(out)
        frames, metadata = importexport.load_frames_from_h5(str(out))
        # save() read the camera at save time: AttributeError, nothing written.
        self.assertEqual(frames.shape, (2, 4, 4))
        self.assertEqual(metadata["fps"], 123)
        self.assertEqual(metadata["ExposureTime"], 5.0)
        self.assertEqual(list(metadata["timestamps"]), [0.0, 0.5])
        self.assertEqual(metadata["recording_id"], "rec-1")
        # Saving again (e.g. as a second copy) still has the datasets.
        self._save_as(self.tmp / "copy.h5")
        _, again = importexport.load_frames_from_h5(str(self.tmp / "copy.h5"))
        self.assertEqual(list(again["timestamps"]), [0.0, 0.5])

    def test_reopened_recording_is_saved_with_its_own_metadata(self):
        source = self.tmp / "b.h5"
        importexport.save_frames_to_h5(
            np.ones((3, 4, 4), np.uint8),
            str(source),
            metadata={
                "recording_id": "B",
                "timestamp": "20260102120000",
                "fps": 500,
                "frame_counters": [1, 2, 3],
            },
        )
        with patch(
            "ui.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(source), "")
        ):
            self.window.open_file()
        self._save_as(self.tmp / "c.h5")
        _, metadata = importexport.load_frames_from_h5(str(self.tmp / "c.h5"))
        self.assertEqual(metadata["recording_id"], "B")
        self.assertEqual(metadata["timestamp"], "20260102120000")
        self.assertEqual(metadata["fps"], 500)
        self.assertEqual(list(metadata["frame_counters"]), [1, 2, 3])

    def test_cancelled_dialogs_change_nothing(self):
        window = self.window
        worker = MagicMock()
        window.videothread = worker
        window.is_running = True
        with patch("ui.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")):
            window.open_file()  # used to pause the live view, then raise AxisError
        worker.stop.assert_not_called()
        window.recording = np.zeros((1, 2, 2), np.uint8)
        with patch("ui.importexport.save_frames_to_h5") as write:
            self._save_as("", "")
        write.assert_not_called()
        window.videothread = None
        with (
            patch("ui.QtWidgets.QFileDialog.getOpenFileName", return_value=("", "")),
            patch.object(window, "populate_video_sources") as rescan,
        ):
            window.find_cti_dialog()
        # A rescan reloads every producer; not for a cancelled dialog.
        rescan.assert_not_called()

    def test_close_is_refused_while_recording(self):
        from PyQt6 import QtGui

        window = self.window
        window.is_recording = True
        event = QtGui.QCloseEvent()
        with (
            patch.object(window, "disconnect_video") as disconnect,
            patch("ui.cam.release_harvester") as release,
        ):
            window.closeEvent(event)
        # The recording loop pumps events; a close from inside it destroyed
        # the camera under the loop and discarded the recording.
        self.assertFalse(event.isAccepted())
        disconnect.assert_not_called()
        release.assert_not_called()

    def test_recording_locks_out_everything_that_reenters_the_camera(self):
        window = self.window
        window.videothread = MagicMock()
        window.videothread.isRunning.return_value = False
        window.ui.actionOpen.setEnabled(True)
        controls = {
            "File>Open": window.ui.actionOpen,
            "File>Exit": window.ui.actionExit,
            "Settings": window.ui.pb_settings,
            "camera controls": window.ui.settings,
            "Save frame": window.ui.pb_save_frame,
            "Connect": window.ui.pb_connect,
        }
        seen = {}

        def record(n_frames, progressbar):
            seen.update({name: widget.isEnabled() for name, widget in controls.items()})

        with (
            patch.object(window, "pause_video"),
            patch.object(window, "acquire", side_effect=record),
        ):
            window.on_btn_acquire_clicked()
        self.assertEqual({name: False for name in controls}, seen)
        self.assertTrue(window.ui.actionOpen.isEnabled())  # restored afterwards

    def test_frame_export_keeps_the_chosen_format(self):
        window = self.window
        window.Image.setImage(np.zeros((8, 6), np.uint8))
        out = self.tmp / "frame.png"
        with patch(
            "ui.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(out), "Image (*.png)"),
        ):
            window.save_current_frame()
        # 'frame.png' used to be written as a JPEG named 'frame.png.jpg'.
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["frame.png"])

    def test_image_export_replaces_a_longer_earlier_export(self):
        target = self.tmp / "rec.png"
        importexport.save_frames_to_png(np.zeros((5, 4, 4), np.uint8), str(target))
        importexport.save_frames_to_png(np.ones((2, 4, 4), np.uint8), str(target))
        # The folder used to mix the new frames with the old export's tail.
        self.assertEqual(
            sorted(p.name for p in (self.tmp / "rec").iterdir()), ["rec_0.png", "rec_1.png"]
        )


if __name__ == "__main__":
    unittest.main()
