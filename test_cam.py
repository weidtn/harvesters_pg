#!/usr/bin/env python3
"""Contract tests for the recording-speed + node-guard fixes.

Frames are stored and displayed in the camera's natural (height, width)
orientation throughout the app (single exceptions: none). These tests pin the
new observable contracts:
  * info-field extraction reads the top-left pixels of the (h, w) frame,
  * h5 save/load round-trips preserve (n, h, w) without any transpose,
  * PNG export writes the upright (h, w) frame,
  * setup_cam tolerates cameras without AcquisitionFrameRate/Max or other
    optional GenICam nodes (e.g. the bundled TLSimu producer),
  * record_harvester_frames yields an owned, C-contiguous (n, h, w) stack.

Run:  .venv/bin/python -m unittest test_cam -v
"""

import unittest
from pathlib import Path

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

    def test_resolution_config_roundtrip(self):
        original = cam.RESOLUTION_CONFIG_PATH
        cam.RESOLUTION_CONFIG_PATH = Path(__file__).parent / "test_res_config.json"
        try:
            cam.save_resolution_config(
                {"width": 160, "height": 120, "offset_x": None, "offset_y": 4}
            )
            self.assertEqual(
                cam.load_resolution_config(),
                {"width": 160, "height": 120, "offset_y": 4},
            )
        finally:
            cam.RESOLUTION_CONFIG_PATH.unlink(missing_ok=True)
            cam.RESOLUTION_CONFIG_PATH = original


class _FakeComponent:
    """Minimal stand-in for harvesters' Component2DImage (read-only attrs)."""

    def __init__(self, fmt, data, width, height):
        self.data_format = fmt
        self.data = data
        self.width = width
        self.height = height


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
        mosaic = (np.arange(8 * 8) * 7) % 256
        mosaic = mosaic.reshape(8, 8).astype(np.uint8)
        for pattern, code in (
            ("BayerGR8", "GR"),
            ("BayerRG8", "RG"),
            ("BayerGB8", "GB"),
            ("BayerBG8", "BG"),
        ):
            import cv2

            expected = cv2.cvtColor(mosaic, getattr(cv2, f"COLOR_Bayer{code}2GRAY"))
            self.assertEqual(
                cam.grayscale8_from_bytes(pattern, mosaic, 8, 8).tolist(),
                expected.tolist(),
                pattern,
            )

    def test_bayer10_16bit_demosaic(self):
        import cv2

        mosaic16 = (np.arange(8 * 8).reshape(8, 8) * 512).astype(np.uint16)
        mosaic8 = (mosaic16 >> 2).astype(np.uint8)
        self.assertEqual(
            cam.grayscale8_from_bytes("BayerRG10", mosaic16, 8, 8).tolist(),
            cv2.cvtColor(mosaic8, cv2.COLOR_BayerRG2GRAY).tolist(),
        )
        mosaic16b = (np.arange(8 * 8).reshape(8, 8) * 32).astype(np.uint16)
        self.assertEqual(
            cam.grayscale8_from_bytes("BayerBG12Packed", mosaic16b, 8, 8).tolist(),
            cv2.cvtColor((mosaic16b >> 4).astype(np.uint8), cv2.COLOR_BayerBG2GRAY).tolist(),
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
        "camera is wedged until I replug it" symptom.
        """
        from harvesters.core import Harvester

        h = Harvester()
        h.add_file(str(self.cti))
        h.update()
        names = [device.display_name for device in h.device_info_list]
        h.reset()
        self.assertGreater(len(names), 1, "TLSimu should expose several devices")

        target = names[-1]  # deliberately not index 0
        for attempt in range(3):
            harvester, ia = cam.setup_cam(
                cti=[str(self.cti)], search_key={"display_name": target}, fps=5
            )
            try:
                self.assertIsNotNone(ia, f"connect {attempt} produced no camera")
                frame = cam.get_harvester_image(ia)
                self.assertEqual(frame.dtype, np.uint8)
                self.assertEqual(frame.ndim, 2)
            finally:
                cam.teardown_cam(harvester, ia)

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
    window.ui.settings_window.teardown()


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
        img = cam.get_harvester_image(ia)
        self.assertEqual(img.shape, (6, 5))
        self.assertEqual(int(img.max()), 0)
        ia.fetch.assert_not_called()

    def test_recording_fetch_aborts_instead_of_blocking(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        ia.try_fetch.return_value = None
        out = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(TimeoutError):
            cam._harvester_frame_into(ia, out)
        ia.fetch.assert_not_called()

    def test_record_stops_the_camera_when_a_fetch_times_out(self):
        from unittest.mock import MagicMock

        ia = MagicMock()
        ia.try_fetch.return_value = None
        ia.remote_device.node_map.Height.value = 4
        ia.remote_device.node_map.Width.value = 4
        with self.assertRaises(TimeoutError):
            cam.record_harvester_frames(ia, 2, None)
        # A camera left acquiring keeps filling buffers nobody drains, and the
        # next start() then runs against an already-started device.
        ia.stop.assert_called_once()


class CameraReleaseTests(unittest.TestCase):
    """A camera must never stay open because something failed on the way in."""

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
        h.reset.assert_called_once()

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
            self.assertEqual(panel.timer.receivers(panel.timer.timeout), 1)

            panel.teardown()
            self.assertEqual(panel.timer.receivers(panel.timer.timeout), 0)
            self.assertFalse(panel.timer.isActive())
            self.assertEqual(panel.settings, {})
            self.assertEqual(panel.ui.formLayout.rowCount(), 0)
        finally:
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

if __name__ == "__main__":
    unittest.main()
