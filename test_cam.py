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

import os
import shutil
import tempfile
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


if __name__ == "__main__":
    unittest.main()
