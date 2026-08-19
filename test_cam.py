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
            frames = cam.record_harvester_frames(ia, 3, None)
            self.assertEqual(frames.ndim, 3)
            self.assertEqual(frames.shape[1:], (ia.remote_device.node_map.Height.value,
                                                ia.remote_device.node_map.Width.value))
            # fast path: owned, C-contiguous (n, h, w) stack
            self.assertTrue(frames.flags["C_CONTIGUOUS"])
            self.assertTrue(frames.flags["OWNDATA"])
            self.assertFalse(np.isfortran(frames[0]))
        finally:
            cam.disconnect_cam(h, ia)


if __name__ == "__main__":
    unittest.main()
