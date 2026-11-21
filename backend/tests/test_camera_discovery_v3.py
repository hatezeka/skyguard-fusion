import unittest
from unittest.mock import Mock
import numpy as np
from app.camera_management.discovery import probe_usb, probe_selected

class DiscoveryTests(unittest.TestCase):
    def test_reads_frame_dimensions_not_just_driver(self):
        c = Mock()
        c.isOpened.return_value = True
        c.get.side_effect = lambda prop: {3: 1920, 4: 1080, 5: 30}.get(prop, 0)
        c.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        r = probe_usb(0, lambda _: c)
        self.assertTrue(r.available)
        self.assertEqual((r.width_px, r.height_px), (640, 480))
        self.assertEqual(r.fps_reported, 30)
        self.assertFalse(r.fps_verified)
        c.release.assert_called_once()

    def test_unavailable_releases(self):
        c = Mock()
        c.isOpened.return_value = False
        r = probe_usb(1, lambda _: c)
        self.assertFalse(r.available)
        c.release.assert_called_once()

    def test_invalid_index_does_not_open(self):
        factory = Mock()
        with self.assertRaises(ValueError):
            probe_usb(-1, factory)
        factory.assert_not_called()

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            probe_selected([0, 0])

    def test_frame_failure(self):
        c = Mock()
        c.isOpened.return_value = True
        c.get.return_value = 30
        c.read.return_value = (False, None)
        self.assertEqual(probe_usb(0, lambda _: c).error, 'frame_unavailable')
        c.release.assert_called_once()

if __name__ == '__main__':
    unittest.main()
