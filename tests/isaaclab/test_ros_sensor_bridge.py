"""Sensor wire-format checks; run with Isaac's bundled Python and ROS libraries."""
import io
import struct
import unittest

import numpy as np
from PIL import Image as PILImage

from src.python.sensors.ros_sensor_bridge import depth_millimeters, encode_depth, points_in_sensor_frame, stamp_parts, lidar_payload, POINT_DTYPE


class SensorPayloadTests(unittest.TestCase):
    def test_invalid_depth_is_zero_and_valid_depth_is_millimeters(self):
        depth = np.array([[1.25, np.nan, np.inf, -1, 0, 80]], dtype=np.float32)
        np.testing.assert_array_equal(depth_millimeters(depth), [[1250, 0, 0, 0, 0, 65535]])

    def test_compressed_depth_decodes_with_transport_header(self):
        mm = np.array([[0, 1234, 65535]], dtype=np.uint16)
        payload = encode_depth(mm)
        self.assertEqual(struct.unpack('<Iff', payload[:12]), (0, 0.0, 0.0))
        np.testing.assert_array_equal(np.asarray(PILImage.open(io.BytesIO(payload[12:]))), mm)

    def test_world_hits_are_transformed_to_sensor_frame_and_misses_dropped(self):
        q = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        points = points_in_sensor_frame(np.array([[1., 3., 3.], [np.inf, 0, 0]]), [1, 2, 3], q)
        np.testing.assert_allclose(points, [[1., 0., 0.]], atol=1e-6)

    def test_lidar_payload_filters_misses_and_keeps_scan_metadata(self):
        hits = np.array([[1, 0, 0], [np.inf, 0, 0], [3, 0, 0]], dtype=float)
        payload = lidar_payload(hits, [0, 0, 0], [1, 0, 0, 0], horizontal_samples=2, period=.1)
        self.assertEqual(payload.dtype, POINT_DTYPE)
        self.assertEqual(payload['line'].tolist(), [0, 1])
        self.assertEqual(payload['timestamp'].tolist(), [0., 0.])
        np.testing.assert_array_equal(payload['x'], [1, 3])

    def test_stamp_carries_nanoseconds_into_seconds(self):
        self.assertEqual(stamp_parts(1.9999999996), (2, 0))
        with self.assertRaises(ValueError):
            stamp_parts(float('nan'))


if __name__ == '__main__':
    unittest.main()
