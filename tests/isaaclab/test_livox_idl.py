"""Check Isaac's Livox CustomMsg IDL against the pinned driver's wire type."""
import unittest
from src.python.sensors.livox_dds import CustomMsg_, CustomPoint_, Header_, Time_


class LivoxIdlTests(unittest.TestCase):
    """Serialise and deserialise a CustomMsg."""

    def test_cdr_roundtrip_preserves_header_and_points(self):
        """A CDR round trip keeps the header and points, under the driver's DDS type name."""
        original = CustomMsg_(Header_(Time_(7, 123), 'lidar_link'), 7000000123, 1, 0, bytes(3),
                              [CustomPoint_(500, 1., 2., 3., 0, 0, 5)])
        decoded = CustomMsg_.deserialize(original.serialize())
        self.assertEqual(decoded, original)
        self.assertEqual(CustomMsg_.__idl_typename__, 'livox_ros_driver2.msg.dds_.CustomMsg_')


if __name__ == '__main__':
    unittest.main()
