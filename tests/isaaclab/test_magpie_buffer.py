"""Check the Magpie command buffer between the ROS services and the physics loop."""
import unittest
from src.python.sensors.magpie_bridge import MagpieCommandBuffer


class MagpieBufferTests(unittest.TestCase):
    """Check MagpieCommandBuffer's validation and acknowledgement."""

    def test_commands_need_physics_ack_and_snapshots_are_independent(self):
        """A command counts as applied only once the physics acknowledges it, and snapshots are copies."""
        buffer = MagpieCommandBuffer()
        sequence = buffer.submit('left', {'aperture': 40., 'force': 8., 'speed': .5})
        commands = buffer.get_commands()
        self.assertFalse(buffer.is_applied('left', sequence))
        buffer.acknowledge(commands)
        self.assertTrue(buffer.is_applied('left', sequence))
        commands['left']['aperture'] = 99.
        self.assertEqual(buffer.get_commands()['left']['aperture'], 40.)
        self.assertEqual(buffer.get_commands()['right']['aperture'], 110.)

    def test_invalid_commands_never_replace_valid_targets(self):
        """Non-finite or out-of-range commands raise and leave the targets unchanged."""
        buffer = MagpieCommandBuffer()
        for values in ({'aperture': float('nan')}, {'aperture': -1}, {'force': 101}, {'speed': 0}, {'speed': 2}):
            with self.assertRaises(ValueError):
                buffer.submit('left', values)
        self.assertEqual(buffer.get_commands()['left']['sequence'], 0)

    def test_superseded_target_is_not_reported_applied(self):
        """A command replaced before the acknowledgement is not reported applied."""
        buffer = MagpieCommandBuffer()
        older = buffer.submit('left', {'aperture': 50})
        buffer.submit('left', {'aperture': 30})
        buffer.acknowledge(buffer.get_commands())
        self.assertFalse(buffer.is_applied('left', older))

    def test_old_ack_does_not_ack_new_command(self):
        """An acknowledgement of an older snapshot does not apply a newer command."""
        buffer = MagpieCommandBuffer()
        buffer.submit('left', {'aperture': 50})
        old = buffer.get_commands()
        newer = buffer.submit('left', {'aperture': 30})
        buffer.acknowledge(old)
        self.assertFalse(buffer.is_applied('left', newer))


if __name__ == '__main__':
    unittest.main()
