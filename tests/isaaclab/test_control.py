"""Check the Isaac sim's motor wire order, command packing and real-time pacing, without Kit."""
import unittest
import torch
from src.python.control.motor_contract import BODY_JOINT_NAMES, HAND_JOINT_NAMES, pack_motor_command
from src.python.control.pacing import RealtimePacer


class ControlTests(unittest.TestCase):
    """Check motor_contract and RealtimePacer."""

    def test_wire_order(self):
        """The body has 27 joints in Unitree wire order, left hip first, and the hands 12."""
        self.assertEqual(BODY_JOINT_NAMES[:3], ('left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint'))
        self.assertEqual(len(BODY_JOINT_NAMES), 27)
        self.assertEqual(len(HAND_JOINT_NAMES), 12)

    def test_packing_preserves_all_six_fields_and_ignores_unused_idl_slots(self):
        """Packing keeps all six motor fields and drops the IDL slots past the 27 body motors."""
        fields = [('positions',1.),('velocities',2.),('torques',3.),('kp',4.),('kd',5.),('mode',1)]
        cmd = {key: [value]*27+[999.]*8 for key,value in fields}
        packed = pack_motor_command(cmd, 'cpu').reshape(6,27)
        for row, (_, value) in zip(packed, fields):
            torch.testing.assert_close(row, torch.full((27,), float(value)))

    def test_bad_commands_are_rejected(self):
        """A non-finite value or a short field raises ValueError."""
        cmd = {k:[0.]*27 for k in ('positions','velocities','torques','kp','kd','mode')}
        cmd['positions'][0] = float('nan')
        with self.assertRaises(ValueError): pack_motor_command(cmd,'cpu')
        cmd['positions'] = [0.]*26
        with self.assertRaises(ValueError): pack_motor_command(cmd,'cpu')

    def test_pacing_100_steps_is_one_second(self):
        """100 paced steps of 10 ms take one second."""
        clock=[1.]
        p=RealtimePacer(.01, clock=lambda:clock[0], sleep=lambda t:clock.__setitem__(0,clock[0]+t))
        for _ in range(100): p.wait()
        self.assertAlmostEqual(clock[0],2.)

    def test_overrun_catches_up_without_sleeping_or_skipping(self):
        """After an overrun the pacer runs the late steps without sleeping, then sleeps again."""
        clock=[1.]; sleeps=[]
        def sleep(dt):
            """Record a sleep and advance the fake clock by it."""
            sleeps.append(dt); clock[0]+=dt
        p=RealtimePacer(.01,clock=lambda:clock[0],sleep=sleep)
        clock[0]+=.025
        p.wait(); p.wait(); self.assertEqual(sleeps,[])
        p.wait(); self.assertAlmostEqual(sleeps[0],.005)

if __name__ == '__main__': unittest.main()
