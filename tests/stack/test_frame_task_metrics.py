"""Check check_frame_task.py's arm scoring: a succeeded action alone cannot pass."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('frame_check', Path(__file__).with_name('check_frame_task.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ArmMetricsTests(unittest.TestCase):
    """Score synthetic arm moves 2 cm forward in the pelvis frame."""

    def setUp(self):
        """Set a start pose and a goal 2 cm ahead of it (XYZ, then quaternion XYZW)."""
        self.start = [0., .2, .3, 0., 0., 0., 1.]
        self.goal = [.02, .2, .3, 0., 0., 0., 1.]

    def score(self, pose, joints):
        """Score a measured pose and arm joints against the start and goal, from zero joints."""
        return module.arm_metrics(self.start, self.goal, pose, [0.] * 7, joints)

    def test_ack_without_motion_fails(self):
        """An arm that stays at its start fails."""
        self.assertFalse(module.motion_passes(self.score(self.start, [0.] * 7)))

    def test_pose_without_measured_joint_motion_fails(self):
        """A pose at the goal with no measured joint motion fails."""
        self.assertFalse(module.motion_passes(self.score(self.goal, [0.] * 7)))

    def test_motion_away_from_target_fails(self):
        """A move the same distance in the wrong direction fails."""
        wrong = self.goal.copy()
        wrong[0] = -.02
        self.assertFalse(module.motion_passes(self.score(wrong, [.02] * 7)))

    def test_measured_pose_and_joint_progress_pass(self):
        """A pose at the goal with measured joint motion passes."""
        self.assertTrue(module.motion_passes(self.score(self.goal, [.02] * 7)))

    def test_quaternion_sign_is_equivalent(self):
        """A quaternion and its negation score zero orientation error."""
        pose = self.goal.copy()
        pose[-1] = -1.
        self.assertEqual(self.score(pose, [.02] * 7)['orientation_error_rad'], 0.)

    def test_nonfinite_state_is_rejected(self):
        """Non-finite joint positions raise ValueError."""
        with self.assertRaises(ValueError):
            self.score(self.goal, [float('nan')] * 7)

if __name__ == '__main__':
    unittest.main()
