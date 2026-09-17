"""Pure scoring regressions: action acknowledgement alone cannot pass physics checks."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('frame_check', Path(__file__).with_name('check_frame_task.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ArmMetricsTests(unittest.TestCase):
    def setUp(self):
        self.start = [0., .2, .3, 0., 0., 0., 1.]
        self.goal = [.02, .2, .3, 0., 0., 0., 1.]

    def score(self, pose, joints):
        return module.arm_metrics(self.start, self.goal, pose, [0.] * 7, joints)

    def test_ack_without_motion_fails(self):
        self.assertFalse(module.motion_passes(self.score(self.start, [0.] * 7)))

    def test_pose_without_measured_joint_motion_fails(self):
        self.assertFalse(module.motion_passes(self.score(self.goal, [0.] * 7)))

    def test_motion_away_from_target_fails(self):
        wrong = self.goal.copy()
        wrong[0] = -.02
        self.assertFalse(module.motion_passes(self.score(wrong, [.02] * 7)))

    def test_measured_pose_and_joint_progress_pass(self):
        self.assertTrue(module.motion_passes(self.score(self.goal, [.02] * 7)))

    def test_quaternion_sign_is_equivalent(self):
        pose = self.goal.copy()
        pose[-1] = -1.
        self.assertEqual(self.score(pose, [.02] * 7)['orientation_error_rad'], 0.)

    def test_nonfinite_state_is_rejected(self):
        with self.assertRaises(ValueError):
            self.score(self.goal, [float('nan')] * 7)

if __name__ == '__main__':
    unittest.main()
