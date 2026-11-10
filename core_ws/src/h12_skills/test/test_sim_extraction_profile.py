"""Calibration data checks; no ROS initialization or gripper commands."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE))
from h12_skills.marker_targets import MarkerCatalog


class SimExtractionProfileTests(unittest.TestCase):
    def setUp(self):
        self.path = PACKAGE / 'h12_skills/marker_profile_sim_extraction.json'
        self.data = json.loads(self.path.read_text())
        self.baseline = json.loads((PACKAGE / 'h12_skills/marker_profile.json').read_text())

    def test_sim_calibration_retains_existing_admission_and_freshness(self):
        for name in ('linear_tolerance_m', 'angular_tolerance_rad', 'max_pose_age_sec',
                     'max_board_jump_m', 'max_board_jump_rad', 'max_goal_sec',
                     'active_pegs', 'top_down_rpy_pelvis_deg'):
            self.assertEqual(self.data[name], self.baseline[name])
        for arm in ('left', 'right'):
            for name in ('workspace_min_m', 'workspace_max_m'):
                self.assertEqual(self.data['arms'][arm][name], self.baseline['arms'][arm][name])

    def test_distal_contact_and_simulated_opening_command(self):
        catalog = ROOT / 'CL_Assets/mujoco_assets/manipulation_net/manipulation_catalog.json'
        profile = MarkerCatalog(catalog, self.path)
        goal = profile.geometry('cylinder_1', 'cylinder_1', 'left', top_down=True)
        np.testing.assert_allclose(goal.contact_offset, [0, .002238, .1786])
        np.testing.assert_allclose(goal.target_matrix(np.eye(4), 'grasp')[:3, 2],
                                   [0, 0, -1], atol=1e-12)
        command_mm = 1000 * goal.aperture_m
        self.assertAlmostEqual(command_mm, 46.65097, places=3)


if __name__ == '__main__':
    unittest.main()
