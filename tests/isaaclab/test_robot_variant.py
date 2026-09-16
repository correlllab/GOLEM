"""Asset-specific joint and sensor contracts, runnable without Isaac Sim."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'))
from src.python.control.robot_variant import robot_variant


class RobotVariantTests(unittest.TestCase):
    def test_magpie_is_default_and_has_no_inspire_joints(self):
        variant = robot_variant()
        self.assertIn('h1_2_magpie/h1_2_magpie.usd', variant.usd_path)
        self.assertEqual(len(variant.hand_joints), 12)
        self.assertEqual(variant.hand_joints[0], 'lg_left_hinge_1')
        self.assertEqual(variant.head_link, 'head_camera_link')
        self.assertEqual(variant.lidar_link, 'livox_link')
        self.assertFalse(variant.inspire)

    def test_inspire_keeps_existing_contract(self):
        variant = robot_variant('inspire')
        self.assertEqual(variant.head_link, 'camera_link')
        self.assertEqual(variant.lidar_link, 'lidar_link')
        self.assertTrue(variant.inspire)
        self.assertEqual(variant.hand_joints[0], 'R_pinky_proximal_joint')

    def test_unknown_variant_rejected(self):
        with self.assertRaises(ValueError):
            robot_variant('unknown')
