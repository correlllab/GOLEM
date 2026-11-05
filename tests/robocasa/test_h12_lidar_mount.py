"""Check that the H1-2 sim's MID-360 hangs upside down, in the URDF's raw sensor frames.

Both MJCFs (the hand-written source and robosuite's robot.xml generated from it) must orient
livox_link and livox_imu_site as the URDF orients livox_link and livox_imu_link, and
h12_sensors' ray fan, cast from that upside-down frame, must reach the floor below the robot
and the space above the horizon. Needs MuJoCo, NumPy and h12_sensors' imports (ROS, the
Livox messages): run it with tests/robocasa/run_in_container.sh.
"""
import math
import os
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import h12_sensors

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'CL_Assets')
URDF = os.path.join(ASSETS, 'ros_assets', 'h1_2_magpie.urdf')
MJCFS = {
    'robosuite': os.path.join(ASSETS, 'robosuite_assets', 'robots', 'h1_2', 'robot.xml'),
    'mujoco': os.path.join(ASSETS, 'mujoco_assets', 'h1_2_magpie.xml'),
}


def _rpy(roll, pitch, yaw):
    """Return the rotation matrix of a URDF rpy: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _urdf_rotation_in_pelvis(link):
    """Return `link`'s rotation in the pelvis frame from the URDF, every joint at zero."""
    joints = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        origin = joint.find('origin')
        rpy = [float(v) for v in (origin.get('rpy') or '0 0 0').split()] if origin is not None \
            else [0.0, 0.0, 0.0]
        joints[joint.find('child').get('link')] = (joint.find('parent').get('link'), _rpy(*rpy))
    rot = np.eye(3)
    while link != 'pelvis':
        link, r = joints[link]
        rot = r @ rot
    return rot


class LidarMountTests(unittest.TestCase):
    """Compare the URDF's lidar frames with both MJCFs and the ray fan."""

    @classmethod
    def setUpClass(cls):
        """Read the URDF's lidar rotations and load both MJCFs at their default pose."""
        cls.urdf_livox = _urdf_rotation_in_pelvis('livox_link')
        cls.urdf_imu = _urdf_rotation_in_pelvis('livox_imu_link')
        cls.sims = {}
        for name, path in MJCFS.items():
            model = mujoco.MjModel.from_xml_path(path)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            cls.sims[name] = (model, data)

    def _in_pelvis(self, model, data, kind, name):
        """Return a body's or site's rotation in the pelvis frame; `kind` is 'body' or 'site'."""
        pelvis = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')]
        if kind == 'body':
            rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
        else:
            rot = data.site_xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)]
        return pelvis.reshape(3, 3).T @ rot.reshape(3, 3)

    def test_urdf_frames_are_upside_down(self):
        """The URDF's lidar and IMU frames point their z axis within 20 degrees of straight down."""
        self.assertLess(self.urdf_livox[2, 2], -math.cos(math.radians(20.0)))
        self.assertLess(self.urdf_imu[2, 2], -math.cos(math.radians(20.0)))

    def test_mjcf_frames_match_the_urdf(self):
        """Both MJCFs' lidar body, lidar site and IMU site have the URDF's orientations."""
        for name, (model, data) in self.sims.items():
            with self.subTest(mjcf=name):
                np.testing.assert_allclose(self._in_pelvis(model, data, 'body', 'livox_link'),
                                           self.urdf_livox, atol=1e-6)
                np.testing.assert_allclose(self._in_pelvis(model, data, 'site', 'livox_site'),
                                           self.urdf_livox, atol=1e-6)
                np.testing.assert_allclose(self._in_pelvis(model, data, 'site', 'livox_imu_site'),
                                           self.urdf_imu, atol=1e-6)

    def test_fan_reaches_the_floor_and_above_the_horizon(self):
        """The fan reaches below -45 degrees and above the horizon, mostly downwards."""
        local = h12_sensors.lidar_fan(360, 56)  # the azimuth x elevation grid h12_mujoco uses
        for name, (model, data) in self.sims.items():
            with self.subTest(mjcf=name):
                rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'livox_link')]
                elevation = np.degrees(np.arcsin((local @ rot.reshape(3, 3).T)[:, 2]))
                # The MID-360's -7..+52 deg field, upside down: down to about -50 deg, and up past
                # the horizon on the side the mount's pitch lifts.
                self.assertLess(elevation.min(), -45.0)
                self.assertGreater(elevation.max(), 0.0)
                self.assertGreater(np.mean(elevation < 0.0), 0.5)


if __name__ == '__main__':
    unittest.main()
