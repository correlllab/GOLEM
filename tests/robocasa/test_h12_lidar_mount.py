"""Check that the H1-2 sim's MID-360 hangs upside down, in the URDF's raw sensor frames.

Both MJCFs (the hand-written source and robosuite's robot.xml generated from it) must orient
livox_link and livox_imu_site as the URDF orients livox_link and livox_imu_link; the MID-360's
scan pattern, cast from that upside-down frame, must reach the floor below the robot and the
space above the horizon; and h12_sensors' scan thread must fire the pattern from livox_link.
Needs MuJoCo, NumPy and h12_sensors' imports (ROS, the Livox messages): run it with
tests/robocasa/run_in_container.sh.
"""
import math
import os
import time
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from livox_ros_driver2.msg import CustomMsg
from rclpy.serialization import deserialize_message

import h12_sensors
import mid360
from sim_timing import LoopTimings

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
    """Compare the URDF's lidar frames with both MJCFs, the scan pattern and the H1-2's scans."""

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

    def test_pattern_reaches_the_floor_and_above_the_horizon(self):
        """A frame of the MID-360's pattern reaches below -45 degrees and above the horizon, mostly downwards."""
        local = mid360.directions(mid360.load_pattern(), 0, 20_000)   # one 10 Hz frame
        for name, (model, data) in self.sims.items():
            with self.subTest(mjcf=name):
                rot = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'livox_link')]
                elevation = np.degrees(np.arcsin((local @ rot.reshape(3, 3).T)[:, 2]))
                # The MID-360's -7..+52 deg field, upside down: down to about -50 deg, and up past
                # the horizon on the side the mount's pitch lifts.
                self.assertLess(elevation.min(), -45.0)
                self.assertGreater(elevation.max(), 0.0)
                self.assertGreater(np.mean(elevation < 0.0), 0.5)

    def test_h12_frame_has_20000_rays_on_4_lines(self):
        """The H1-2's scan thread fires 20,000 pattern rays a frame from livox_link, numbered on lines 0-3.

        In a walled room around the H1-2 at its MJCF's default pose, with every geom of the
        robot's subtree giving no return: each frame casts the next 20,000 rows of the pattern
        and holds them in order, point k on line k % 4, each return along its ray and each ray
        without one at (0, 0, 0); most rays return.
        """
        spec = mujoco.MjSpec.from_file(MJCFS['mujoco'])
        for pos, size in (((3.05, 0, 1.5), (.05, 3.1, 1.5)), ((-3.05, 0, 1.5), (.05, 3.1, 1.5)),
                          ((0, 3.05, 1.5), (3.1, .05, 1.5)), ((0, -3.05, 1.5), (3.1, .05, 1.5))):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        torso = body('torso_link')
        robot = model.body_rootid[model.geom_bodyid] == model.body_rootid[torso]
        published = []
        scans = h12_sensors._LidarScans(model, body('livox_link'), torso, robot, 'cpu', 'livox_frame',
                                        published.append, lambda msg: None, LoopTimings())
        frames = []
        update = scans.scanner.update

        def record(snapshot, t):
            """Keep each frame the scanner completes."""
            frame = update(snapshot, t)
            if frame is not None:
                frames.append(frame)
            return frame

        scans.scanner.update = record
        try:
            dt = model.opt.timestep
            for k in range(round(0.2 / dt) + 1):   # two frames
                while scans._free.empty():
                    time.sleep(0.0005)
                scans.tick(data, k * dt)
            end = time.time() + 60.0
            while len(published) < 2 and time.time() < end:
                time.sleep(0.01)
        finally:
            scans.close()
        self.assertEqual(len(frames), 2)
        pattern = mid360.load_pattern()
        for f, frame in enumerate(frames):
            self.assertEqual(frame.rays, 20_000)
            self.assertEqual(len(frame.points), 20_000)
            self.assertEqual(deserialize_message(published[f], CustomMsg).point_num, 20_000)
            returns = frame.points.any(axis=1)
            self.assertGreater(returns.sum(), 0.9 * 20_000)
            rays = mid360.directions(pattern, 20_000 * f, 20_000)
            hits = frame.points[returns].astype(np.float64)
            np.testing.assert_allclose(hits / np.linalg.norm(hits, axis=1)[:, None], rays[returns],
                                       atol=1e-4)
            np.testing.assert_array_equal(frame.line, np.arange(20_000) % 4)


if __name__ == '__main__':
    unittest.main()
