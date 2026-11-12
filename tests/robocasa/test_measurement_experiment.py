"""Check measurement_bridge.py's opt-in experiment surface (/sim/ground_truth, /sim/reset_objects).

The bridge is built on a small scene with recording publishers and services: no ROS node starts
and no DDS traffic flows.
"""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from rclpy.node import Node

from measurement_bridge import MeasurementBridge

# A robot (prefix robot0_) with its feet, a part with a hinged link under it and a loose lid.
SCENE = '''<mujoco>
  <worldbody>
    <geom type="plane" size="4 4 .1"/>
    <body name="robot0_base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="10"/>
      <body name="robot0_left_ankle_roll_link" pos="0 .1 -.9">
        <geom type="sphere" size=".05" mass="1"/>
      </body>
      <body name="robot0_right_ankle_roll_link" pos="0 -.1 -.9">
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </body>
    <body name="part" pos="1 0 .5">
      <freejoint/>
      <geom type="box" size=".05 .05 .05" mass=".2"/>
      <body name="segment" pos=".1 0 0">
        <joint name="segment_hinge" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 .1 0 0" size=".01" mass=".01"/>
      </body>
    </body>
    <body name="lid" pos="-1 0 .5">
      <freejoint/>
      <geom type="box" size=".05 .05 .01" mass=".1"/>
    </body>
  </worldbody>
</mujoco>'''


class Recorder:
    """Stand in for a ROS publisher: record what is published."""

    def __init__(self):
        """Start with no messages."""
        self.messages = []

    def publish(self, msg):
        """Record msg."""
        self.messages.append(msg)


def make_env():
    """Return a stand-in env on SCENE, shaped as the bridge reads a robosuite env."""
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        sim=SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data)),
        robots=[SimpleNamespace(robot_model=SimpleNamespace(naming_prefix='robot0_'))],
        _check_success=lambda: False,
        get_ep_meta=lambda: {})


def make_bridge(env, experiment):
    """Build a MeasurementBridge on env without a ROS node, with GOLEM_SIM_EXPERIMENT=experiment."""
    publishers, services = {}, {}

    def create_publisher(self, msg_type, topic, qos):
        """Record and return a publisher for topic."""
        publishers[topic] = Recorder()
        return publishers[topic]

    def create_service(self, srv_type, name, callback):
        """Record name's callback."""
        services[name] = callback

    logger = SimpleNamespace(warn=lambda *a, **k: None, info=lambda *a, **k: None)
    with patch.dict(os.environ, {'GOLEM_SIM_EXPERIMENT': experiment}), \
            patch.object(Node, '__init__', lambda self, name: None), \
            patch.object(MeasurementBridge, 'create_publisher', create_publisher), \
            patch.object(MeasurementBridge, 'create_service', create_service), \
            patch.object(MeasurementBridge, 'get_logger', lambda self: logger):
        bridge = MeasurementBridge(env)
    return bridge, publishers, services


class ExperimentSurfaceTest(unittest.TestCase):
    """The experiment surface: off by default, ground truth and object reset when on."""

    def test_off_by_default(self):
        """Unset or 0, the bridge publishes no ground truth and serves no reset."""
        for value in ('', '0', 'off'):
            bridge, publishers, services = make_bridge(make_env(), value)
            bridge.tick()
            self.assertNotIn('/sim/ground_truth', publishers)
            self.assertNotIn('/sim/reset_objects', services)

    def test_ground_truth_names_every_object_link_and_foot(self):
        """Each tick publishes the objects, the links under them, the feet, the pelvis and the CoM."""
        env = make_env()
        bridge, publishers, _ = make_bridge(env, '1')
        bridge.tick()
        gt = json.loads(publishers['/sim/ground_truth'].messages[-1].data)
        self.assertEqual(sorted(gt['objects']), ['lid', 'part'])
        self.assertEqual(sorted(gt['links']), ['segment'])
        self.assertEqual(sorted(gt['feet']), ['left', 'right'])
        np.testing.assert_allclose(gt['objects']['part'], [1, 0, .5, 1, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(gt['pelvis'][:3], [0, 0, 1], atol=1e-9)
        self.assertEqual(len(gt['robot_com']), 3)

    def test_reset_puts_objects_and_their_joints_back_at_rest(self):
        """A reset restores every object joint's authored pose with zero velocity, leaving the robot."""
        env = make_env()
        model, data = env.sim.model._model, env.sim.data._data
        bridge, _, services = make_bridge(env, '1')
        data.qpos[:] += 0.3
        data.qvel[:] = 1.0
        robot_qpos = data.qpos[:7].copy()
        response = services['/sim/reset_objects'](None, SimpleNamespace(success=False, message=''))
        self.assertTrue(response.success)
        bridge.tick()
        np.testing.assert_allclose(data.qpos[7:], model.qpos0[7:])
        np.testing.assert_allclose(data.qvel[6:], 0.0)
        np.testing.assert_allclose(data.qpos[:7], robot_qpos)


if __name__ == '__main__':
    unittest.main()
