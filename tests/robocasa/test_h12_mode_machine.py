"""Check that the H1-2 RoboCasa sim's rt/lowstate carries the real H1-2's mode_machine.

The test opens no DDS participant: unitree_interface's DDS channel names and cyclonedds'
ddspy_write are replaced by in-process fakes (dds_fakes.py) for the length of each test. It runs in the
standard runner (tests/robocasa/run_in_container.sh) and in a throwaway container without a
network:

    docker run --rm --network none -v "$PWD:/repo:ro" -w /repo -e PYTHONPATH=golem_robocasa \
        -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 golem_sim_robocasa:latest \
        -m unittest discover -s tests/robocasa -p test_h12_mode_machine.py -v

(discover, because the image has another `tests` package that a test path would import first).
"""
import os
import types
import unittest
from unittest import mock

import mujoco
import numpy as np

import unitree_interface
from dds_fakes import fake_dds

MODE_MACHINE = 6  # what the real H1-2 reports in every LowState_, in recordings of real runs


def dummy_model_and_resolver():
    """Return a model of MOTOR_NUM hinge joints, with their motors and torque sensors, and its resolver.

    It holds what SimInterface reads through the resolver, without the kitchen scene.
    """
    n = unitree_interface.MOTOR_NUM
    bodies = ''.join(f'<body><joint name="j{i}" axis="0 0 1"/><geom size="0.01"/></body>'
                     for i in range(n))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"/>'
        f'<worldbody>{bodies}</worldbody>'
        '<actuator>' + ''.join(f'<motor joint="j{i}"/>' for i in range(n)) + '</actuator>'
        '<sensor>' + ''.join(f'<jointactuatorfrc joint="j{i}"/>' for i in range(n)) + '</sensor>'
        '</mujoco>')
    resolver = types.SimpleNamespace(
        motor_qpos=model.jnt_qposadr[:n], motor_qvel=model.jnt_dofadr[:n],
        motor_tau=model.sensor_adr[:n], motor_ctrl=np.arange(n), sensor_adr={})
    return model, resolver


class H12SimLowStateTests(unittest.TestCase):
    """Run SimInterface on the stand-in model with the DDS channels faked."""

    def setUp(self):
        """Fake the DDS channels and make a SimInterface on a stand-in model."""
        fake_dds(self, unitree_interface)
        model, resolver = dummy_model_and_resolver()
        self.iface = unitree_interface.SimInterface(model, mujoco.MjData(model), resolver=resolver)

    def test_h12_sim_lowstate_reports_6(self):
        """The sim's rt/lowstate writer reports mode_machine 6, on the numpy path."""
        self.assertEqual(self.iface._low_state.mode_machine, MODE_MACHINE)
        self.assertTrue(self.iface._low_state.fast)

    def test_published_sample_carries_it(self):
        """The sample written after a physics step carries mode_machine 6."""
        self.iface.after_step()
        pub = self.iface._low_state_pub
        self.assertEqual((len(pub.sent), pub.last().mode_machine), (1, MODE_MACHINE))


class DomainIdTests(unittest.TestCase):
    """Check which ROS_DOMAIN_ID values the sim's DDS start-up accepts."""

    def test_refuses_unset_and_real_robot_domain(self):
        """An unset, empty, zero or non-numeric ROS_DOMAIN_ID raises."""
        for env in ({}, {'ROS_DOMAIN_ID': ''}, {'ROS_DOMAIN_ID': '0'}, {'ROS_DOMAIN_ID': 'x'}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    unitree_interface.domain_id()

    def test_reads_sim_domain(self):
        """A sim domain is read as its number."""
        with mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '7'}):
            self.assertEqual(unitree_interface.domain_id(), 7)


if __name__ == '__main__':
    unittest.main()
