"""The H1-2 RoboCasa sim's rt/lowstate carries the real H1-2's mode_machine.

SimInterface opens Unitree DDS channels on ROS_DOMAIN_ID, so run this
only where DDS cannot reach a robot, e.g. a throwaway robocasa container without a network:

    docker run --rm --network none -v "$PWD:/repo:ro" -w /repo -e PYTHONPATH=h1_robocasa \
        -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 golem_sim_robocasa:latest \
        -m unittest discover -s tests/robocasa -p test_h12_mode_machine.py -v

(discover, because the image has another `tests` package that a test path would import first).
"""
import os
import types
import unittest

# A sim domain, never the real robots' 0 (unitree_interface refuses it); a domain set by the
# caller wins.
os.environ.setdefault('ROS_DOMAIN_ID', '199')

import mujoco
import numpy as np

import unitree_interface

MODE_MACHINE = 6  # what the real H1-2 reports in every LowState_ (bags of real runs)


def body_model_and_resolver():
    """A dummy model with MOTOR_NUM hinge joints and the sensors and motors SimInterface reads
    through the resolver, without the kitchen scene."""
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
    @classmethod
    def setUpClass(cls):
        model, resolver = body_model_and_resolver()
        cls.iface = unitree_interface.SimInterface(model, mujoco.MjData(model), resolver=resolver)
        cls.addClassCleanup(cls.iface.low_state_thread.Wait)
        cls.addClassCleanup(cls.iface.timeout_thread.Wait)

    def test_h12_sim_lowstate_reports_6(self):
        self.assertEqual(self.iface.low_state.mode_machine, MODE_MACHINE)

    def test_published_sample_carries_it(self):
        published = []
        self.iface.low_state_publisher.Write = lambda sample: published.append(sample.mode_machine)
        self.iface.publish_low_state()
        self.assertEqual(published[-1], MODE_MACHINE)


if __name__ == '__main__':
    unittest.main()
