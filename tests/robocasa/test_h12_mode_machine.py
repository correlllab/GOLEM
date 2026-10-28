"""The H1-2 RoboCasa sim's rt/lowstate carries the real H1-2's mode_machine.

The test opens no DDS participant and starts no thread: unitree_interface's DDS channel
and thread names are replaced by in-process fakes for the length of each test. It runs
in the standard runner (tests/robocasa/run_in_container.sh) and in a throwaway container
without a network:

    docker run --rm --network none -v "$PWD:/repo:ro" -w /repo -e PYTHONPATH=h1_robocasa \
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

# unitree_interface reads ROS_DOMAIN_ID once, at import, and refuses a missing value and the
# real robots' 0. The fakes below open nothing, so this value reaches no network; it is set
# for the import only.
with mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '199'}):
    import unitree_interface

MODE_MACHINE = 6  # what the real H1-2 reports in every LowState_ (bags of real runs)


class FakePublisher:
    """Stands in for ChannelPublisher: records the mode_machine of each sample written."""
    written = []

    def __init__(self, topic, kind):
        self.topic = topic

    def Init(self):
        pass

    def Write(self, sample):
        self.written.append(sample.mode_machine)


class FakeSubscriber:
    def __init__(self, topic, kind):
        self.topic = topic

    def Init(self, handler, queue_size):
        pass


class FakeThread:
    """Stands in for RecurrentThread: never runs its target."""
    def __init__(self, interval, target, name):
        self.target = target

    def Start(self):
        pass


def dummy_model_and_resolver():
    """A model with MOTOR_NUM hinge joints and the sensors and motors SimInterface reads
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
    def setUp(self):
        FakePublisher.written = []
        for name, fake in (('ChannelFactoryInitialize', lambda id: None),
                           ('ChannelPublisher', FakePublisher),
                           ('ChannelSubscriber', FakeSubscriber),
                           ('RecurrentThread', FakeThread)):
            patcher = mock.patch.object(unitree_interface, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        model, resolver = dummy_model_and_resolver()
        self.iface = unitree_interface.SimInterface(model, mujoco.MjData(model), resolver=resolver)

    def test_h12_sim_lowstate_reports_6(self):
        self.assertEqual(self.iface.low_state.mode_machine, MODE_MACHINE)

    def test_published_sample_carries_it(self):
        self.iface.publish_low_state()
        self.assertEqual(FakePublisher.written, [MODE_MACHINE])


if __name__ == '__main__':
    unittest.main()
