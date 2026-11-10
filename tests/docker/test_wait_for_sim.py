"""Check `wait_for_sim.py`'s robot-identity check against a fake Unitree SDK, without DDS or the SDK."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'docker/scripts/wait_for_sim.py'

# The parts of unitree_sdk2py that wait_for_sim.py imports. The subscriber's first sample is a
# LowState_ whose mode_machine is $FAKE_MODE_MACHINE, or no sample at all when that is 'none'.
FAKE_SDK = {
    'unitree_sdk2py/__init__.py': '',
    'unitree_sdk2py/core/__init__.py': '',
    'unitree_sdk2py/core/channel.py': '''
import os


def ChannelFactoryInitialize(domain, interface=None):
    pass


class ChannelSubscriber:
    def __init__(self, topic, kind):
        assert topic == 'rt/lowstate', topic
        self.kind = kind

    def Init(self):
        pass

    def Read(self, timeout):
        value = os.environ['FAKE_MODE_MACHINE']
        if value == 'none':
            return None
        sample = self.kind()
        sample.mode_machine = int(value)
        return sample

    def Close(self):
        pass
''',
    'unitree_sdk2py/idl/__init__.py': '',
    'unitree_sdk2py/idl/unitree_hg/__init__.py': '',
    'unitree_sdk2py/idl/unitree_hg/msg/__init__.py': '',
    'unitree_sdk2py/idl/unitree_hg/msg/dds_/__init__.py': '''
class LowState_:
    def __init__(self):
        self.mode_pr = 0
        self.mode_machine = 0
''',
}


def load_wait_for_sim():
    """Import `wait_for_sim.py` as a module."""
    spec = importlib.util.spec_from_file_location('wait_for_sim', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExpectedModeMachineTests(unittest.TestCase):
    """Check the `mode_machine` each robot's sims report."""

    def test_each_robot_has_its_simulators_mode_machine(self):
        """The H1-2 expects `mode_machine` 6 and the G1 expects 5."""
        wait_for_sim = load_wait_for_sim()
        self.assertEqual(wait_for_sim.expected_mode_machine('h12'), 6)
        self.assertEqual(wait_for_sim.expected_mode_machine('g1'), 5)


class RobotIdentityTests(unittest.TestCase):
    """Run `wait_for_sim.py` against a fake sim and check which robot it accepts."""

    def setUp(self):
        """Write the fake SDK into a temporary directory."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.sdk = Path(tmp.name)
        for name, text in FAKE_SDK.items():
            path = self.sdk / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def wait(self, mode_machine, robot=None):
        """Run `wait_for_sim.py` against a sim whose first LowState_ carries `mode_machine`.

        `robot` is the `GOLEM_ROBOT` value, or None to leave it unset.
        """
        env = {k: v for k, v in os.environ.items() if k not in ('GOLEM_ROBOT', 'PYTHONPATH')}
        env.update(ROS_DOMAIN_ID='7', GOLEM_SIM_TIMEOUT='1', FAKE_MODE_MACHINE=str(mode_machine),
                   PYTHONPATH=str(self.sdk), PYTHONDONTWRITEBYTECODE='1')
        if robot is not None:
            env['GOLEM_ROBOT'] = robot
        return subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True,
                              capture_output=True, timeout=60)

    def assertRefused(self, result, message):
        """Assert the run failed with `message` on stderr and never reported the sim publishing."""
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertNotIn('simulator is publishing', result.stdout)

    def assertAccepted(self, result):
        """Assert the run exited 0 after reporting the sim publishing."""
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('simulator is publishing', result.stdout)

    def test_g1_stack_refuses_the_h12_simulator(self):
        """A G1 stack refuses a sim reporting the H1-2's `mode_machine`."""
        self.assertRefused(self.wait(6, 'g1'), 'the simulator is not a g1 (mode_machine 6)')

    def test_h12_stack_accepts_the_h12_simulator(self):
        """An H1-2 stack accepts a sim reporting `mode_machine` 6."""
        self.assertAccepted(self.wait(6, 'h12'))

    def test_h12_stack_refuses_the_g1_simulator(self):
        """An H1-2 stack refuses a sim reporting the G1's `mode_machine`."""
        self.assertRefused(self.wait(5, 'h12'), 'the simulator is not a h12 (mode_machine 5)')

    def test_h12_stack_refuses_a_simulator_that_reports_the_idl_default(self):
        """An H1-2 stack refuses a sim reporting the IDL default `mode_machine` 0."""
        self.assertRefused(self.wait(0, 'h12'), 'the simulator is not a h12 (mode_machine 0)')

    def test_g1_stack_accepts_the_g1_simulator(self):
        """A G1 stack accepts a sim reporting `mode_machine` 5."""
        self.assertAccepted(self.wait(5, 'g1'))

    def test_unset_or_empty_robot_is_h12(self):
        """An unset or empty `GOLEM_ROBOT` expects the H1-2."""
        for robot in (None, ''):
            with self.subTest(GOLEM_ROBOT=robot):
                self.assertAccepted(self.wait(6, robot))
                self.assertRefused(self.wait(5, robot), 'the simulator is not a h12 (mode_machine 5)')

    def test_robot_name_is_case_insensitive(self):
        """`GOLEM_ROBOT=G1` selects the G1."""
        self.assertAccepted(self.wait(5, 'G1'))

    def test_unknown_robot_is_refused_before_waiting(self):
        """An unknown `GOLEM_ROBOT` is refused with the allowed names."""
        self.assertRefused(self.wait(6, 'h1'), "GOLEM_ROBOT must be h12 or g1, not 'h1'")

    def test_no_sample_still_fails(self):
        """No `rt/lowstate` sample within the timeout fails."""
        self.assertRefused(self.wait('none', 'g1'), 'no simulator sample received')


if __name__ == '__main__':
    unittest.main()
