"""Check the H1-2 sim's copy of the controller's start-up arm moves against golem_upper_ik's config."""
import ast
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'golem_robocasa' / 'h12_mujoco.py'
ROBOT = ROOT / 'core_ws' / 'src' / 'golem_upper_ik' / 'config' / 'h12' / 'robot.yaml'


def sim_stages():
    """Return h12_mujoco.STARTUP_ARM_STAGES, read from the source without importing the sim."""
    for node in ast.parse(SIM.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == 'STARTUP_ARM_STAGES' for t in node.targets):
            return [(name, list(arms)) for name, arms in ast.literal_eval(node.value)]
    raise AssertionError(f'{SIM} has no STARTUP_ARM_STAGES')


class StartupPoseTests(unittest.TestCase):
    """The spawn search clears the arm moves the controller really makes."""

    def test_stages_are_the_controllers_startup_sequence(self):
        """STARTUP_ARM_STAGES is robot.yaml's startup_sequence with its named_configs' values."""
        robot = yaml.safe_load(ROBOT.read_text())
        expected = [(name, [float(q) for q in robot['named_configs'][name]])
                    for name in robot['startup_sequence']]
        self.assertEqual([(n, [float(q) for q in a]) for n, a in sim_stages()], expected)


if __name__ == '__main__':
    unittest.main()
