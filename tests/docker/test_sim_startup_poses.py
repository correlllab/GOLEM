"""Check each robot's sim copy of the controller's start-up arm moves against golem_upper_ik's config."""
import ast
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Each robot's sim module and its controller's robot.yaml.
ROBOTS = {
    'h12': (ROOT / 'golem_robocasa' / 'h12_robot.py',
            ROOT / 'core_ws' / 'src' / 'golem_upper_ik' / 'config' / 'h12' / 'robot.yaml'),
    'g1': (ROOT / 'golem_robocasa' / 'g1' / 'g1_robot.py',
           ROOT / 'core_ws' / 'src' / 'golem_upper_ik' / 'config' / 'g1' / 'robot.yaml'),
}


def sim_stages(sim):
    """Return the module's STARTUP_ARM_STAGES, read from the source without importing the sim."""
    for node in ast.parse(sim.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == 'STARTUP_ARM_STAGES' for t in node.targets):
            return [(name, list(arms)) for name, arms in ast.literal_eval(node.value)]
    raise AssertionError(f'{sim} has no STARTUP_ARM_STAGES')


class StartupPoseTests(unittest.TestCase):
    """The spawn search clears the arm moves the controller really makes."""

    def test_stages_are_the_controllers_startup_sequence(self):
        """Each robot's STARTUP_ARM_STAGES is its robot.yaml's startup_sequence with its named_configs' values."""
        for name, (sim, config) in ROBOTS.items():
            with self.subTest(robot=name):
                robot = yaml.safe_load(config.read_text())
                expected = [(stage, [float(q) for q in robot['named_configs'][stage]])
                            for stage in robot['startup_sequence']]
                self.assertEqual([(n, [float(q) for q in a]) for n, a in sim_stages(sim)], expected)


if __name__ == '__main__':
    unittest.main()
