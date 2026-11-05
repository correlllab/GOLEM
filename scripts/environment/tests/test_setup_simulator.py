"""Command composition tests; no environment installation or ROS initialization."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'setup_simulator.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('setup_simulator', SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class NativeSimulatorTests(unittest.TestCase):
    def test_help_is_independent_of_cwd_and_ros(self):
        result = subprocess.run([str(setup.ROOT / 'scripts/setup_environment.sh'), '--help'], cwd='/tmp/kilo',
                                env={'HOME': os.environ['HOME'], 'PATH': '/usr/bin:/bin',
                                     'PYTHONPATH': '/invalid', 'VIRTUAL_ENV': '/invalid'},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('No command launches nodes', result.stdout)

    def test_system_python_and_project_are_separate(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/opt/ros/humble/local/lib/python3.10/dist-packages'}, clear=True):
            env = setup.environment()
        self.assertEqual(env['UV_PROJECT_ENVIRONMENT'], str(setup.VENV))
        self.assertEqual(env['UV_PYTHON_DOWNLOADS'], 'never')
        self.assertTrue(env['PYTHONPATH'].startswith(str(setup.ROOT / 'unitree_sdk2_python')))
        self.assertNotIn('core_ws/environments', env['PATH'])

    def test_check_never_constructs_ros_or_simulator(self):
        with patch.object(setup, 'require_venv'), patch.object(setup, 'run') as run:
            setup.check()
        command = run.call_args.args[0]
        self.assertEqual(command, [setup.PYTHON, setup.HERE / 'check_simulator.py'])

    def test_bridge_check_uses_local_not_saved_underlays(self):
        with patch.object(setup, 'require_venv'), patch.object(setup, 'run') as run, \
                patch.object(Path, 'is_file', return_value=True):
            setup.check(True)
        command = run.call_args.args[0]
        self.assertIn(setup.ROOT / 'core_ws/install/local_setup.bash', command)
        self.assertIn('--bridges', command)
        self.assertNotIn('setup.bash', command)

    def test_reject_changed_livox_pin_before_writes(self):
        with patch.object(Path, 'is_file', return_value=True), \
                patch.object(setup.subprocess, 'check_output', return_value='unexpected\n'), \
                patch.object(setup, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Livox must remain'):
                setup.build_livox(2)
        run.assert_not_called()

    def test_reject_modified_message_before_writes(self):
        with patch.object(Path, 'is_file', return_value=True), \
                patch.object(setup.subprocess, 'check_output', side_effect=[setup.LIVOX_SHA, b'invalid']), \
                patch.object(setup, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'differs from pinned source'):
                setup.build_livox(2)
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
