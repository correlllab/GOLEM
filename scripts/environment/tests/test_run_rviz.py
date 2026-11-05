"""RViz entrypoint validation; never starts a ROS participant or GUI."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('run_rviz', ROOT / 'scripts/environment/run_rviz.py')
rviz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rviz)


class RvizTests(unittest.TestCase):
    def test_config_and_package_metadata(self):
        ET.parse(ROOT / 'core_ws/src/h12_bringup/package.xml')
        ET.parse(ROOT / 'core_ws/src/model_server/package.xml')
        config = yaml.safe_load((ROOT / 'core_ws/src/h12_bringup/rviz/pegboard.rviz').read_text())
        manager = config['Visualization Manager']
        self.assertEqual(manager['Global Options']['Fixed Frame'], 'pelvis')
        self.assertFalse(any('Goal' in tool['Class'] or 'Publish' in tool['Class'] for tool in manager['Tools']))
        camera = next(d for d in manager['Displays'] if d['Class'] == 'rviz_default_plugins/Camera')
        self.assertEqual(camera['Topic']['Value'], '/realsense/head/color/image_raw')
        self.assertEqual(camera['Topic']['Reliability Policy'], 'Best Effort')
        markers = next(d for d in manager['Displays'] if d['Class'] == 'rviz_default_plugins/MarkerArray')
        self.assertEqual(markers['Topic']['Value'], '/perception/pegboard/markers')

    def test_domain_zero_guard_and_help(self):
        env = dict(os.environ, ROS_DOMAIN_ID='0')
        result = subprocess.run([str(ROOT / 'scripts/run_rviz.sh')], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('real robot bus', result.stderr)
        result = subprocess.run([str(ROOT / 'scripts/run_rviz.sh'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Starts only RViz', result.stdout)

    def test_clean_wrapper_check_ignores_foreign_python_and_conda(self):
        env = dict(os.environ, ROS_DOMAIN_ID='1', PYTHONHOME='/bogus', PYTHONPATH='/bogus',
                   CONDA_PREFIX='/bogus', QT_PLUGIN_PATH='/bogus')
        result = subprocess.run([str(ROOT / 'scripts/run_rviz.sh'), '--check'], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fixed_frame=pelvis; use_sim_time=true', result.stdout)

    def test_exec_is_only_cpp_rviz_not_bringup(self):
        with patch.object(sys, 'argv', ['run_rviz']), patch.dict(os.environ, {'ROS_DOMAIN_ID': '1', 'DISPLAY': ':0'}), \
                patch.object(os, 'execve') as execute:
            rviz.main()
        _, argv, env = execute.call_args.args
        self.assertIn('/opt/ros/humble/lib/rviz2/rviz2', argv)
        self.assertIn('use_sim_time:=true', argv)
        self.assertNotIn('launch', argv)
        self.assertEqual(env['QT_QPA_PLATFORM'], 'xcb')


if __name__ == '__main__':
    unittest.main()
