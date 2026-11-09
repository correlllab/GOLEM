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

    def test_full_robot_recipe_is_derived_from_existing_main(self):
        directory = ROOT / 'core_ws/src/h12_bringup/rviz'
        base = yaml.safe_load((directory / 'main.rviz').read_text())['Visualization Manager']
        manager = yaml.safe_load((directory / 'pegboard.rviz').read_text())['Visualization Manager']
        robot = next(d for d in manager['Displays'] if d['Class'] == 'rviz_default_plugins/RobotModel')
        original = next(d for d in base['Displays'] if d['Class'] == robot['Class'])
        for key in ('Enabled', 'Value', 'Visual Enabled', 'Collision Enabled', 'Alpha',
                    'Description Source', 'Description Topic', 'TF Prefix', 'Mass Properties'):
            self.assertEqual(robot[key], original[key], key)
        self.assertTrue(robot['Links']['All Links Enabled'])
        grid = next(d for d in manager['Displays'] if d['Class'] == 'rviz_default_plugins/Grid')
        self.assertEqual(grid['Reference Frame'], '<Fixed Frame>')
        self.assertAlmostEqual(grid['Offset']['Z'], -0.97)
        tf = next(d for d in manager['Displays'] if d['Class'] == 'rviz_default_plugins/TF')
        self.assertTrue(tf['Enabled'])
        for frame in ('pelvis', 'head_camera_link', 'head_color_optical_frame',
                      'left_grasp_frame', 'right_grasp_frame'):
            self.assertTrue(tf['Frames'][frame]['Value'])
        view = manager['Views']['Current']
        self.assertEqual(view['Class'], base['Views']['Current']['Class'])
        self.assertEqual(view['Target Frame'], '<Fixed Frame>')
        self.assertGreaterEqual(view['Distance'], 2.5)
        self.assertLess(abs(view['Focal Point']['Z']), 0.3)
        images = {d['Name']: d for d in manager['Displays']
                  if d['Class'] == 'rviz_default_plugins/Image'}
        for name, topic in (('HeadColorImage', '/realsense/head/color/image_raw'),
                            ('Annotated RGB', '/perception/pegboard/annotated')):
            self.assertTrue(images[name]['Enabled'])
            self.assertEqual(images[name]['Topic']['Value'], topic)
            self.assertEqual(images[name]['Topic']['Reliability Policy'], 'Best Effort')
            self.assertEqual(images[name]['Topic']['Durability Policy'], 'Volatile')
        self.assertIn('/aligned_depth_to_color/', images['HeadDepthImage']['Topic']['Value'])

    def test_installed_robot_description_resolves_real_mesh_assets(self):
        share = ROOT / 'core_ws/install/h12_ros2_model/share/h12_ros2_model'
        robot = ET.parse(share / 'assets/h1_2/h1_2_magpie_ros.urdf').getroot()
        children = {joint.find('child').get('link') for joint in robot.findall('joint')}
        self.assertEqual({link.get('name') for link in robot.findall('link')} - children, {'pelvis'})
        meshes = robot.findall('.//mesh')
        self.assertTrue(meshes)
        for mesh in meshes:
            uri = mesh.get('filename')
            prefix = 'package://h12_ros2_model/'
            self.assertTrue(uri.startswith(prefix), uri)
            path = share / uri.removeprefix(prefix)
            self.assertTrue(path.is_file(), path)
            with path.open('rb') as stream:
                self.assertFalse(stream.read(80).startswith(b'version https://git-lfs.github.com/spec/'), path)

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
                                cwd='/tmp/kilo', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('fixed_frame=pelvis; use_sim_time=true', result.stdout)
        self.assertIn(f'config={ROOT}/core_ws/src/h12_bringup/rviz/pegboard.rviz;', result.stdout)

    def test_exec_is_only_cpp_rviz_not_bringup(self):
        with patch.object(sys, 'argv', ['run_rviz']), patch.dict(os.environ, {'ROS_DOMAIN_ID': '1', 'DISPLAY': ':0'}), \
                patch.object(os, 'execve') as execute:
            rviz.main()
        _, argv, env = execute.call_args.args
        self.assertIn('/opt/ros/humble/lib/rviz2/rviz2', argv)
        self.assertIn('use_sim_time:=true', argv)
        self.assertEqual(argv[argv.index('-d') + 1],
                         str(ROOT / 'core_ws/src/h12_bringup/rviz/pegboard.rviz'))
        self.assertNotIn('launch', argv)
        self.assertEqual(env['QT_QPA_PLATFORM'], 'xcb')

    def test_existing_main_config_override_is_preserved_without_exec(self):
        config = ROOT / 'core_ws/src/h12_bringup/rviz/main.rviz'
        with patch.object(sys, 'argv', ['run_rviz', '--config', str(config)]), \
                patch.dict(os.environ, {'ROS_DOMAIN_ID': '1', 'DISPLAY': ':0'}), \
                patch.object(os, 'execve') as execute:
            rviz.main()
        argv = execute.call_args.args[1]
        self.assertEqual(argv[argv.index('-d') + 1], str(config))


if __name__ == '__main__':
    unittest.main()
