"""Offline entrypoint safety and composition; never execute the simulator."""

import json
import io
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'core_ws/scripts'))
import run_simulation as runtime
import setup_simulator as simulator
import run_ros as ros_runtime


class EntrypointTests(unittest.TestCase):
    def test_optional_recording_options_forward_without_changing_defaults(self):
        options = ['--record', '/tmp/kilo/trial clip.webm', '--record-fps', '20']
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, 'argv', ['run_simulation.py', *options]), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve') as execute:
            runtime.main()
        self.assertEqual(execute.call_args.args[1][-4:],
                         ['--record', '/tmp/kilo/trial clip.webm', '--record-fps', '20.0'])
        defaults = runtime.parser().parse_args([])
        self.assertIsNone(defaults.record)
        self.assertIsNone(defaults.record_fps)

    def test_help_without_dependencies_and_from_other_cwd(self):
        for script in ('setup_environment.sh', 'run_simulation.sh', 'run_experiment.sh', 'run_ros.sh'):
            result = subprocess.run([str(ROOT / 'scripts' / script), '--help'], cwd='/tmp/kilo',
                                    env={'HOME': str(Path.home()), 'PATH': '/invalid',
                                         'PYTHONHOME': '/invalid', 'PYTHONPATH': '/invalid'},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_zero_and_invalid_domains_rejected_before_python(self):
        for script in ('run_simulation.sh', 'run_experiment.sh', 'run_ros.sh'):
            for domain in ('0', '00', '000', '-1', '233', 'x', '1; touch /invalid'):
                result = subprocess.run([str(ROOT / 'scripts' / script), 'topic', 'list'],
                                        env={'HOME': str(Path.home()), 'PATH': '/invalid',
                                             'ROS_DOMAIN_ID': domain}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('real robot bus', result.stderr)

    def test_python_also_rejects_domain_before_check(self):
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '0'}), \
                patch.object(sys, 'argv', ['run_simulation.py']), patch.object(simulator, 'check') as check:
            with self.assertRaisesRegex(RuntimeError, 'real robot bus'):
                runtime.main()
        check.assert_not_called()

    def test_default_task_domain_and_ground_truth_are_safe(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', ['run_simulation.py']), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve') as execute:
            runtime.main()
        _, command, env = execute.call_args.args
        self.assertEqual(command[-2:], ['--task', 'PegInHole'])
        self.assertNotIn('--lowerbody', command)
        self.assertEqual(env['ROS_DOMAIN_ID'], '1')
        self.assertEqual(env['GOLEM_SIM_ODOM'], '0')
        self.assertNotIn('GOLEM_SIM_TETHER_ROT_STIFFNESS', env)
        self.assertNotIn('GOLEM_SIM_TETHER_ROT_BODY', env)
        self.assertEqual({key: env[key] for key in runtime.PEGBOARD_CAMERA_DEFAULTS},
                         {'GOLEM_CAMERA_SCALE': '1', 'GOLEM_CAMERA_DEPTH': '0',
                           'GOLEM_CAMERA_POINTCLOUD': '0'})

    def test_omitted_policy_reports_legacy_spawn(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', ['run_simulation.py']), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve'), patch('sys.stdout', new_callable=io.StringIO) as output:
            runtime.main()
        self.assertIn('legacy baked spawn (--lowerbody omitted); no policy selected', output.getvalue())

    def test_manipulation_hoist_selects_existing_fixture_without_new_simulator_flags(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, 'argv', ['run_simulation.py', '--manipulation-hoist']), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve') as execute, \
                patch('sys.stdout', new_callable=io.StringIO) as output:
            runtime.main()
        _, command, env = execute.call_args.args
        self.assertEqual(command[-2:], ['--task', 'PegInHole'])
        self.assertEqual(env['GOLEM_SIM_TETHER_ROT_STIFFNESS'], '2000')
        self.assertEqual(env['GOLEM_SIM_TETHER_ROT_BODY'], 'pelvis')
        self.assertIn('NOT free-standing balance', output.getvalue())

    def test_selected_policies_forward_unchanged_without_controller_launch(self):
        for policy in ('fame_finetune', 'almi_manip_stand'):
            with self.subTest(policy=policy), patch.dict(os.environ, {}, clear=True), \
                    patch.object(sys, 'argv', ['run_simulation.py', '--lowerbody', policy]), \
                    patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                    patch.object(os, 'execve') as execute, \
                    patch('sys.stdout', new_callable=io.StringIO) as output:
                runtime.main()
            command = execute.call_args.args[1]
            self.assertEqual(command[-4:], ['--task', 'PegInHole', '--lowerbody', policy])
            self.assertIn(str(ROOT / 'h1_robocasa/h12_mujoco.py'), command)
            self.assertIn(f'lowerbody:={policy}', output.getvalue())
            self.assertIn('does not start or confirm controllers', output.getvalue())

    def test_invalid_policy_rejected_before_prerequisites_even_with_mocked_check(self):
        for policy in ('unknown', '/tmp/policy.yaml', 'almi_manip_stand; touch /invalid'):
            with self.subTest(policy=policy), \
                    patch.object(sys, 'argv', ['run_simulation.py', '--lowerbody', policy]), \
                    patch.object(simulator, 'check') as check, \
                    patch.object(Path, 'is_file') as prerequisite, patch.object(os, 'execve') as execute, \
                    patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit) as error:
                    runtime.main()
                self.assertEqual(error.exception.code, 2)
            check.assert_not_called()
            prerequisite.assert_not_called()
            execute.assert_not_called()

    def test_scene_options_and_policy_compose_without_changing_values(self):
        options = ['--layout', '2', '--style', '3', '--seed', '42', '--peg', 'cylinder_2',
                   '--source-hole', 'cylinder_1', '--lowerbody', 'almi_manip_stand']
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(sys, 'argv', ['run_simulation.py', *options]), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve') as execute:
            runtime.main()
        self.assertEqual(execute.call_args.args[1][-len(options):], options)

    def test_guide_anchors_copy_targets_and_simulator_policy_examples(self):
        class GuideParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.ids = []
                self.targets = []
                self.code = []
                self.current_code = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs:
                    self.ids.append(attrs['id'])
                if attrs.get('href', '').startswith('#'):
                    self.targets.append(attrs['href'][1:])
                if 'data-copy' in attrs:
                    self.targets.append(attrs['data-copy'])
                if tag == 'code':
                    self.current_code = []

            def handle_data(self, data):
                if self.current_code is not None:
                    self.current_code.append(data)

            def handle_endtag(self, tag):
                if tag == 'code' and self.current_code is not None:
                    self.code.append(''.join(self.current_code))
                    self.current_code = None

        guide = GuideParser()
        guide.feed((ROOT / 'docs/h12_simulation_guide.html').read_text())
        self.assertEqual(len(guide.ids), len(set(guide.ids)))
        self.assertTrue(guide.targets)
        self.assertFalse(set(guide.targets) - set(guide.ids))
        commands = [code for code in guide.code if './scripts/run_simulation.sh --task' in code]
        self.assertTrue(commands)
        for command in commands:
            self.assertIn('--lowerbody ', command)
        primary = './scripts/run_simulation.sh --task PegInHole --lowerbody almi_manip_stand'
        self.assertTrue(any(command.startswith(primary) and '--manipulation-hoist' in command
                            for command in commands))

    def test_pegboard_camera_defaults_preserve_explicit_overrides(self):
        overrides = {'GOLEM_CAMERA_SCALE': '0.5', 'GOLEM_CAMERA_DEPTH': '1',
                     'GOLEM_CAMERA_POINTCLOUD': '1'}
        with patch.dict(os.environ, overrides, clear=True), \
                patch.object(sys, 'argv', ['run_simulation.py']), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve') as execute:
            runtime.main()
        env = execute.call_args.args[2]
        for key, value in overrides.items():
            self.assertEqual(env[key], value)

    def test_other_tasks_keep_existing_camera_defaults(self):
        for task in ('Battery', 'OpenFridge'):
            with self.subTest(task=task), patch.dict(os.environ, {}, clear=True), \
                    patch.object(sys, 'argv', ['run_simulation.py', '--task', task]), \
                    patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                    patch.object(os, 'execve') as execute:
                runtime.main()
            env = execute.call_args.args[2]
            for key in runtime.PEGBOARD_CAMERA_DEFAULTS:
                self.assertNotIn(key, env)

    @unittest.skipUnless(shutil.which('zsh'), 'requires zsh')
    def test_zsh_caller_needs_no_shell_switch(self):
        result = subprocess.run([shutil.which('zsh'), '-f', '-c', 'exec "$1" --help',
                                 'test', str(ROOT / 'scripts/run_simulation.sh')],
                                cwd='/tmp/kilo', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PegInHole', result.stdout)

    def test_exec_preserves_arguments_without_eval_and_uses_local_overlays(self):
        task = 'PegInHole; $(touch /invalid)'
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '2'}, clear=True), \
                patch.object(sys, 'argv', ['run_simulation.py', '--task', task, '--seed', '42',
                                          '--headless', '--timing']), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check') as check, \
                patch.object(os, 'execve') as execute:
            runtime.main()
        check.assert_called_once_with(True)
        executable, command, env = execute.call_args.args
        self.assertEqual(executable, '/bin/bash')
        self.assertEqual(command[-6:], ['--task', task, '--seed', '42', '--headless', '--timing'])
        self.assertIn(str(ROOT / 'core_ws/install/local_setup.bash'), command)
        self.assertNotIn('eval', command[4])
        self.assertEqual(env['GOLEM_SIM_ODOM'], '0')
        self.assertTrue(env['PYTHONPATH'].startswith(str(ROOT / 'unitree_sdk2_python')))

    def test_missing_prerequisites_never_import_bridges(self):
        with patch.object(sys, 'argv', ['run_simulation.py']), \
                patch.object(Path, 'is_file', return_value=False), patch.object(simulator, 'check') as check:
            with self.assertRaisesRegex(RuntimeError, 'setup_environment.sh simulator'):
                runtime.main()
        check.assert_not_called()

    def test_clean_child_preserves_only_selected_gui_and_optin_settings(self):
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            root = Path(directory)
            (root / 'environment').mkdir()
            shutil.copy2(ROOT / 'scripts/run_simulation.sh', root)
            shutil.copy2(HERE / 'clean_shell.sh', root / 'environment')
            (root / 'environment/run_simulation.py').write_text(
                'import json, os, site, sys; print(json.dumps({"env":dict(os.environ),'
                '"usersite":site.ENABLE_USER_SITE,"argv":sys.argv[1:]}))')
            env = os.environ.copy()
            env.update(CONDA_PREFIX='/foreign/conda', PYTHONPATH='/foreign/python',
                       LD_LIBRARY_PATH='/foreign/lib', AMENT_PREFIX_PATH='/foreign/overlay',
                       ROS_DOMAIN_ID='3', DISPLAY=':99', XAUTHORITY='/tmp/test-xauth',
                       WAYLAND_DISPLAY='wayland-7', XDG_RUNTIME_DIR='/tmp/test-runtime',
                       GOLEM_SIM_ODOM='1', GOLEM_CAMERA_SCALE='1', GOLEM_CAMERA_DEPTH='0',
                       GOLEM_CAMERA_POINTCLOUD='0', CYCLONEDDS_URI='/foreign/dds')
            result = subprocess.run([str(root / 'run_simulation.sh'), '--seed', '5'],
                                    env=env, capture_output=True, text=True, cwd='/tmp/kilo')
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(result.stdout)
            self.assertFalse(actual['usersite'])
            self.assertNotIn('/foreign/', result.stdout)
            for key in ('DISPLAY', 'XAUTHORITY', 'WAYLAND_DISPLAY', 'XDG_RUNTIME_DIR',
                        'GOLEM_SIM_ODOM', 'GOLEM_CAMERA_SCALE',
                        'GOLEM_CAMERA_DEPTH', 'GOLEM_CAMERA_POINTCLOUD'):
                self.assertEqual(actual['env'][key], env[key])
            self.assertEqual(actual['env']['ROS_DOMAIN_ID'], '3')
            self.assertEqual(actual['argv'], ['--seed', '5'])

    def test_setup_rejects_ros_only_debs_on_simulator(self):
        result = subprocess.run([str(ROOT / 'scripts/setup_environment.sh'), 'simulator', '--local-ros-debs'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires ros setup', result.stderr)

    def test_selected_peg_forwarding_is_explicit_and_task_specific(self):
        with patch.object(sys, 'argv', ['run_simulation.py', '--peg', 'cylinder_2',
                                       '--source-hole', 'cylinder_1']), \
                patch.dict(os.environ, {'ROS_DOMAIN_ID': '1'}), \
                patch.object(Path, 'is_file', return_value=True), patch.object(simulator, 'check'), \
                patch.object(os, 'execve') as execute:
            runtime.main()
        self.assertEqual(execute.call_args.args[1][-4:],
                         ['--peg', 'cylinder_2', '--source-hole', 'cylinder_1'])
        with patch.object(sys, 'argv', ['run_simulation.py', '--task', 'Battery', '--peg', 'cylinder_1']):
            with self.assertRaisesRegex(RuntimeError, 'require --task PegInHole'):
                runtime.main()

    def test_experiment_command_is_clean_explicit_and_has_no_hidden_build(self):
        sys.path.insert(0, str(ROOT / 'core_ws/scripts'))
        import build_ros_workspace as workspace
        with patch.object(sys, 'argv', ['run_ros.py', '--experiment']), \
                 patch.dict(os.environ, {'ROS_DOMAIN_ID': '1'}, clear=True), \
                 patch.object(workspace, 'require_venv'), \
                 patch.object(ros_runtime, 'acquire_experiment_lock') as lock, \
                 patch.object(Path, 'is_file', return_value=True), patch.object(os, 'execve') as execute:
            ros_runtime.main()
        _, command, env = execute.call_args.args
        self.assertEqual(command[-3:], ['launch', 'h12_bringup', 'h12_native_experiment.launch.py'])
        self.assertEqual(env['OMP_NUM_THREADS'], '1')
        self.assertEqual(env['MKL_NUM_THREADS'], '1')
        self.assertTrue(env['GOLEM_CONTROLLER_PYTHON'].endswith('/environments/controller/.venv/bin/python'))
        self.assertTrue(env['GOLEM_MARKER_CATALOG'].endswith('/manipulation_catalog.json'))
        self.assertNotIn('start_position_verified:=true', command)
        lock.assert_called_once_with('1')

    def test_experiment_lock_rejects_duplicate_and_releases_on_close(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(ros_runtime, 'ROOT', Path(directory)):
            first = ros_runtime.acquire_experiment_lock('1')
            try:
                self.assertTrue(os.get_inheritable(first))
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    ros_runtime.acquire_experiment_lock('1')
                other_domain = ros_runtime.acquire_experiment_lock('2')
                os.close(other_domain)
            finally:
                os.close(first)
            replacement = ros_runtime.acquire_experiment_lock('1')
            os.close(replacement)

    def test_experiment_forwards_standard_image_transport(self):
        import build_ros_workspace as workspace
        for transport in ('raw', 'compressed'):
            with self.subTest(transport=transport), \
                    patch.object(sys, 'argv', ['run_ros.py', '--experiment',
                                              f'image_transport:={transport}']), \
                    patch.dict(os.environ, {'ROS_DOMAIN_ID': '41'}, clear=True), \
                    patch.object(workspace, 'require_venv'), \
                    patch.object(ros_runtime, 'acquire_experiment_lock'), \
                    patch.object(Path, 'is_file', return_value=True), \
                    patch.object(os, 'execve') as execute:
                ros_runtime.main()
            self.assertEqual(execute.call_args.args[1][-1], f'image_transport:={transport}')

    def test_experiment_rejects_unknown_image_transport_before_launch(self):
        import build_ros_workspace as workspace
        with patch.object(sys, 'argv', ['run_ros.py', '--experiment', 'image_transport:=invalid']), \
                patch.object(workspace, 'require_venv'), \
                patch.object(Path, 'is_file', return_value=True), \
                patch.object(os, 'execve') as execute:
            with self.assertRaisesRegex(RuntimeError, 'raw or compressed'):
                ros_runtime.main()
        execute.assert_not_called()

    def test_experiment_forwards_upper_controller_and_explicit_policy(self):
        import build_ros_workspace as workspace
        for upper in ('frame', 'counter'):
            for policy in ('fame_finetune', 'almi_manip_stand'):
                arguments = [f'upper_controller:={upper}', f'lowerbody:={policy}']
                with self.subTest(upper=upper, policy=policy), \
                        patch.object(sys, 'argv', ['run_ros.py', '--experiment', *arguments]), \
                        patch.dict(os.environ, {'ROS_DOMAIN_ID': '44'}, clear=True), \
                        patch.object(workspace, 'require_venv'), \
                        patch.object(ros_runtime, 'acquire_experiment_lock'), \
                        patch.object(Path, 'is_file', return_value=True), \
                        patch.object(os, 'execve') as execute, \
                        patch('sys.stdout', new_callable=io.StringIO) as output:
                    ros_runtime.main()
                self.assertEqual(execute.call_args.args[1][-2:], arguments)
                self.assertIn(f'Upper controller: {upper}', output.getvalue())

    def test_experiment_rejects_bad_upper_controller_before_launch_or_lock(self):
        import build_ros_workspace as workspace
        for upper in ('', 'robust3c', 'Frame', 'counter; touch /invalid'):
            with self.subTest(upper=upper), \
                    patch.object(sys, 'argv', ['run_ros.py', '--experiment', f'upper_controller:={upper}']), \
                    patch.object(workspace, 'require_venv'), \
                    patch.object(Path, 'is_file', return_value=True), \
                    patch.object(ros_runtime, 'acquire_experiment_lock') as lock, \
                    patch.object(os, 'execve') as execute:
                with self.assertRaisesRegex(RuntimeError, 'upper_controller must be frame or counter'):
                    ros_runtime.main()
            execute.assert_not_called()
            lock.assert_not_called()

    def test_comparison_order_forwards_as_one_argument_and_rejects_invalid_csv(self):
        import build_ros_workspace as workspace
        for order in ('frame,counter', 'counter,frame,frame,counter',
                      '', 'frame,blue', 'frame,', 'counter; touch /invalid'):
            valid = order in ('frame,counter', 'counter,frame,frame,counter')
            arguments = ['upper_controller:=counter', f'comparison_order:={order}']
            with self.subTest(order=order), \
                    patch.object(sys, 'argv', ['run_ros.py', '--experiment', *arguments]), \
                    patch.dict(os.environ, {'ROS_DOMAIN_ID': '45'}, clear=True), \
                    patch.object(workspace, 'require_venv'), \
                    patch.object(Path, 'is_file', return_value=True), \
                    patch.object(ros_runtime, 'acquire_experiment_lock') as lock, \
                    patch.object(os, 'execve') as execute:
                if valid:
                    ros_runtime.main()
                    self.assertEqual(execute.call_args.args[1][-2:], arguments)
                    self.assertNotIn('eval', execute.call_args.args[1][4])
                else:
                    with self.assertRaisesRegex(RuntimeError, 'comparison_order'):
                        ros_runtime.main()
                    lock.assert_not_called()
                    execute.assert_not_called()

    def test_generic_param_cli_preserves_full_standard_arguments(self):
        import build_ros_workspace as workspace
        arguments = ['param', 'set', '/frame_task_server', 'comparison_order',
                     'counter,frame,frame,counter']
        with patch.object(sys, 'argv', ['run_ros.py', *arguments]), \
                patch.dict(os.environ, {'ROS_DOMAIN_ID': '45'}, clear=True), \
                patch.object(workspace, 'require_venv'), \
                patch.object(Path, 'is_file', return_value=True), \
                patch.object(ros_runtime, 'acquire_experiment_lock') as lock, \
                patch.object(os, 'execve') as execute:
            ros_runtime.main()
        self.assertEqual(execute.call_args.args[1][-len(arguments):], arguments)
        lock.assert_not_called()

    def test_experiment_rejects_unowned_launch_parameters(self):
        import build_ros_workspace as workspace
        for argument in ('controller_python:=/foreign/python', 'use_sim_time:=false', '--show-args'):
            with patch.object(sys, 'argv', ['run_ros.py', '--experiment', argument]), \
                    patch.object(workspace, 'require_venv'), \
                    patch.object(Path, 'is_file', return_value=True), \
                    patch.object(os, 'execve') as execute:
                with self.assertRaises((RuntimeError, SystemExit)):
                    ros_runtime.main()
            execute.assert_not_called()

    def test_controller_and_cpu_torch_metadata_are_scoped(self):
        ros = (ROOT / 'environments/ros/pyproject.toml').read_text()
        for setting in ('"torch==2.6.0+cpu"', 'torch = { index = "pytorch-cpu" }',
                        'url = "https://download.pytorch.org/whl/cpu"', 'explicit = true'):
            self.assertIn(setting, ros)
        self.assertNotIn('torchvision', ros)
        self.assertNotIn('name = "nvidia-', (ROOT / 'environments/ros/uv.lock').read_text())
        deps = (ROOT / 'environments/controller/pyproject.toml').read_text()
        for dependency in ('numpy==2.2.6', 'pin==3.9.0', 'pin-pink==4.1.0',
                           'cmeel-urdfdom==4.0.1', 'cmeel-tinyxml2==10.0.0', 'crocoddyl==3.2.0'):
            self.assertIn(dependency, deps)
        for name in ('torch==', 'opencv-python==', 'pynput'):
            self.assertNotIn(name, deps)
