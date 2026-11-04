"""From core_ws run /usr/bin/python3 -B -m unittest discover -s scripts/tests."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
SCRIPT = HERE / 'build_ros_workspace.sh'
spec = importlib.util.spec_from_file_location('build_ros_workspace', HERE / 'build_ros_workspace.py')
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


class WorkspaceBuildTests(unittest.TestCase):
    def test_standard_layout_and_discovery_boundary(self):
        self.assertEqual(workspace.WORKSPACE, HERE.parent)
        self.assertEqual(workspace.ROOT, HERE.parent.parent)
        self.assertEqual(workspace.ROS_ENV, workspace.WORKSPACE / 'environments')
        self.assertEqual(workspace.VENV, workspace.ROS_ENV / '.venv')
        self.assertEqual(workspace.INSTALL, workspace.WORKSPACE / 'install')
        self.assertEqual(workspace.environment()['UV_CACHE_DIR'],
                         str(workspace.WORKSPACE / '.cache/uv'))
        self.assertTrue((workspace.ROS_ENV / 'COLCON_IGNORE').is_file())
        self.assertFalse(workspace.ROS_ENV.is_relative_to(workspace.WORKSPACE / 'src'))
        self.assertFalse((workspace.ROS_ENV / 'ros').exists())
        self.assertFalse((workspace.ROOT / 'pyproject.toml').exists())
        for location in ('build/', 'install/', 'log/', 'environments/.venv/', '.cache/'):
            result = subprocess.run(
                ['git', 'check-ignore', str(workspace.WORKSPACE / location / 'probe')],
                cwd=workspace.ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, (location, result.stderr))

    def test_exact_allowlist_and_workspace_dependencies(self):
        expected = {
            'estop', 'h12_safety_layer', 'custom_ros_messages', 'magpie_msgs',
            'unitree_hg', 'h12_ros2_model', 'h12_ros2_controller',
            'magpie_control', 'h12_lowerbody_rl', 'h12_skills',
        }
        self.assertEqual(set(workspace.MANIPULATION), expected)
        all_names = {ET.parse(path).getroot().findtext('name')
                      for path in (workspace.WORKSPACE / 'src').glob('**/package.xml')}
        for name in expected:
            manifest = ET.parse(workspace.package_path(name) / 'package.xml').getroot()
            self.assertEqual(manifest.findtext('name'), name)
            deps = {element.text for element in manifest
                    if element.tag in ('depend', 'build_depend', 'build_export_depend', 'exec_depend')}
            self.assertLessEqual(deps & all_names, expected)

    def test_build_interpreter_paths_and_no_recursive_closure(self):
        command = list(map(str, workspace.build_command('manipulation')))
        self.assertEqual(command[:3], [str(workspace.PYTHON), '-m', 'colcon'])
        self.assertNotIn('--packages-up-to', command)
        self.assertNotIn('--base-paths', command)
        selected = command[command.index('--packages-select') + 1:command.index('--event-handlers')]
        self.assertEqual(selected, list(workspace.MANIPULATION))
        paths = command[command.index('--paths') + 1:command.index('--packages-select')]
        self.assertEqual(paths, list(map(str, map(workspace.package_path, workspace.MANIPULATION))))
        self.assertIn(f'-DPython3_EXECUTABLE={workspace.PYTHON}', command)
        self.assertIn(f'-DPYTHON_EXECUTABLE={workspace.PYTHON}', command)
        self.assertEqual(command[command.index('--executor') + 1], 'sequential')
        for location in ('build', 'install', 'log'):
            self.assertIn(str(workspace.WORKSPACE / location), command)
        env = workspace.environment()
        self.assertEqual(env['COLCON_DEFAULTS_FILE'], '/dev/null')
        self.assertEqual(env['COLCON_HOME'], str(workspace.VENV / 'colcon-home'))
        self.assertFalse(Path(env['UV_CACHE_DIR']).is_relative_to(workspace.VENV))
        self.assertEqual(env['UV_PROJECT_ENVIRONMENT'], str(workspace.VENV.resolve()))
        self.assertEqual(env['UV_PYTHON_DOWNLOADS'], 'never')
        self.assertEqual(env['UV_NO_CONFIG'], '1')

    def test_build_argument_protection(self):
        for arguments in (
            ['build', '--packages-select', 'model_server'],
            ['build', '--cmake-args', '-DPython3_EXECUTABLE=/bad/python'],
            ['build', '--profile', 'all'], ['build', '--profile', 'lean'],
            ['build', '--compile-jobs', '0'], ['build', '--compile-jobs', '65'],
            ['build', '--jobs', '2'], ['build', '--', '--packages-up-to', 'h12_bringup'],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    workspace.parser().parse_args(arguments)

    def test_profile_defaults_and_compile_jobs_are_explicit(self):
        default = workspace.parser().parse_args(['build'])
        self.assertEqual(default.profile, 'manipulation')
        self.assertEqual(default.compile_jobs, 2)
        with patch.object(sys, 'argv', ['build_ros_workspace.py', 'build', '--profile', 'core',
                                       '--compile-jobs', '4']), \
                patch.object(workspace, 'prerequisites'), patch.object(workspace, 'check'), \
                patch.object(workspace, 'run') as run:
            workspace.main()
        self.assertEqual(run.call_args.kwargs['env']['CMAKE_BUILD_PARALLEL_LEVEL'], '4')
        self.assertEqual(run.call_args.kwargs['env']['MAKEFLAGS'], '-j4')

    def test_shell_entrypoint_has_interpreter_and_is_executable(self):
        self.assertEqual(SCRIPT.read_text().splitlines()[0], '#!/bin/bash')
        self.assertTrue(os.access(SCRIPT, os.X_OK), SCRIPT)

    def test_setup_explicit_local_interpreters_and_sdk_no_deps(self):
        with patch.object(sys, 'argv', ['build_ros_workspace.py', 'setup']), \
                patch.object(workspace, 'prerequisites'), patch.object(workspace, 'check'), \
                patch.object(workspace, 'require_venv'), patch.object(workspace, 'run') as run, \
                patch.dict(os.environ, {'GOLEM_NATIVE_UV': '/usr/bin/true'}):
            workspace.main()
        commands = [list(map(str, call.args[0])) for call in run.call_args_list]
        if not workspace.VENV.exists():
            self.assertEqual(commands[0][1:5], ['venv', '--python', '/usr/bin/python3', '--system-site-packages'])
        installs = [cmd for cmd in commands if 'install' in cmd]
        self.assertEqual(len(installs), 1)
        self.assertEqual(installs[0][installs[0].index('--python') + 1], str(workspace.PYTHON))
        self.assertNotIn('--system', installs[0])
        self.assertIn('--no-deps', installs[0])
        self.assertIn('--no-build-isolation', installs[0])
        self.assertEqual(installs[0][-1], str(workspace.ROOT / 'unitree_sdk2_python'))
        syncs = [call for call in run.call_args_list if 'sync' in call.args[0]]
        self.assertEqual(len(syncs), 1)
        self.assertEqual(list(map(str, syncs[0].args[0])), [
            '/usr/bin/true', 'sync', '--locked', '--inexact', '--no-build',
            '--project', str(workspace.ROS_ENV), '--python', '/usr/bin/python3',
        ])
        self.assertEqual(syncs[0].kwargs['env']['UV_PROJECT_ENVIRONMENT'], str(workspace.VENV))
        self.assertLess(commands.index(list(map(str, syncs[0].args[0]))), commands.index(installs[0]))
        self.assertFalse(any('apt-get' in ' '.join(command) for command in commands))

    def test_environment_metadata_replaces_script_requirements(self):
        self.assertTrue((workspace.ROS_ENV / 'pyproject.toml').is_file())
        self.assertTrue((workspace.ROS_ENV / 'uv.lock').is_file())
        self.assertTrue((workspace.ROS_ENV / 'apt-packages.txt').is_file())
        self.assertFalse((HERE / 'apt-packages.txt').exists())
        self.assertFalse((HERE / 'requirements-ros.txt').exists())
        self.assertNotIn('requirements-ros.txt', (HERE / 'build_ros_workspace.py').read_text())

    def test_failed_locked_sync_stops_before_sdk_install_or_checks(self):
        with patch.object(sys, 'argv', ['build_ros_workspace.py', 'setup']), \
                patch.object(workspace, 'prerequisites'), patch.object(workspace, 'check') as check, \
                patch.object(workspace, 'require_venv'), \
                patch.dict(os.environ, {'GOLEM_NATIVE_UV': '/usr/bin/true'}):
            def stub(command, **kwargs):
                if 'sync' in command:
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(workspace, 'run', side_effect=stub) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    workspace.main()
            self.assertFalse(any('install' in call.args[0] for call in run.call_args_list))
            check.assert_not_called()

    @unittest.skipUnless(shutil.which('uv'), 'requires uv')
    def test_lock_is_current_and_stale_project_fails_closed_offline(self):
        env = workspace.environment()
        command = [shutil.which('uv'), 'lock', '--python', '/usr/bin/python3', '--check', '--offline']
        lock = (workspace.ROS_ENV / 'uv.lock').read_bytes()
        result = subprocess.run([*command, '--project', str(workspace.ROS_ENV)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory(prefix='golem-native-lock-') as directory:
            project = Path(directory)
            manifest = (workspace.ROS_ENV / 'pyproject.toml').read_text()
            (project / 'pyproject.toml').write_text(manifest.replace(
                'dependencies = [', 'dependencies = [\n    "idna==3.10",', 1))
            (project / 'uv.lock').write_bytes(lock)
            env['UV_PROJECT_ENVIRONMENT'] = str(project / 'venv')
            result = subprocess.run([
                shutil.which('uv'), 'sync', '--project', str(project), '--python', '/usr/bin/python3',
                '--locked', '--inexact', '--no-build', '--offline',
            ], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertEqual((project / 'uv.lock').read_bytes(), lock)
            self.assertFalse(list((project / 'venv').glob('lib/python*/site-packages/*.dist-info')))

    def test_build_failure_never_checks_or_reports_success(self):
        with patch.object(sys, 'argv', ['build_ros_workspace.py', 'build', '--profile', 'core']), \
                patch.object(workspace, 'prerequisites'), patch.object(workspace, 'check') as check, \
                patch.object(workspace, 'run', side_effect=subprocess.CalledProcessError(7, 'colcon')):
            with self.assertRaises(subprocess.CalledProcessError):
                workspace.main()
        check.assert_called_once_with()

    def test_missing_manipulation_generator_fails_before_build(self):
        with patch.object(sys, 'argv', ['build_ros_workspace.py', 'build']), \
                patch.object(workspace, 'prerequisites'), patch.object(workspace, 'check'), \
                patch.object(workspace, 'manipulation_prerequisites',
                             side_effect=subprocess.CalledProcessError(1, 'probe')), \
                patch.object(workspace, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'ros-humble-rosidl-generator-dds-idl'):
                workspace.main()
        run.assert_not_called()

    def test_rootless_ros_debs_only_extract_allowlisted_archives(self):
        with tempfile.TemporaryDirectory(prefix='golem-native-debs-') as directory:
            venv = Path(directory)
            commands = []

            def stub(command, **kwargs):
                commands.append(list(map(str, command)))
                if command[:2] == ['/usr/bin/apt-get', 'download']:
                    (Path(kwargs['cwd']) / f'{command[2]}_test_amd64.deb').touch()

            with patch.object(workspace, 'VENV', venv), patch.object(workspace, 'run', side_effect=stub):
                workspace.local_ros_debs()
            downloads = [cmd for cmd in commands if cmd[0] == '/usr/bin/apt-get']
            self.assertEqual(downloads, [['/usr/bin/apt-get', 'download', package]
                                         for package in workspace.LOCAL_DEBS])
            extracts = [cmd for cmd in commands if cmd[0] == '/usr/bin/dpkg-deb']
            self.assertEqual(len(extracts), 2)
            self.assertTrue(all(cmd[1] == '--extract' and cmd[-1] == str(venv / 'ros-debs')
                                for cmd in extracts))

    def test_native_origin_rejects_foreign_python_and_c_artifacts(self):
        check_spec = importlib.util.spec_from_file_location('workspace_check', HERE / 'check_ros_workspace.py')
        checker = importlib.util.module_from_spec(check_spec)
        check_spec.loader.exec_module(checker)
        for origin in ('/foreign/install/estop.py', '/foreign/install/generated.so'):
            with self.assertRaises(AssertionError):
                checker.native_origin(SimpleNamespace(__file__=origin), 'estop', source=True)
        for origin in (workspace.WORKSPACE / 'build/estop/estop.py',
                       workspace.WORKSPACE / 'install/estop/generated.so',
                       workspace.package_path('estop') / 'estop/estop_node.py'):
            checker.native_origin(SimpleNamespace(__file__=str(origin)), 'estop', source=True)

    def test_help_path_independent(self):
        for command in (['--help'], ['build', '--help'], ['setup', '--help']):
            result = subprocess.run(
                [str(SCRIPT), *command], cwd='/tmp',
                env={'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin',
                     'PYTHONHOME': '/foreign/python', 'PYTHONPATH': '/foreign/packages'},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertIn('build_ros_workspace.sh', result.stdout)
        result = subprocess.run(['./scripts/build_ros_workspace.sh', '--help'],
                                cwd=workspace.WORKSPACE, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Examples (from core_ws)', result.stdout)

    def test_build_help_explains_compiler_and_package_parallelism(self):
        result = subprocess.run([str(SCRIPT), 'build', '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--compile-jobs', result.stdout)
        self.assertIn('one at a time', result.stdout)
        self.assertIn('default: 2', result.stdout)
        self.assertIn('manipulation', result.stdout)

    def test_help_explains_lock_maintenance_and_scope(self):
        result = subprocess.run([str(SCRIPT), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ('uv add', 'uv lock', '--no-sync', '--check --offline', '--upgrade-package',
                      '--locked --inexact --no-build', 'environments/pyproject.toml',
                      'environments/apt-packages.txt', 'SDK revision', 'outside the uv graph'):
            self.assertIn(text, result.stdout)

    @unittest.skipUnless(shutil.which('uv'), 'requires uv')
    def test_first_run_uv_venv_creation_with_separate_cache(self):
        with tempfile.TemporaryDirectory(prefix='golem-native-venv-') as directory:
            folder = Path(directory)
            venv = folder / 'venv'
            result = subprocess.run(
                [shutil.which('uv'), 'venv', '--python', '/usr/bin/python3',
                 '--system-site-packages', str(venv)],
                env={'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin',
                     'UV_CACHE_DIR': str(folder / 'cache'), 'UV_NO_CONFIG': '1',
                     'UV_PYTHON_DOWNLOADS': 'never'}, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('include-system-site-packages = true', (venv / 'pyvenv.cfg').read_text())

    @unittest.skipUnless(Path('/opt/ros/humble/setup.bash').is_file(), 'requires installed Humble')
    def test_wrapper_discards_conda_foreign_overlay_and_user_python(self):
        with tempfile.TemporaryDirectory(prefix='golem-native-test-') as directory:
            folder = Path(directory)
            script = folder / 'build_ros_workspace.sh'
            shutil.copy2(SCRIPT, script)
            helpers = folder
            # The entrypoint is a harmless subprocess stub, not imported robot code.
            (helpers / 'build_ros_workspace.py').write_text(
                'import json, os, sys, site\n'
                'print(json.dumps({"env": dict(os.environ), "python": sys.executable, '
                '"usersite": site.ENABLE_USER_SITE}))\n')
            env = os.environ.copy()
            env.update({
                'CONDA_PREFIX': '/foreign/conda', 'CONDA_DEFAULT_ENV': 'base',
                'VIRTUAL_ENV': '/foreign/venv', 'PYTHONPATH': '/foreign/python',
                'PYTHONHOME': '', 'AMENT_PREFIX_PATH': '/foreign/install',
                'CMAKE_PREFIX_PATH': '/foreign/install', 'LD_LIBRARY_PATH': '/foreign/lib',
                'COLCON_PREFIX_PATH': '/foreign/install', 'ROS_DOMAIN_ID': '0',
                'CYCLONEDDS_URI': '/foreign/dds.xml', 'RMW_IMPLEMENTATION': 'foreign_rmw',
            })
            result = subprocess.run([str(script), 'check'],
                                    cwd='/tmp', env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(result.stdout)
            self.assertEqual(actual['python'], '/usr/bin/python3')
            self.assertFalse(actual['usersite'])
            self.assertEqual(actual['env']['PYTHONNOUSERSITE'], '1')
            self.assertEqual(actual['env']['ROS_DOMAIN_ID'], '1')
            self.assertEqual(actual['env']['ROS_DISTRO'], 'humble')
            self.assertNotIn('/foreign/', json.dumps(actual))
            self.assertNotIn('CONDA_PREFIX', actual['env'])
            self.assertNotIn('VIRTUAL_ENV', actual['env'])
            self.assertNotIn('CYCLONEDDS_URI', actual['env'])

    def test_tooling_does_not_initialize_robot_runtime(self):
        import ast
        forbidden = {
            'init', 'spin', 'ChannelFactoryInitialize', 'DomainParticipant',
            'SafetyLayer', 'SafetyNode', 'EstopNode', 'Serial', 'FrameController',
        }
        for filename in ('build_ros_workspace.py', 'check_ros_workspace.py'):
            tree = ast.parse((HERE / filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    function = node.func
                    name = function.id if isinstance(function, ast.Name) else getattr(function, 'attr', '')
                    self.assertNotIn(name, forbidden, f'{filename}:{node.lineno}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
