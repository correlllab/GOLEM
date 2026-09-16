"""Host-side Docker workflow regressions; no daemon or GPU required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]


class Workflows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='golem test ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(REPO / 'docker', self.root / 'docker', ignore=shutil.ignore_patterns('.env', '__pycache__'))
        for name in ('unitree_sdk2_python/setup.py', 'CL_Assets/ros_assets/h1_2_magpie_ros.urdf',
                     'CL_isaaclab_sim/sim_main.py', 'h1_robocasa/h12_mujoco.py',
                     'core_ws/src/magpie_msgs/package.xml', 'core_ws/src/custom_ros_messages/package.xml',
                     'core_ws/src/livox_ros_driver2/package_ROS2.xml', 'core_ws/cyclonedds.xml',
                     'core_ws/src/h1_bringup/package.xml', 'mujoco_mpc/CMakeLists.txt'):
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture')
        for name in ('CL_Assets/graspgenx/assets/magpie', 'CL_Assets/graspgenx/assets/meshes/magpie',
                     '.git/modules/mujoco_mpc', 'tools'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        docker = self.bin / 'docker'
        docker.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'inspect':
    if os.environ.get('EXISTING'):
        print('true')
    else:
        sys.exit(1)
if sys.argv[1] == 'info':
    print('aarch64' if os.environ.get('TEST_MAC') else 'x86_64')
if 'config' in sys.argv:
    os.execv('/usr/bin/docker', ['docker'] + sys.argv[1:])
''')
        docker.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('GOLEM_', 'ROS_', 'HAMS_'))}
        self.env.update(PATH=str(self.bin) + ':' + os.environ['PATH'], CALLS=str(self.root / 'calls'), DISPLAY='')

    def run_script(self, path, *args, **env):
        return subprocess.run(['bash', str(self.root / path), *args], cwd='/tmp',
                              env=dict(self.env, **env), text=True, capture_output=True)

    def calls(self):
        p = self.root / 'calls'
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_pull_refreshes_upstream_base_without_pulling_local_child_base(self):
        r = self.run_script('docker/scripts/docker_build.sh', '--pull', '--no-cache', 'robocasa')
        self.assertEqual(r.returncode, 0, r.stderr)
        builds = [c for c in self.calls() if 'build' in c]
        self.assertEqual(len(builds), 2)
        self.assertIn('--pull', builds[0])
        self.assertNotIn('--pull', builds[1])
        self.assertTrue(all('--no-cache' in c for c in builds))

    def test_run_never_force_removes_existing_container(self):
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', '--headless')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(any(c[:2] == ['rm', '-f'] for c in self.calls()))

    def test_zero_with_leading_zeros_is_rejected(self):
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', ROS_DOMAIN_ID='00')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(any('run' in c for c in self.calls()))

    def test_missing_source_fails_before_container_creation(self):
        (self.root / 'h1_robocasa/h12_mujoco.py').unlink()
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('h12_mujoco.py', r.stderr)
        self.assertFalse(any('run' in c for c in self.calls()))

    def test_isaac_launcher_forwards_options_without_a_terminal(self):
        interpreter = self.root / 'isaac lab/_isaac_sim/python.sh'
        interpreter.parent.mkdir(parents=True)
        interpreter.write_text("#!/usr/bin/env python3\nimport json, os, sys\nprint(json.dumps({'args':sys.argv[1:],'domain':os.environ['ROS_DOMAIN_ID']}))\n")
        interpreter.chmod(0o755)
        # Isolate the compatibility symlink's host filesystem effects.
        for tool in ('mkdir', 'ln'):
            stub = self.bin / tool
            stub.write_text('#!/bin/sh\nexit 0\n')
            stub.chmod(0o755)
        r = self.run_script('docker/scripts/launch_isaac.sh', '--task', 'custom', '--',
                            '--rendering_mode', 'balanced', ROS_DOMAIN_ID='7',
                            ISAACLAB_PATH=str(interpreter.parent.parent))
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout.splitlines()[-1])
        self.assertEqual(result['domain'], '7')
        self.assertIn('--headless', result['args'])
        self.assertEqual(result['args'][-2:], ['--rendering_mode', 'balanced'])
        self.assertEqual(result['args'][result['args'].index('--task') + 1], 'custom')

    def test_lfs_pointer_is_not_accepted_as_an_asset(self):
        asset = self.root / 'CL_Assets/ros_assets/h1_2_magpie_ros.urdf'
        asset.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 10\n')
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('LFS', r.stderr)
        self.assertFalse(any('run' in c for c in self.calls()))

    def test_mac_rejects_isaac(self):
        r = self.run_script('docker/mac/scripts/docker_run_mac.sh', 'isaac', TEST_MAC='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(any('run' in c for c in self.calls()))

    def test_shell_domain_overrides_env_file(self):
        (self.root / 'docker/.env').write_text('ROS_DOMAIN_ID=0\n')
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', ROS_DOMAIN_ID='7')
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_headless_compose_has_no_desktop_mounts_and_uses_asset_override(self):
        assets = self.root / 'external assets'
        assets.mkdir()
        r = subprocess.run(['/usr/bin/docker', 'compose', '--env-file', '/dev/null', '-f',
                            str(self.root / 'docker/docker-compose.yml'), '--profile', 'robocasa',
                            'config', '--format', 'json'], env=dict(self.env, GOLEM_ASSETS_DIR=str(assets)),
                           text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        service = json.loads(r.stdout)['services']['robocasa']
        mounts = {v['target']: v for v in service['volumes']}
        self.assertNotIn('/root/.Xauthority', mounts)
        self.assertNotIn('/tmp/.X11-unix', mounts)
        self.assertEqual(mounts['/home/code/CL_Assets']['source'], str(assets))
        self.assertFalse(mounts['/home/code/CL_Assets'].get('bind', {}).get('create_host_path', False))

    def test_mac_stack_warms_ros_before_starting_pair(self):
        r = self.run_script('docker/mac/scripts/docker_stack_mac.sh', 'up', 'robocasa', TEST_MAC='1')
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.calls()
        warm = next(i for i, c in enumerate(calls) if 'run' in c and '/bin/true' in c)
        start = next(i for i, c in enumerate(calls) if 'up' in c)
        self.assertLess(warm, start)
        self.assertIn('ros', calls[start])
        self.assertIn('robocasa', calls[start])

    def test_existing_container_requires_explicit_restart(self):
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', EXISTING='1')
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(any(c[0] == 'rm' or 'run' in c for c in self.calls()))

    def test_explicit_restart_removes_only_selected_container(self):
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', '--restart', EXISTING='1')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(['rm', '-f', 'golem_sim_robocasa'], self.calls())

    def test_asset_and_cache_paths_with_spaces(self):
        assets = self.root / 'asset disk'
        shutil.copytree(self.root / 'CL_Assets', assets)
        cache = self.root / 'cache disk'
        r = self.run_script('docker/scripts/docker_run.sh', 'isaac', '--headless',
                            GOLEM_ASSETS_DIR=str(assets), GOLEM_CACHE_DIR=str(cache))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((cache / 'isaac/nvidia').is_dir())
        self.assertFalse((self.root / 'CL_isaaclab_sim/.isaac_cache').exists())

    def test_runtime_sim_domain_guard_applies_without_wrapper(self):
        script = self.root / 'docker/scripts/runtime_common.sh'
        for domain in ('0', '00', '-1', '233', 'invalid'):
            with self.subTest(domain=domain):
                r = subprocess.run(['bash', '-c', 'source "$1"; validate_sim_domain', 'bash', str(script)],
                                   env=dict(self.env, ROS_DOMAIN_ID=domain), capture_output=True)
                self.assertNotEqual(r.returncode, 0)

    def test_mac_compose_keeps_native_build_volumes_without_gpu_requirements(self):
        r = subprocess.run(['/usr/bin/docker', 'compose', '--env-file', '/dev/null', '-f',
                            str(self.root / 'docker/mac/docker-compose.yml'), 'config', '--format', 'json'],
                           env=self.env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        services = json.loads(r.stdout)['services']
        self.assertEqual(set(services), {'ros', 'robocasa'})
        for service in services.values():
            self.assertEqual(service['platform'], 'linux/arm64')
            self.assertNotIn('runtime', service)
            self.assertEqual(service['ipc'], 'host')
            self.assertTrue(any(v['type'] == 'volume' and v['target'].endswith('/build') for v in service['volumes']))

    def test_stack_restart_replaces_single_service_containers(self):
        r = self.run_script('docker/scripts/docker_stack.sh', 'restart', 'robocasa', EXISTING='1')
        self.assertEqual(r.returncode, 0, r.stderr)
        calls = self.calls()
        for name in ('golem_ros', 'golem_sim_robocasa'):
            self.assertIn(['rm', '-f', name], calls)
        first_run = next(i for i, c in enumerate(calls) if 'run' in c)
        self.assertTrue(all(i < first_run for i, c in enumerate(calls) if c[:2] == ['rm', '-f']))

    def test_stack_stop_does_not_require_display_or_mounted_assets(self):
        shutil.rmtree(self.root / 'CL_Assets')
        r = self.run_script('docker/scripts/docker_stack.sh', 'stop', 'robocasa', GOLEM_DISPLAY='gui')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(any('run' in c or 'up' in c for c in self.calls()))


if __name__ == '__main__':
    unittest.main()
