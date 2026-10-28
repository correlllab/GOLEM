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
                     'CL_isaaclab_sim/sim_main.py', 'golem_robocasa/h12_mujoco.py',
                     'core_ws/src/magpie_msgs/package.xml', 'core_ws/src/custom_ros_messages/package.xml',
                     'core_ws/src/livox_ros_driver2/package_ROS2.xml', 'core_ws/cyclonedds.xml',
                     'core_ws/src/golem_bringup/package.xml', 'mujoco_mpc/CMakeLists.txt'):
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
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('GOLEM_', 'ROS_', 'HAMS_', 'G1_'))}
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
        (self.root / 'golem_robocasa/h12_mujoco.py').unlink()
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

    # --- robot selection: GOLEM_ROBOT, --g1 / --h12 -------------------------------------------

    G1_URDF = 'third_party/leprechaun_g1/assets/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
    G1_FILES = (
        G1_URDF,
        'third_party/leprechaun_g1/assets/g1_description/g1_29dof_with_hand_rev_1_0.xml',
        'core_ws/src/g1/g1_locomotion/policy/deploy.onnx',
        'core_ws/src/g1/g1_locomotion/policy/g1.yaml',
        'core_ws/src/g1/g1_locomotion/policy/homie_policy.py',
    )
    SUBMODULE_HINT = 'git submodule update --init third_party/leprechaun_g1'

    def add_g1_files(self):
        for name in self.G1_FILES:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('fixture')

    def overlay(self):
        return str(self.root / 'docker/docker-compose.g1.yml')

    @staticmethod
    def compose_files(argv):
        return [b for a, b in zip(argv, argv[1:]) if a == '-f']

    def compose_calls(self):
        return [c for c in self.calls() if c[0] == 'compose']

    def run_calls(self):
        return [c for c in self.calls() if c[0] == 'compose' and 'run' in c]

    def run_with_input(self, path, *args, stdin='', **env):
        return subprocess.run(['bash', str(self.root / path), *args], cwd='/tmp', input=stdin,
                              env=dict(self.env, **env), text=True, capture_output=True)

    def install_ip_stub(self):
        ip = self.bin / 'ip'
        ip.write_text('''#!/usr/bin/env bash
dev=${!#}
[ "$dev" != nosuchif0 ] || exit 1
case " $* " in
    *' -br link '*) echo "$dev $IP_STATE 00:11:22:33:44:55 <BROADCAST,MULTICAST,UP,LOWER_UP>";;
    *' -o -4 addr '*) [ -z "$IP_INET" ] || echo "2: $dev    inet $IP_INET brd 192.168.123.255 scope global $dev";;
esac
''')
        ip.chmod(0o755)

    def test_default_robot_compose_unchanged(self):
        base = ['compose', '--env-file', '/dev/null', '-f', str(self.root / 'docker/docker-compose.yml')]
        for service, name in (('robocasa', 'golem_sim_robocasa'), ('ros', 'golem_ros')):
            with self.subTest(service=service):
                r = self.run_script('docker/scripts/docker_run.sh', service)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(self.run_calls()[-1], base + ['run', '--rm', '--name', name, service])
                self.assertFalse(any(self.overlay() in c for c in self.calls()))
                self.assertIn('robot: h12', r.stderr)

    def test_g1_flag_adds_overlay(self):
        self.add_g1_files()
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', '--g1')
        self.assertEqual(r.returncode, 0, r.stderr)
        run = self.run_calls()[-1]
        self.assertIn(self.overlay(), self.compose_files(run))
        self.assertEqual(run[run.index('run'):], ['run', '--rm', '--name', 'golem_sim_robocasa', 'robocasa'])
        self.assertEqual(r.stderr.count('robot: g1'), 1)

    def test_robot_flags_are_stripped_wherever_they_appear(self):
        self.add_g1_files()
        cases = (
            (('robocasa', '--g1', '--headless'), 'g1', ['launch_robocasa.sh', '--headless']),
            (('robocasa', '--headless', '--g1'), 'g1', ['launch_robocasa.sh', '--headless']),
            (('--g1', 'ros', 'bash'), 'g1', ['bash']),
            (('ros', 'bash', '--g1'), 'g1', ['bash']),
            (('ros', '--h12', 'bash'), 'h12', ['bash']),
        )
        for args, robot, tail in cases:
            with self.subTest(args=args):
                (self.root / 'calls').unlink(missing_ok=True)
                r = self.run_script('docker/scripts/docker_run.sh', *args)
                self.assertEqual(r.returncode, 0, r.stderr)
                run = self.run_calls()[-1]
                self.assertEqual(self.overlay() in self.compose_files(run), robot == 'g1')
                got = run[len(run) - len(tail):]
                self.assertTrue(got[0].endswith(tail[0]), run)
                self.assertEqual(got[1:], tail[1:])
                self.assertFalse({'--g1', '--h12'} & set(run), run)

    def test_robot_flag_composes_with_restart_in_either_order(self):
        self.add_g1_files()
        for args in (('ros', '--g1', '--restart'), ('ros', '--restart', '--g1')):
            with self.subTest(args=args):
                (self.root / 'calls').unlink(missing_ok=True)
                r = self.run_script('docker/scripts/docker_run.sh', *args, EXISTING='1')
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(['rm', '-f', 'golem_ros'], self.calls())
                self.assertIn(self.overlay(), self.compose_files(self.run_calls()[-1]))

    def test_quoted_command_containing_robot_flag_is_left_alone(self):
        self.add_g1_files()
        r = self.run_script('docker/scripts/docker_run.sh', 'ros', 'bash', '-lc', 'echo --g1')
        self.assertEqual(r.returncode, 0, r.stderr)
        run = self.run_calls()[-1]
        self.assertEqual(run[-3:], ['bash', '-lc', 'echo --g1'])
        self.assertNotIn(self.overlay(), self.compose_files(run))

    def test_robot_precedence_flag_over_shell_over_env_file(self):
        self.add_g1_files()
        (self.root / 'docker/.env').write_text('GOLEM_ROBOT=g1\n')
        cases = (
            ((), {}, True),                                   # .env alone
            ((), {'GOLEM_ROBOT': 'h12'}, False),              # shell beats .env
            (('--h12',), {}, False),                          # flag beats .env
            (('--h12',), {'GOLEM_ROBOT': 'g1'}, False),       # flag beats shell
            (('--g1',), {'GOLEM_ROBOT': 'h12'}, True),        # flag beats shell
            ((), {'GOLEM_ROBOT': 'G1'}, True),                # case-insensitive
        )
        for args, env, g1 in cases:
            with self.subTest(args=args, env=env):
                (self.root / 'calls').unlink(missing_ok=True)
                r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', *args, **env)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(self.overlay() in self.compose_files(self.run_calls()[-1]), g1)
                self.assertIn('robot: g1' if g1 else 'robot: h12', r.stderr)

    def test_env_robot_g1_adds_overlay(self):
        self.add_g1_files()
        (self.root / 'docker/.env').write_text('GOLEM_ROBOT=g1\n')
        for service in ('robocasa', 'ros'):
            with self.subTest(service=service):
                r = self.run_script('docker/scripts/docker_run.sh', service)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertIn(self.overlay(), self.compose_files(self.run_calls()[-1]))

    def test_invalid_or_conflicting_robot_is_rejected(self):
        self.add_g1_files()
        for args, env in ((('robocasa',), {'GOLEM_ROBOT': 'h1'}),
                          (('robocasa', '--g1', '--h12'), {})):
            with self.subTest(args=args, env=env):
                r = self.run_script('docker/scripts/docker_run.sh', *args, **env)
                self.assertNotEqual(r.returncode, 0)
                self.assertFalse(self.compose_calls())

    def test_g1_writes_cyclonedds_local_xml(self):
        self.add_g1_files()
        xml = self.root / 'container_cache/cyclonedds_local.xml'
        r = self.run_script('docker/scripts/docker_run.sh', 'ros', '--g1')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(xml.read_text().strip(), '<CycloneDDS/>')
        r = self.run_script('docker/scripts/docker_run.sh', 'ros', '--g1', G1_DDS_INTERFACE='eth7')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(xml.read_text().strip(),
                         '<CycloneDDS><Domain Id="any"><General><Interfaces>'
                         '<NetworkInterface name="eth7"/></Interfaces></General></Domain></CycloneDDS>')

    def test_h12_does_not_write_cyclonedds_local_xml(self):
        r = self.run_script('docker/scripts/docker_run.sh', 'ros', G1_DDS_INTERFACE='eth7')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse((self.root / 'container_cache/cyclonedds_local.xml').exists())

    # --- the shared cyclonedds_local.xml and a live G1 ros container ---------------------------

    PIN = ('<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="{}"/>'
           '</Interfaces></General></Domain></CycloneDDS>')

    def live_pin(self):
        """The cyclonedds_local.xml a running domain-0 G1 ros container was started with."""
        xml = self.root / 'container_cache/cyclonedds_local.xml'
        xml.parent.mkdir(parents=True, exist_ok=True)
        content = self.PIN.format('enp130s0') + '\n'
        xml.write_text(content)
        return xml, content, xml.stat()

    def assert_pin_untouched(self, xml, content, before):
        after = xml.stat()
        self.assertEqual((after.st_ino, after.st_mtime_ns), (before.st_ino, before.st_mtime_ns))
        self.assertEqual(xml.read_text(), content)

    def install_copy(self, name):
        dest = self.root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, dest)

    def install_flock_stub(self):
        # tests/robocasa/surface/run.sh locks a host-wide file in /tmp; a real probe may hold it.
        flock = self.bin / 'flock'
        flock.write_text('#!/bin/sh\nexit 0\n')
        flock.chmod(0o755)

    def run_g1_test_setup(self, out):
        """tests/g1/common.sh's g1_test_setup, then its EXIT cleanup, under the stub docker."""
        return subprocess.run(['bash', '-c', 'set -eo pipefail; source "$1"; g1_test_setup t bringup arm 195',
                               'bash', str(self.root / 'tests/g1/common.sh')], cwd='/tmp',
                              env=dict(self.env, OUT=str(out)), text=True, capture_output=True)

    def test_g1_sim_runs_leave_the_dds_pin_alone(self):
        # A running G1 ros container reads the shared file through a single-file bind
        # mount each time a process in it starts, so nothing that starts no ros
        # container may change it.
        self.add_g1_files()
        self.install_flock_stub()
        for name in ('tests/robocasa/g1/run_in_container.sh', 'tests/robocasa/surface/run.sh',
                     'tests/g1/common.sh'):
            self.install_copy(name)
        (self.root / 'tests/robocasa/surface/g1.json').write_text('{}\n')
        runs = (
            ('docker_run.sh robocasa --g1, another interface',
             lambda: self.run_script('docker/scripts/docker_run.sh', 'robocasa', '--g1', G1_DDS_INTERFACE='eth7')),
            ('docker_run.sh robocasa --g1, empty interface',
             lambda: self.run_script('docker/scripts/docker_run.sh', 'robocasa', '--g1', G1_DDS_INTERFACE='')),
            ('tests/robocasa/g1/run_in_container.sh',
             lambda: self.run_script('tests/robocasa/g1/run_in_container.sh', G1_DDS_INTERFACE='eth7')),
            ('tests/robocasa/surface/run.sh g1',
             lambda: self.run_script('tests/robocasa/surface/run.sh', 'g1', G1_DDS_INTERFACE='eth7')),
        )
        for label, run in runs:
            with self.subTest(label):
                xml, content, before = self.live_pin()
                (self.root / 'calls').unlink(missing_ok=True)
                r = run()
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(self.run_calls(), f'{label} never reached its compose run')
                self.assert_pin_untouched(xml, content, before)
        with self.subTest('tests/g1/common.sh binds a pin of its own'):
            xml, content, before = self.live_pin()
            (self.root / 'calls').unlink(missing_ok=True)
            out = self.root / 'g1 out'
            r = self.run_g1_test_setup(out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assert_pin_untouched(xml, content, before)
            self.assertEqual((out / 'cyclonedds_local.xml').read_text().strip(), '<CycloneDDS/>')
            r = subprocess.run(['/usr/bin/docker', 'compose', '--env-file', '/dev/null',
                                '-f', str(self.root / 'docker/docker-compose.yml'), '-f', self.overlay(),
                                '-f', str(out / 'test_compose.yml'), '--profile', 'ros', 'config', '--format', 'json'],
                               env=dict(self.env, GOLEM_CACHE_DIR=str(self.root / 'container_cache')),
                               text=True, capture_output=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            volumes = json.loads(r.stdout)['services']['ros']['volumes']
            pins = [v for v in volumes if v['target'] == '/home/code/cyclonedds_local.xml']
            self.assertEqual([v['source'] for v in pins], [str(out / 'cyclonedds_local.xml')])
            self.assertTrue(pins[0]['read_only'])
            # every container g1_test_setup's runs create is removed on exit, the build's too
            removed = next(c for c in self.calls() if c[:2] == ['rm', '-f'])
            self.assertTrue(any(n.endswith('-ros-build') for n in removed), removed)

    def test_g1_ros_start_replaces_dds_pin_atomically(self):
        self.add_g1_files()
        starts = (
            ('docker_run.sh ros --g1', 'docker/scripts/docker_run.sh', ('ros', '--g1'), 'eth7'),
            ('docker_stack.sh up', 'docker/scripts/docker_stack.sh', ('up', 'robocasa', '--g1', '--headless'), ''),
            ('docker_stack.sh restart', 'docker/scripts/docker_stack.sh', ('restart', 'robocasa', '--g1'), 'eth7'),
        )
        for label, script, args, interface in starts:
            with self.subTest(label):
                xml, content, before = self.live_pin()
                # The running container's bind mount holds the inode it started with.
                with open(xml) as mounted:
                    r = self.run_script(script, *args, G1_DDS_INTERFACE=interface)
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertEqual(xml.read_text().strip(),
                                     self.PIN.format(interface) if interface else '<CycloneDDS/>')
                    self.assertNotEqual(xml.stat().st_ino, before.st_ino)
                    self.assertEqual(mounted.read(), content)
                self.assertEqual(xml.stat().st_mode & 0o777, 0o644)
                self.assertEqual(list(xml.parent.glob('cyclonedds_local.xml?*')), [])

    def test_g1_dds_pin_is_written_only_when_a_ros_container_starts(self):
        self.add_g1_files()
        self.install_ip_stub()
        up = {'IP_STATE': 'UP', 'IP_INET': '192.168.123.222/24'}
        refused = (
            ('golem_ros exists, no --restart', 'docker/scripts/docker_run.sh', ('ros', '--g1'), '',
             {'EXISTING': '1'}),
            ('domain 0 declined', 'docker/scripts/docker_run.sh', ('ros', '--g1'), 'n\n',
             dict(up, ROS_DOMAIN_ID='0')),
            ('stack already running', 'docker/scripts/docker_stack.sh', ('up', 'robocasa', '--g1'), '',
             {'EXISTING': '1'}),
        )
        for label, script, args, stdin, env in refused:
            with self.subTest(label):
                xml, content, before = self.live_pin()
                r = self.run_with_input(script, *args, stdin=stdin, G1_DDS_INTERFACE='eth7', **env)
                self.assertNotEqual(r.returncode, 0)
                self.assert_pin_untouched(xml, content, before)

    def test_g1_dds_interface_must_be_an_interface_name(self):
        self.add_g1_files()
        xml = self.root / 'container_cache/cyclonedds_local.xml'
        for name in ('eth0"/><Tracing', 'a' * 16, 'eth 0', 'eth0\n', 'éth0', 'eth/0'):
            for service in ('ros', 'robocasa'):
                with self.subTest(name=name, service=service):
                    (self.root / 'calls').unlink(missing_ok=True)
                    xml.unlink(missing_ok=True)
                    r = self.run_script('docker/scripts/docker_run.sh', service, '--g1', G1_DDS_INTERFACE=name)
                    self.assertNotEqual(r.returncode, 0)
                    self.assertIn('is not a network interface name', r.stderr)
                    self.assertFalse(self.compose_calls())
                    self.assertFalse(xml.exists())
        for name in ('enp130s0', 'eth0.100', 'br-0123456789ab', 'wlp0s20f3'):
            with self.subTest(name=name):
                r = self.run_script('docker/scripts/docker_run.sh', 'ros', '--g1', G1_DDS_INTERFACE=name)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(xml.read_text().strip(), self.PIN.format(name))

    def test_g1_interface_on_a_sim_domain_warns(self):
        self.add_g1_files()
        self.install_ip_stub()
        up = {'IP_STATE': 'UP', 'IP_INET': '192.168.123.222/24'}

        def warnings(r):
            self.assertEqual(r.returncode, 0, r.stderr)
            return [line for line in r.stderr.splitlines() if 'WARNING' in line]

        warned = (
            ('docker_run.sh robocasa', 'docker/scripts/docker_run.sh', ('robocasa', '--g1'), {}),
            ('docker_run.sh ros', 'docker/scripts/docker_run.sh', ('ros', '--g1'), {}),
            ('docker_stack.sh up', 'docker/scripts/docker_stack.sh', ('up', 'robocasa', '--g1'), {}),
        )
        for label, script, args, env in warned:
            with self.subTest(label):
                lines = warnings(self.run_script(script, *args, G1_DDS_INTERFACE='eth7', **env))
                self.assertEqual(len(lines), 1, lines)
                for text in ('G1_DDS_INTERFACE=eth7', 'ros container', 'domain 0', 'empty', 'docs/g1/dds.md'):
                    self.assertIn(text, lines[0])
        quiet = (
            ('empty interface', ('robocasa', '--g1'), {'G1_DDS_INTERFACE': ''}, ''),
            ('the H1-2', ('robocasa',), {'G1_DDS_INTERFACE': 'eth7'}, ''),
            ('the real robot on domain 0', ('ros', '--g1'), dict(up, G1_DDS_INTERFACE='eth7', ROS_DOMAIN_ID='0'), 'y\n'),
        )
        for label, args, env, stdin in quiet:
            with self.subTest(label):
                r = self.run_with_input('docker/scripts/docker_run.sh', *args, stdin=stdin, **env)
                self.assertEqual(warnings(r), [])

    def test_stack_wrappers_ignore_golem_robot(self):
        # A G1 selection left in docker/.env must not reach the H1 test wrappers.
        (self.root / 'docker/.env').write_text('GOLEM_ROBOT=g1\n')
        for name in ('tests/stack/run.sh', 'tests/robocasa/run_in_container.sh'):
            dest = self.root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(REPO / name, dest)
        cases = (('tests/stack/run.sh', ('robocasa', 'arms')), ('tests/robocasa/run_in_container.sh', ()))
        for name, args in cases:
            with self.subTest(wrapper=name):
                (self.root / 'calls').unlink(missing_ok=True)
                self.run_script(name, *args)
                # The fixture has no G1 files: reaching compose proves no G1 mount check ran.
                self.assertTrue(self.run_calls(), f'{name} never reached its compose run')
                self.assertFalse(any(self.overlay() in c for c in self.calls()))

    def test_g1_missing_leprechaun_fails_early(self):
        self.add_g1_files()
        (self.root / self.G1_URDF).unlink()
        r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', '--g1')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(self.SUBMODULE_HINT, r.stderr)
        self.assertFalse(self.compose_calls())

    def test_g1_missing_policy_file_fails_early(self):
        self.add_g1_files()
        policy = self.root / 'core_ws/src/g1/g1_locomotion/policy/deploy.onnx'
        policy.unlink()
        for service in ('robocasa', 'ros'):
            with self.subTest(service=service):
                r = self.run_script('docker/scripts/docker_run.sh', service, '--g1')
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('deploy.onnx', r.stderr)
                self.assertFalse(self.compose_calls())

    def test_h12_ignores_missing_leprechaun(self):
        self.assertFalse((self.root / self.G1_URDF).exists())
        for args in ((), ('--h12',)):
            with self.subTest(args=args):
                r = self.run_script('docker/scripts/docker_run.sh', 'robocasa', *args)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(self.run_calls())
        r = self.run_script('docker/scripts/docker_run.sh', 'ros', GOLEM_ROBOT='h12')
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_g1_rejected_on_mac_and_isaac(self):
        self.add_g1_files()
        cases = (
            ('docker/scripts/docker_run.sh', ('robocasa', '--g1'), {'GOLEM_PLATFORM': 'mac', 'TEST_MAC': '1'}),
            ('docker/mac/scripts/docker_run_mac.sh', ('ros', '--g1'), {'TEST_MAC': '1'}),
            ('docker/mac/scripts/docker_stack_mac.sh', ('up', 'robocasa', '--g1'), {'TEST_MAC': '1'}),
            ('docker/scripts/docker_run.sh', ('isaac', '--g1'), {}),
            ('docker/scripts/docker_run.sh', ('isaac',), {'GOLEM_ROBOT': 'g1'}),
            ('docker/scripts/docker_stack.sh', ('up', 'isaac', '--g1'), {}),
        )
        for path, args, env in cases:
            with self.subTest(path=path, args=args, env=env):
                r = self.run_script(path, *args, **env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('G1', r.stderr)
                self.assertFalse(self.compose_calls())

    def test_g1_domain0_requires_interface(self):
        self.add_g1_files()
        self.install_ip_stub()
        up = {'IP_STATE': 'UP', 'IP_INET': '192.168.123.222/24'}

        def domain0(stdin='y\n', **env):
            (self.root / 'calls').unlink(missing_ok=True)
            return self.run_with_input('docker/scripts/docker_run.sh', 'ros', stdin=stdin,
                                       ROS_DOMAIN_ID='0', GOLEM_ROBOT='g1', **env)

        refused = (
            ('empty', {'G1_DDS_INTERFACE': ''}, 'G1_DDS_INTERFACE'),
            ('missing', dict(up, G1_DDS_INTERFACE='nosuchif0'), 'is not a network interface'),
            ('down', dict(up, G1_DDS_INTERFACE='eth7', IP_STATE='DOWN'), 'not UP'),
            ('no ipv4', dict(up, G1_DDS_INTERFACE='eth7', IP_INET=''), 'has no IPv4 address'),
        )
        for label, env, message in refused:
            with self.subTest(label):
                # 'y' is on stdin: a prompt that was read would proceed to compose.
                r = domain0(**env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(message, r.stderr)
                self.assertNotIn('Real-robot launch cancelled', r.stderr)
                self.assertFalse(self.compose_calls())
        with self.subTest('ip missing'):
            tools = self.root / 'tools-without-ip'
            tools.mkdir()
            for tool in ('bash', 'env', 'sed', 'dirname', 'tr', 'mkdir'):
                (tools / tool).symlink_to(shutil.which(tool))
            r = domain0(G1_DDS_INTERFACE='eth7', PATH=str(tools))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("'ip' (iproute2) is needed", r.stderr)
            self.assertFalse(self.compose_calls())
        with self.subTest('usable interface reaches the prompt and a decline stops there'):
            r = domain0(stdin='n\n', G1_DDS_INTERFACE='eth7', **up)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('Real-robot launch cancelled', r.stderr)
            self.assertFalse(self.compose_calls())
            prompt = self.domain0_prompt(r)
            for text in ('robot: g1', 'interface: eth7', '192.168.123.222/24'):
                self.assertIn(text, prompt)
        with self.subTest('usable interface and a yes proceeds with the overlay'):
            r = domain0(stdin='y\n', G1_DDS_INTERFACE='eth7', **up)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(self.overlay(), self.compose_files(self.run_calls()[-1]))
        with self.subTest('only the ros service may use domain 0'):
            r = self.run_with_input('docker/scripts/docker_run.sh', 'robocasa', '--g1', stdin='y\n',
                                    ROS_DOMAIN_ID='0', G1_DDS_INTERFACE='eth7', **up)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('reserved for the real robot', r.stderr)

    def test_g1_domain0_check_follows_every_robot_selection(self):
        # check_domain must see the robot the wrapper resolved from --g1 or from a
        # value that only normalises after configure_robot (uppercase), not just a
        # GOLEM_ROBOT=g1 that is already in the environment.
        self.add_g1_files()
        self.install_ip_stub()
        up = {'IP_STATE': 'UP', 'IP_INET': '192.168.123.222/24'}
        selections = (
            ('flag, GOLEM_ROBOT unset', ('--g1',), None, {}),
            ('flag after the service', ('--restart', '--g1'), None, {}),
            ('uppercase GOLEM_ROBOT in the shell', (), None, {'GOLEM_ROBOT': 'G1'}),
            ('uppercase GOLEM_ROBOT in docker/.env', (), 'GOLEM_ROBOT=G1\n', {}),
        )
        for label, args, dotenv, shell in selections:
            env_file = self.root / 'docker/.env'
            env_file.unlink(missing_ok=True)
            if dotenv:
                env_file.write_text(dotenv)

            def domain0(stdin, **env):
                (self.root / 'calls').unlink(missing_ok=True)
                return self.run_with_input('docker/scripts/docker_run.sh', 'ros', *args, stdin=stdin,
                                           ROS_DOMAIN_ID='0', **shell, **env)

            with self.subTest(label, case='empty interface'):
                # 'y' is on stdin: a prompt that was read would proceed to compose.
                r = domain0('y\n', G1_DDS_INTERFACE='')
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('G1_DDS_INTERFACE', r.stderr)
                self.assertNotIn('Real-robot launch cancelled', r.stderr)
                self.assertFalse(self.compose_calls())
            with self.subTest(label, case='no such interface'):
                r = domain0('y\n', G1_DDS_INTERFACE='nosuchif0', **up)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('is not a network interface', r.stderr)
                self.assertFalse(self.compose_calls())
            with self.subTest(label, case='usable interface reaches the prompt'):
                r = domain0('n\n', G1_DDS_INTERFACE='eth7', **up)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('Real-robot launch cancelled', r.stderr)
                self.assertFalse(self.compose_calls())
        with self.subTest('the stack wrapper refuses domain 0 for its simulator'):
            (self.root / 'docker/.env').unlink(missing_ok=True)
            (self.root / 'calls').unlink(missing_ok=True)
            r = self.run_with_input('docker/scripts/docker_stack.sh', 'up', 'robocasa', '--g1', stdin='y\n',
                                    ROS_DOMAIN_ID='0', G1_DDS_INTERFACE='eth7', **up)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('reserved for the real robot', r.stderr)
            self.assertFalse(self.compose_calls())

    def domain0_prompt(self, r):
        """The domain-0 confirmation the wrapper printed, from its first line to the question."""
        start = 'Domain 0 is the REAL ROBOT bus.'
        self.assertIn(start, r.stderr)
        prompt = r.stderr[r.stderr.index(start):]
        self.assertIn('Proceed? [y/N]', prompt)
        return prompt[:prompt.index('Proceed? [y/N]')]

    def test_h12_domain0_needs_no_interface(self):
        r = self.run_with_input('docker/scripts/docker_run.sh', 'ros', stdin='n\n', ROS_DOMAIN_ID='0')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Real-robot launch cancelled', r.stderr)
        prompt = self.domain0_prompt(r)
        self.assertIn('robot: h12', prompt)
        self.assertNotIn('interface', prompt)
        r = self.run_with_input('docker/scripts/docker_run.sh', 'ros', stdin='y\n', ROS_DOMAIN_ID='0')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn(self.overlay(), self.compose_files(self.run_calls()[-1]))

    def install_env_recording_docker(self):
        """The stub docker, plus the robot/bringup environment each call was made with."""
        docker = self.bin / 'docker'
        docker.write_text(docker.read_text().replace(
            "if sys.argv[1] == 'inspect':",
            "with open(os.environ['CALLS'] + '.env', 'a') as f:\n"
            "    f.write(json.dumps({k: os.environ.get(k) for k in ('GOLEM_ROBOT', 'GOLEM_BRINGUP_PACKAGE',"
            " 'GOLEM_BRINGUP_LAUNCH')}) + '\\n')\n"
            "if sys.argv[1] == 'inspect':", 1))

    def stack_up_env(self):
        calls = self.calls()
        envs = [json.loads(x) for x in (self.root / 'calls.env').read_text().splitlines()]
        return envs[next(i for i, c in enumerate(calls) if 'up' in c)]

    def test_stack_g1_selects_the_g1_sim_launch(self):
        self.add_g1_files()
        self.install_env_recording_docker()
        r = self.run_script('docker/scripts/docker_stack.sh', 'up', 'robocasa', '--g1', '--headless')
        self.assertEqual(r.returncode, 0, r.stderr)
        for call in self.compose_calls():
            self.assertIn(self.overlay(), self.compose_files(call))
        self.assertEqual(self.stack_up_env(), {'GOLEM_ROBOT': 'g1', 'GOLEM_BRINGUP_PACKAGE': 'golem_bringup',
                                               'GOLEM_BRINGUP_LAUNCH': 'g1_sim.launch.py'})
        self.assertEqual(r.stderr.count('robot: g1'), 1)

    def test_stack_g1_overrides_leftover_bringup_selection(self):
        self.add_g1_files()
        self.install_env_recording_docker()
        (self.root / 'docker/.env').write_text('GOLEM_BRINGUP_PACKAGE=leftover_bringup\n')
        r = self.run_script('docker/scripts/docker_stack.sh', 'up', 'robocasa', '--g1', '--headless',
                            GOLEM_BRINGUP_LAUNCH='h12_sim.launch.py')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.stack_up_env(), {'GOLEM_ROBOT': 'g1', 'GOLEM_BRINGUP_PACKAGE': 'golem_bringup',
                                               'GOLEM_BRINGUP_LAUNCH': 'g1_sim.launch.py'})

    def test_stack_default_robot_leaves_bringup_to_compose_defaults(self):
        self.install_env_recording_docker()
        r = self.run_script('docker/scripts/docker_stack.sh', 'up', 'robocasa', '--headless')
        self.assertEqual(r.returncode, 0, r.stderr)
        for call in self.compose_calls():
            self.assertNotIn(self.overlay(), self.compose_files(call))
        self.assertEqual(self.stack_up_env(), {'GOLEM_ROBOT': 'h12', 'GOLEM_BRINGUP_PACKAGE': None,
                                               'GOLEM_BRINGUP_LAUNCH': None})

    def test_stack_stop_and_logs_do_not_need_the_robot_selection(self):
        (self.root / 'docker/.env').write_text('GOLEM_ROBOT=nonsense\n')
        for action in ('stop', 'logs'):
            with self.subTest(action=action):
                r = self.run_script('docker/scripts/docker_stack.sh', action, 'robocasa')
                self.assertEqual(r.returncode, 0, r.stderr)

    def compose_config(self, *files, profile, **env):
        r = subprocess.run(['/usr/bin/docker', 'compose', '--env-file', '/dev/null',
                            *[a for f in files for a in ('-f', str(self.root / 'docker' / f))],
                            '--profile', profile, 'config', '--format', 'json'],
                           env=dict(self.env, **env), text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)['services']

    def test_base_robocasa_service_has_no_cmd_timeout(self):
        # H1 raises on an empty GOLEM_CMD_TIMEOUT (golem_robocasa/unitree_interface.py).
        env = self.compose_config('docker-compose.yml', profile='robocasa')['robocasa']['environment']
        self.assertNotIn('GOLEM_CMD_TIMEOUT', env)
        self.assertNotIn('GOLEM_ROBOT', env)

    def test_g1_overlay_compose_contents(self):
        services = self.compose_config('docker-compose.yml', 'docker-compose.g1.yml', profile='robocasa')
        sim = services['robocasa']['environment']
        self.assertEqual(sim['GOLEM_ROBOT'], 'g1')
        for key in ('G1_DDS_INTERFACE', 'GOLEM_CMD_TIMEOUT', 'GOLEM_ARM_SDK_TIMEOUT', 'GOLEM_LIDAR',
                    'GOLEM_LIDAR_POINTS', 'GOLEM_LIDAR_HZ', 'GOLEM_CAMERA_HZ', 'GOLEM_SPAWN_BACKOFF'):
            self.assertEqual(sim[key], '', key)
        self.assertEqual(sim['GOLEM_CAMERA_SCALE'], '1.0')
        ros = self.compose_config('docker-compose.yml', 'docker-compose.g1.yml', profile='ros',
                                  GOLEM_CACHE_DIR='/cache')['ros']
        self.assertEqual(ros['environment']['GOLEM_ROBOT'], 'g1')
        self.assertIn('G1_DDS_INTERFACE', ros['environment'])
        self.assertEqual(ros['environment']['CYCLONEDDS_URI'],
                         'file:///home/code/cyclonedds.xml,file:///home/code/cyclonedds_local.xml')
        for service, extra in ((services['robocasa'], {}), (ros, {'/home/code/cyclonedds_local.xml': '/cache/cyclonedds_local.xml'})):
            mounts = {v['target']: v for v in service['volumes']}
            expected = {'/home/code/third_party/leprechaun_g1': str(self.root / 'third_party/leprechaun_g1'),
                        '/home/code/g1_locomotion': str(self.root / 'core_ws/src/g1/g1_locomotion/policy'), **extra}
            for target, source in expected.items():
                self.assertEqual(mounts[target]['source'], source)
                self.assertTrue(mounts[target]['read_only'], target)
                self.assertFalse(mounts[target].get('bind', {}).get('create_host_path', False), target)

    def test_stack_compose_bringup_defaults_and_overrides(self):
        files = ('docker-compose.yml', 'docker-compose.stack.yml')
        command = self.compose_config(*files, profile='ros')['ros']['command']
        self.assertEqual(command[1:5], ['ros2', 'launch', 'golem_bringup', 'h12_sim.launch.py'])
        command = self.compose_config(*files, profile='ros', GOLEM_BRINGUP_PACKAGE='other_bringup',
                                      GOLEM_BRINGUP_LAUNCH='g1_sim.launch.py')['ros']['command']
        self.assertEqual(command[1:5], ['ros2', 'launch', 'other_bringup', 'g1_sim.launch.py'])


if __name__ == '__main__':
    unittest.main()
