"""Check docker_common.sh's configure_cpus: which CPUs the sim and ros containers are pinned to."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
DOCKER = shutil.which('docker')


class CpuPinning(unittest.TestCase):
    """Run configure_cpus in a subshell of a fixture checkout, against a fake sysfs."""

    def setUp(self):
        """Copy docker/ into a fixture checkout (no docker/.env) and build the fake sysfs root."""
        self.tmp = tempfile.TemporaryDirectory(prefix='golem cpus ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(REPO / 'docker', self.root / 'docker', ignore=shutil.ignore_patterns('.env', '__pycache__'))
        self.sysfs = self.root / 'sysfs'
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(('GOLEM_', 'ROS_', 'G1_'))}
        self.env['GOLEM_SYSFS_ROOT'] = str(self.sysfs)

    def fake_cpus(self, hybrid=True, online='0-27'):
        """Write a fake /sys: the i7-14700F's core and atom lists when `hybrid`, and the online list."""
        online_file = self.sysfs / 'sys/devices/system/cpu/online'
        online_file.parent.mkdir(parents=True, exist_ok=True)
        online_file.write_text(online + '\n')
        if hybrid:
            for kind, cpus in (('cpu_core', '0-15'), ('cpu_atom', '16-27')):
                p = self.sysfs / 'sys/devices' / kind / 'cpus'
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(cpus + '\n')

    def configure(self, **env):
        """Run load_config and configure_cpus; return the process, whose stdout is the result as JSON."""
        script = '''set -eo pipefail
source "$1"
load_config
configure_cpus
python3 -c 'import json, os, sys
print(json.dumps({"sim": os.environ.get("GOLEM_SIM_CPUSET"), "ros": os.environ.get("GOLEM_ROS_CPUSET"),
                  "compose": sys.argv[1:]}))' "${COMPOSE[@]}"'''
        return subprocess.run(['bash', '-c', script, 'bash', str(self.root / 'docker/scripts/docker_common.sh')],
                              cwd='/tmp', env=dict(self.env, **env), text=True, capture_output=True)

    def result(self, **env):
        """Return configure_cpus's result dict for `env`, asserting that it succeeded."""
        r = self.configure(**env)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        out['stderr'] = r.stderr
        out['overlays'] = [Path(f).name for f in out['compose'] if f.endswith('.yml') and 'cpus' in Path(f).name]
        return out

    def test_hybrid_cpu_pins_sim_to_core_and_ros_to_atom(self):
        """A hybrid CPU gives the sim cpu_core's list and ros cpu_atom's, and says so."""
        self.fake_cpus()
        r = self.result()
        self.assertEqual((r['sim'], r['ros']), ('0-15', '16-27'))
        self.assertEqual(sorted(r['overlays']), ['docker-compose.cpus-ros.yml', 'docker-compose.cpus-sim.yml'])
        self.assertIn('cpus: sim 0-15, ros 16-27', r['stderr'])

    def test_no_cpu_core_means_no_pinning(self):
        """A CPU without cpu_core and cpu_atom lists pins nothing and adds no overlay."""
        self.fake_cpus(hybrid=False)
        r = self.result()
        self.assertEqual((r['sim'], r['ros'], r['overlays']), (None, None, []))
        self.assertIn('cpus: not pinned', r['stderr'])

    def test_only_one_of_core_and_atom_means_no_pinning(self):
        """A sysfs with cpu_core but no cpu_atom is not hybrid."""
        self.fake_cpus()
        (self.sysfs / 'sys/devices/cpu_atom/cpus').unlink()
        r = self.result()
        self.assertEqual((r['sim'], r['ros']), (None, None))

    def test_pinning_off_disables_detection_and_explicit_sets(self):
        """GOLEM_CPU_PINNING=0 pins nothing, whatever the CPU and the explicit knobs say."""
        self.fake_cpus()
        for env in ({}, {'GOLEM_SIM_CPUSET': '2-3', 'GOLEM_ROS_CPUSET': '4-5'}):
            with self.subTest(env=env):
                r = self.result(GOLEM_CPU_PINNING='0', **env)
                self.assertEqual((r['sim'], r['ros'], r['overlays']), (None, None, []))
                self.assertIn('cpus: not pinned', r['stderr'])

    def test_bad_pinning_switch_is_refused(self):
        """GOLEM_CPU_PINNING other than 0 or 1 fails, naming the knob."""
        self.fake_cpus()
        r = self.configure(GOLEM_CPU_PINNING='maybe')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('GOLEM_CPU_PINNING', r.stderr)

    def test_explicit_sim_cpuset_wins_over_detection(self):
        """An explicit GOLEM_SIM_CPUSET replaces cpu_core's list; ros keeps the detected one."""
        self.fake_cpus()
        r = self.result(GOLEM_SIM_CPUSET='2-3')
        self.assertEqual((r['sim'], r['ros']), ('2-3', '16-27'))
        self.assertIn('cpus: sim 2-3, ros 16-27', r['stderr'])

    def test_explicit_cpuset_pins_without_a_hybrid_cpu(self):
        """An explicit knob pins its container on a CPU that is not hybrid; the other stays free."""
        self.fake_cpus(hybrid=False)
        r = self.result(GOLEM_SIM_CPUSET='0,2,4-5')
        self.assertEqual((r['sim'], r['ros'], r['overlays']), ('0,2,4-5', None, ['docker-compose.cpus-sim.yml']))

    def test_cpuset_outside_online_cpus_fails_naming_the_knob(self):
        """A cpuset naming a CPU that is not online fails before any container starts, naming its knob."""
        self.fake_cpus()
        for knob in ('GOLEM_SIM_CPUSET', 'GOLEM_ROS_CPUSET'):
            with self.subTest(knob=knob):
                r = self.configure(**{knob: '99'})
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(knob, r.stderr)
                self.assertIn('99', r.stderr)

    def test_cpuset_on_an_offline_cpu_fails(self):
        """A CPU inside the machine's range but not in the online list is refused."""
        self.fake_cpus(online='0-3,8-27')
        r = self.configure(GOLEM_SIM_CPUSET='4-5')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('GOLEM_SIM_CPUSET', r.stderr)

    def test_malformed_cpuset_fails_naming_the_knob(self):
        """A cpuset that is not a list of CPU numbers and ranges fails, naming its knob."""
        self.fake_cpus()
        for value in ('a', '3-1', '1-', '1,,2', '-4', '0-99999999'):
            with self.subTest(value=value):
                r = self.configure(GOLEM_ROS_CPUSET=value)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('GOLEM_ROS_CPUSET', r.stderr)

    def test_domain_zero_pins_nothing(self):
        """On ROS_DOMAIN_ID=0, the real robots' bus, a hybrid CPU still pins nothing."""
        self.fake_cpus()
        r = self.result(ROS_DOMAIN_ID='0')
        self.assertEqual((r['sim'], r['ros'], r['overlays']), (None, None, []))
        self.assertIn('cpus: not pinned', r['stderr'])

    def test_domain_zero_refuses_an_explicit_ros_cpuset(self):
        """An explicit GOLEM_ROS_CPUSET on domain 0 is refused with a message, not applied."""
        self.fake_cpus()
        r = self.configure(ROS_DOMAIN_ID='0', GOLEM_ROS_CPUSET='16-27')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('GOLEM_ROS_CPUSET', r.stderr)
        self.assertIn('domain 0', r.stderr)

    def test_domain_zero_ignores_an_explicit_sim_cpuset(self):
        """On domain 0 no sim runs, so an explicit GOLEM_SIM_CPUSET pins nothing."""
        self.fake_cpus()
        r = self.result(ROS_DOMAIN_ID='0', GOLEM_SIM_CPUSET='2-3')
        self.assertEqual((r['sim'], r['ros'], r['overlays']), (None, None, []))

    def test_mac_platform_pins_nothing(self):
        """The Mac wrappers' platform pins nothing, even with a hybrid sysfs and explicit knobs."""
        self.fake_cpus()
        r = self.result(GOLEM_PLATFORM='mac', GOLEM_SIM_CPUSET='2-3')
        self.assertEqual((r['sim'], r['ros'], r['overlays']), (None, None, []))

    def test_env_file_values_apply(self):
        """GOLEM_CPU_PINNING in docker/.env is read, and a shell value wins over it."""
        self.fake_cpus()
        (self.root / 'docker/.env').write_text('GOLEM_CPU_PINNING=0\n')
        self.assertEqual(self.result()['sim'], None)
        self.assertEqual(self.result(GOLEM_CPU_PINNING='1')['sim'], '0-15')


@unittest.skipUnless(DOCKER, 'docker compose is needed to render the compose files')
class CpuPinningCompose(unittest.TestCase):
    """Render the compose files with `docker compose config`."""

    def services(self, overlays=(), **env):
        """Return the `config` of the robocasa and ros services with the given overlays and env."""
        docker = REPO / 'docker'
        cmd = [DOCKER, 'compose', '--env-file', '/dev/null', '-f', str(docker / 'docker-compose.yml')]
        for overlay in overlays:
            cmd += ['-f', str(docker / overlay)]
        cmd += ['--profile', 'robocasa', '--profile', 'ros', 'config', '--format', 'json']
        clean = {k: v for k, v in os.environ.items() if not k.startswith(('GOLEM_', 'ROS_'))}
        r = subprocess.run(cmd, env=dict(clean, **env), text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)['services']

    def test_unpinned_services_have_no_cpuset_key(self):
        """With no overlay, neither service has a cpuset key."""
        services = self.services()
        for name in ('robocasa', 'ros'):
            self.assertNotIn('cpuset', services[name])

    def test_overlays_set_each_services_cpuset(self):
        """The sim overlay sets robocasa's cpuset from GOLEM_SIM_CPUSET and the ros overlay ros's from GOLEM_ROS_CPUSET."""
        services = self.services(('docker-compose.cpus-sim.yml', 'docker-compose.cpus-ros.yml'),
                                 GOLEM_SIM_CPUSET='0-15', GOLEM_ROS_CPUSET='16-27')
        self.assertEqual(services['robocasa']['cpuset'], '0-15')
        self.assertEqual(services['ros']['cpuset'], '16-27')

    def test_one_overlay_leaves_the_other_service_free(self):
        """Only the overlay that is added pins its service."""
        services = self.services(('docker-compose.cpus-sim.yml',), GOLEM_SIM_CPUSET='2-3')
        self.assertEqual(services['robocasa']['cpuset'], '2-3')
        self.assertNotIn('cpuset', services['ros'])


if __name__ == '__main__':
    unittest.main()
