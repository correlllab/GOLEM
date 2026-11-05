"""Check that every image builds from pinned inputs: base digests, commits and locked Python packages."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted((ROOT / 'docker').glob('*Dockerfile')) + sorted((ROOT / 'docker/mac').glob('*Dockerfile*'))

class SourcePinTests(unittest.TestCase):
    """Check the Dockerfiles' base images, clones and downloads are pinned."""

    def test_public_base_images_have_digests(self):
        """Every public base image is pinned by digest."""
        for path in FILES:
            for image in re.findall(r'^FROM (\S+)', path.read_text(), re.M):
                if not image.startswith('golem_base:'):
                    self.assertRegex(image, r'@sha256:[0-9a-f]{64}$', str(path))

    def test_every_git_clone_is_checked_out_at_an_exact_commit(self):
        """Every git clone in a RUN step is checked out at a full commit hash."""
        for path in FILES:
            text = path.read_text().replace('\\\n', ' ')
            args = dict(re.findall(r'^ARG (\w+)=([0-9a-f]{40})$', text, re.M))
            for name, sha in args.items():
                text = text.replace('${' + name + '}', sha)
            for run in re.findall(r'^RUN .*', text, re.M):
                clones = re.findall(r'git clone\b', run)
                if clones:
                    pins = re.findall(r'git (?:-C \S+ )?(?:checkout(?: --quiet)?|reset --hard) "?[0-9a-f]{40}\b', run)
                    self.assertEqual(len(clones), len(pins), f'{path}: {run}')

    def test_every_image_builds_livox_sdk2_at_one_commit(self):
        """Every image that builds Livox-SDK2 checks out the same commit, so they agree on the CustomMsg wire format."""
        refs = {}
        for path in FILES:
            text = path.read_text().replace('\\\n', ' ')
            for run in re.findall(r'^RUN .*', text, re.M):
                if 'Livox-SDK2.git' in run:
                    pin = re.search(r'checkout(?: --quiet)? ([0-9a-f]{40})\b', run)
                    refs[path.relative_to(ROOT).as_posix()] = pin and pin.group(1)
        self.assertLessEqual({'docker/RosDockerfile', 'docker/RobocasaDockerfile',
                              'docker/mac/RobocasaDockerfile.arm64',
                              'docker/mac/RosDockerfile.slim.arm64'}, set(refs))
        self.assertEqual(len(set(refs.values())), 1, refs)

    def test_isaaclab_transitive_vcs_pins_are_retained(self):
        """The Isaac image pins rl_games and robomimic to commits, before Isaac Lab's install."""
        text = (ROOT / 'docker/IsaacDockerfile').read_text()
        for pin in (
            'rl_games.git@6b3534f29568158e9e29ec8bf83cc88fce5f0cae',
            'robomimic.git@7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9',
        ):
            self.assertIn(pin, text)
        self.assertLess(text.index('rl_games.git@6b3534f'), text.index('isaaclab.sh --install'))

    def test_isaac_locks_stay_compatible_with_bundled_packages(self):
        """The Isaac locks keep torch, stable-baselines3, ipython and psutil compatible with what Isaac Sim bundles."""
        # Isaac Sim bundles torch 2.7.0 and psutil 5.9.8 (rl-games needs psutil<6);
        # stable-baselines3 2.9+ needs torch>=2.8 and ipython 9.13+ needs psutil>=7.
        locks = ROOT / 'docker/locks'
        pins = dict(re.findall(r'^([\w.-]+)==(\S+)$', (locks / 'isaac-amd64.txt').read_text(), re.M))
        self.assertEqual(pins['torch'], '2.7.0+cu128')
        self.assertEqual(pins['stable-baselines3'], '2.8.0')
        self.assertEqual(pins['ipython'], '9.10.1')
        for path in sorted(locks.glob('isaac-amd64-*.txt')):
            self.assertIn('\npsutil==5.9.8\n', path.read_text(), str(path))

    def test_bare_ubuntu_bases_have_a_ca_bundle_before_https_snapshots(self):
        """Bare Ubuntu stages copy a CA bundle in before they switch apt to HTTPS snapshots."""
        for path in FILES:
            text = path.read_text()
            if re.search(r'^FROM ubuntu:', text, re.M) and 'apt-sources.sh' in text:
                copy = text.find('COPY --from=ca_bundle /etc/ssl/certs/ca-certificates.crt')
                self.assertNotEqual(copy, -1, str(path))
                self.assertLess(copy, text.index('RUN sh /opt/golem-locks/apt-sources.sh'), str(path))

    def test_ros_removes_base_kinematics_closure_before_reinstall(self):
        """The ROS image uninstalls every package of the base image's Pinocchio closure before reinstalling it."""
        # The base image's pin 4 links cmeel-boost 1.90 (NumPy 2); the ROS locks pin 2.7 with boost 1.83.
        text = (ROOT / 'docker/RosDockerfile').read_text().replace('\\\n', ' ')
        uninstall = re.search(r'^RUN pip uninstall -y (.*)$', text, re.M).group(1).split()
        base = (ROOT / 'docker/locks/base-amd64.txt').read_text()
        for name in re.findall(r'^((?:cmeel[\w-]*|pin|libpinocchio|coal|libcoal|eigenpy))==', base, re.M):
            self.assertIn(name, uninstall)

    def test_downloaded_scripts_do_not_follow_moving_refs(self):
        """No Dockerfile downloads the uv installer or the ROS key from a moving URL."""
        for path in FILES:
            text = path.read_text()
            self.assertNotIn('https://astral.sh/uv/install.sh', text, str(path))
            self.assertNotIn('ros/rosdistro/master/ros.key', text, str(path))

class PythonLockTests(unittest.TestCase):
    """Check every pip install goes through a committed lock file of exact pins."""

    def test_python_installs_use_committed_constraints(self):
        """Every pip install uses a committed lock file under /opt/golem-locks."""
        for path in FILES:
            text = path.read_text().replace('\\\n', ' ')
            for line in text.splitlines():
                if line.startswith('RUN ') and 'pip install' in line:
                    self.assertEqual(len(re.findall(r'pip install\b', line)),
                                     len(re.findall(r'pip install -c /opt/golem-locks/', line)), str(path))
            for name in re.findall(r'/opt/golem-locks/([\w.-]+)', text):
                self.assertTrue((ROOT/'docker/locks'/name).is_file(), name)

    def test_lock_requirements_are_exact(self):
        """Every lock file line is an exact == pin or a -c of another existing lock."""
        for path in (ROOT/'docker/locks').glob('*.txt'):
            for line in path.read_text().splitlines():
                if not line or line.startswith('#'):
                    continue
                if line.startswith('-c '):
                    self.assertTrue((path.parent/line[3:]).is_file())
                else:
                    self.assertRegex(line, r'^[a-z0-9.-]+==[^ ;]+$', str(path))

if __name__ == '__main__':
    unittest.main()
