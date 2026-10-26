'''Processes for the multi-process tests: DDS on loopback only, domain 97.

Every process the tests start gets TEST_DOMAIN and a CycloneDDS config that
discovers peers by unicast on 127.0.0.1 only, so nothing leaves the machine
and no multicast (or NET_ADMIN) is needed.
'''

import os
import signal
import subprocess
import time

import yaml
from ament_index_python.packages import get_package_prefix

from g1_description.paths import urdf_path

TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
HERE = os.path.dirname(os.path.abspath(__file__))

DDS_CONFIG = '''<CycloneDDS><Domain Id="any">
  <General>
    <Interfaces><NetworkInterface name="lo"/></Interfaces>
    <AllowMulticast>false</AllowMulticast>
  </General>
  <Discovery>
    <ParticipantIndex>auto</ParticipantIndex>
    <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
    <Peers><Peer address="127.0.0.1"/></Peers>
  </Discovery>
</Domain></CycloneDDS>
'''


def use_loopback_dds(tmp_dir):
    '''Point this process (before rclpy.init) and its children at TEST_DOMAIN on loopback'''
    path = os.path.join(tmp_dir, 'cyclonedds.xml')
    with open(path, 'w') as f:
        f.write(DDS_CONFIG)
    os.environ['CYCLONEDDS_URI'] = 'file://' + path
    os.environ['ROS_DOMAIN_ID'] = str(TEST_DOMAIN)
    os.environ['ROS_LOCALHOST_ONLY'] = '0'
    os.environ['RCUTILS_LOGGING_BUFFERED_STREAM'] = '1'


class Processes:
    '''Started in their own process groups, logged to files, stopped in reverse order'''

    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir
        self.running = []

    def start(self, name, cmd):
        log = open(os.path.join(self.tmp_dir, name + '.log'), 'w')
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.running.append((name, proc, log))
        return proc

    def robot_state_publisher(self):
        with open(urdf_path('with_hand', ros=True)) as f:
            description = f.read()
        params = {'robot_state_publisher': {'ros__parameters': {'robot_description': description}}}
        path = os.path.join(self.tmp_dir, 'robot_state_publisher.yaml')
        with open(path, 'w') as f:
            yaml.safe_dump(params, f)
        # the executable itself, not ros2 run, so that SIGINT reaches the node
        executable = os.path.join(get_package_prefix('robot_state_publisher'), 'lib',
                                  'robot_state_publisher', 'robot_state_publisher')
        return self.start('robot_state_publisher',
                          [executable, '--ros-args', '--params-file', path])

    def fake_g1(self, *args):
        return self.start('fake_g1', ['python3', os.path.join(HERE, 'fake_g1.py'), *args])

    def navigation(self, *args):
        return self.start('navigation', [
            'ros2', 'launch', 'g1_navigation', 'g1_navigation.launch.py', *args])

    def check_alive(self):
        for name, proc, _ in self.running:
            assert proc.poll() is None, f'{name} exited with {proc.returncode}: {self.tail(name)}'

    def tail(self, name, lines=40):
        with open(os.path.join(self.tmp_dir, name + '.log')) as f:
            return ''.join(f.readlines()[-lines:])

    def stop_all(self):
        for name, proc, log in reversed(self.running):
            _stop(proc)
            log.close()
        self.running = []


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _stop(proc):
    # SIGINT to the leader only (ros2 launch passes it on to its nodes once),
    # then TERM to the group; KILL whatever is left in the group
    # (start_new_session made the leader's pid its id)
    for signal_it, wait in ((lambda: proc.send_signal(signal.SIGINT), 15.0),
                            (lambda: _killpg(proc.pid, signal.SIGTERM), 5.0)):
        if proc.poll() is not None:
            break
        signal_it()
        deadline = time.time() + wait
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.1)
    _killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5.0)
