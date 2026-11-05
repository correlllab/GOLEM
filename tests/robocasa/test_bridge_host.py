"""Check the mujoco_sensors node: /clock, the opt-in ground-truth odom, and the sensor suite it ticks.

rclpy's Node is patched, so no ROS node or DDS participant starts; the publishers record what
they are given and whether the sim lock is held meanwhile.
"""
import threading
import unittest
from unittest.mock import patch

import mujoco
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage

from mujoco_ros_bridge import RosSensorBridge, _sim_time_to_msg

SCENE = '''<mujoco>
  <worldbody>
    <body name="pelvis" pos=".3 -.2 1.05" euler="0 0 30">
      <freejoint/>
      <geom type="sphere" size=".1"/>
    </body>
  </worldbody>
</mujoco>'''
PELVIS = 1


class Publisher:
    """Record each published message with its topic and whether the sim lock was held."""

    def __init__(self, topic, events, lock):
        """Record into `events`, watching `lock`."""
        self.topic, self.events, self.lock = topic, events, lock

    def publish(self, msg):
        """Record one message."""
        self.events.append((self.topic, msg, self.lock.locked()))


class Suite:
    """Stand in for a robot's sensor suite, recording its calls and the sim lock's state."""

    def __init__(self, node, events, lock):
        """Keep the host node and record into `events`, watching `lock`."""
        self.node, self.events, self.lock = node, events, lock

    def tick(self, sim_t, stamp):
        """Record a tick with its sim time and stamp."""
        self.events.append(('suite', (sim_t, stamp), self.lock.locked()))

    def shutdown(self):
        """Record the shutdown."""
        self.events.append(('shutdown', None, self.lock.locked()))

    def stats(self):
        """Return a fixed stats line."""
        return 'lidar 12 scans'


class LockWatchedData:
    """Wrap MjData so that each attribute read records whether the sim lock is held."""

    def __init__(self, data, lock, reads):
        """Wrap `data`, recording (attribute, lock held) into `reads`."""
        object.__setattr__(self, '_data', data)
        object.__setattr__(self, '_lock', lock)
        object.__setattr__(self, '_reads', reads)

    def __getattr__(self, name):
        """Record the read and return the wrapped attribute."""
        self._reads.append((name, self._lock.locked()))
        return getattr(self._data, name)

    def __setattr__(self, name, value):
        """Set the wrapped attribute."""
        setattr(self._data, name, value)


class HostTests(unittest.TestCase):
    """Run RosSensorBridge on a one-body scene with rclpy's Node patched."""

    def setUp(self):
        """Build the scene at sim time 1.25 s and patch Node to record what it creates."""
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.data.time = 1.25
        self.lock = threading.Lock()
        self.events = []
        self.created = []      # (kind, name, type) of every entity a node creates
        self.node_names = []
        test = self

        def init(node, node_name, **kwargs):
            """Record the node's name instead of creating a node."""
            test.node_names.append(node_name)

        def create_publisher(node, msg_type, topic, qos, **kwargs):
            """Record the publisher and return a recording Publisher."""
            test.created.append(('publisher', topic, msg_type))
            return Publisher(topic, test.events, test.lock)

        def create(kind):
            """Return a stand-in for Node.create_<kind> that records what it creates."""
            def record(node, msg_type, name, *args, **kwargs):
                """Record one created entity."""
                test.created.append((kind, name, msg_type))
            return record

        def create_timer(node, period, callback, **kwargs):
            """Record the timer's period."""
            test.created.append(('timer', period, None))

        fakes = {'__init__': init, 'create_publisher': create_publisher,
                 'create_subscription': create('subscription'), 'create_service': create('service'),
                 'create_client': create('client'), 'create_timer': create_timer}
        for name, fake in fakes.items():
            patcher = patch.object(Node, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def host(self, data=None, sim_lock='default', **kwargs):
        """Make the host with a recording Suite and check it hands the suite itself as the node."""
        made = []

        def make_sensors(node):
            """Record the node and return a recording Suite."""
            made.append(node)
            return Suite(node, self.events, self.lock)

        host = RosSensorBridge(self.model, self.data if data is None else data, make_sensors,
                               sim_lock=self.lock if sim_lock == 'default' else sim_lock, **kwargs)
        self.assertEqual(made, [host])
        return host

    def names(self):
        """Return the topics or calls recorded so far, in order."""
        return [e[0] for e in self.events]

    def test_without_odom_the_host_creates_only_clock(self):
        """Without odom, the host is the mujoco_sensors node with a /clock publisher only."""
        host = self.host()
        self.assertEqual(self.node_names, ['mujoco_sensors'])
        self.assertEqual(self.created, [('publisher', '/clock', Clock)])
        self.assertIs(host.suite.node, host)

    def test_with_odom_the_host_creates_clock_odom_and_tf(self):
        """With a base body, the host also publishes /odom and /tf."""
        self.host(base_body_id=PELVIS)
        self.assertEqual(self.node_names, ['mujoco_sensors'])
        self.assertEqual(sorted(self.created, key=lambda c: c[1]),
                         [('publisher', '/clock', Clock), ('publisher', '/odom', Odometry),
                          ('publisher', '/tf', TFMessage)])

    def test_tick_publishes_clock_then_odom_then_ticks_the_suite(self):
        """A tick publishes /clock, then the odom and its TF at 50 Hz of sim time, then ticks the suite."""
        host = self.host(base_body_id=PELVIS)
        host.tick()
        self.assertEqual(self.names(), ['/clock', '/tf', '/odom', 'suite'])
        stamp = _sim_time_to_msg(1.25)
        clock, tf, odom, suite = (e[1] for e in self.events)
        self.assertEqual(clock.clock, stamp)
        self.assertEqual(suite, (1.25, stamp))
        self.assertIsInstance(suite[0], float)
        (edge,) = tf.transforms
        self.assertEqual((edge.header.stamp, edge.header.frame_id, edge.child_frame_id),
                         (stamp, 'odom', 'pelvis'))
        self.assertEqual((odom.header.stamp, odom.header.frame_id, odom.child_frame_id),
                         (stamp, 'odom', 'pelvis'))
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        self.assertEqual([p.x, p.y, p.z], self.data.xpos[PELVIS].tolist())
        self.assertEqual([q.w, q.x, q.y, q.z], self.data.xquat[PELVIS].tolist())
        t = edge.transform
        self.assertEqual([t.translation.x, t.translation.y, t.translation.z], [p.x, p.y, p.z])
        self.assertEqual([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w], [q.x, q.y, q.z, q.w])
        # Odom follows its own 50 Hz sim-time schedule; /clock and the suite run every tick.
        self.events.clear()
        self.data.time = 1.26
        host.tick()
        self.assertEqual(self.names(), ['/clock', 'suite'])
        self.events.clear()
        self.data.time = 1.271
        host.tick()
        self.assertEqual(self.names(), ['/clock', '/tf', '/odom', 'suite'])

    def test_without_base_body_tick_never_publishes_odom(self):
        """Without a base body no tick publishes odom."""
        host = self.host()
        for t in (1.25, 1.3, 2.0):
            self.events.clear()
            self.data.time = t
            host.tick()
            self.assertEqual(self.names(), ['/clock', 'suite'])

    def test_sim_lock_is_held_for_the_reads_and_never_while_publishing(self):
        """With a sim lock, the host reads MjData under it and publishes outside it."""
        reads = []
        host = self.host(data=LockWatchedData(self.data, self.lock, reads), base_body_id=PELVIS)
        reads.clear()
        host.tick()
        self.assertEqual(self.names(), ['/clock', '/tf', '/odom', 'suite'])
        self.assertEqual({name for name, _ in reads}, {'time', 'xpos', 'xquat'})
        self.assertTrue(all(locked for _, locked in reads), reads)
        # Neither publishing nor the suite's tick (which takes its own locks) runs under it.
        self.assertFalse(any(locked for _, _, locked in self.events), self.events)

    def test_without_sim_lock_tick_runs_inside_the_callers_lock(self):
        """Without a sim lock, the whole tick runs under the caller's lock and takes no lock itself."""
        # The G1 runner holds its own, non-reentrant lock around tick() and passes no sim lock.
        reads = []
        host = self.host(data=LockWatchedData(self.data, self.lock, reads), sim_lock=None,
                         base_body_id=PELVIS)
        reads.clear()
        with self.lock:
            host.tick()
        self.assertEqual(self.names(), ['/clock', '/tf', '/odom', 'suite'])
        self.assertEqual({name for name, _ in reads}, {'time', 'xpos', 'xquat'})
        self.assertTrue(all(locked for _, locked in reads), reads)
        self.assertTrue(all(locked for _, _, locked in self.events), self.events)
        self.assertFalse(self.lock.locked())

    def test_shutdown_and_stats_delegate_to_the_suite(self):
        """Shutdown and stats go to the suite."""
        host = self.host()
        self.assertEqual(host.stats(), 'lidar 12 scans')
        host.shutdown()
        self.assertEqual(self.names(), ['shutdown'])


if __name__ == '__main__':
    unittest.main()
