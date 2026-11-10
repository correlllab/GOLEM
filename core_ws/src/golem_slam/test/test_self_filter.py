"""Test self_filter: it skips links that never had a TF and drops clouds when one loses it."""
from types import SimpleNamespace

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformException

from golem_slam.self_filter import cloud_points, SelfFilter

_BOX = (np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))


class _Logger:
    """Record warnings; ignore info."""

    def __init__(self):
        """Start with no warnings."""
        self.warnings = []

    def warn(self, text, **kwargs):
        """Record a warning."""
        self.warnings.append(text)

    def info(self, text, **kwargs):
        """Ignore an info message."""


def _filter(links, resolvable):
    """Return (stand-in node, cloud callback, published); resolvable: each cloud's TF links."""
    clouds = iter(resolvable)
    current = {}

    def lookup(frame, link, stamp):
        """Return an identity transform for a link the current cloud has; else raise."""
        if link not in current['links']:
            raise TransformException(f'no TF for {link}')
        return SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    published = []
    node = SimpleNamespace(
        boxes={link: _BOX for link in links}, seen_links=set(), first_tf_stamp=None,
        unseen_link_grace=2.0,
        reported_unseen=frozenset(), logger=_Logger(), _lookup=lookup,
        pub=SimpleNamespace(publish=published.append))
    node.get_logger = lambda: node.logger

    def on_cloud(msg):
        """Run SelfFilter._on_cloud with the next cloud's TF links."""
        current['links'] = next(clouds)
        SelfFilter._on_cloud(node, msg)

    return node, on_cloud, published


def _cloud(sec=100):
    """Return a two-point cloud stamped sec: one point at the origin, one 1 m out."""
    msg = PointCloud2(header=Header(frame_id='odom'), height=1, width=2, point_step=12,
                      row_step=24, is_dense=True)
    msg.header.stamp.sec = sec
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
    msg.data = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype='<f4').tobytes()
    return msg


def test_never_seen_links_are_skipped():
    """A link that never had a TF is skipped once the grace has passed."""
    node, on_cloud, published = _filter(['pelvis', 'finger'], [['pelvis']] * 2)
    on_cloud(_cloud(100))
    on_cloud(_cloud(103))
    assert len(published) == 1
    _, xyz = cloud_points(published[0])
    np.testing.assert_allclose(xyz, [[1.0, 0.0, 0.0]])


def test_no_link_ever_seen_drops_cloud():
    """No cloud goes out while no link has ever had a TF."""
    node, on_cloud, published = _filter(['pelvis', 'finger'], [[]] * 3)
    for sec in (100, 103, 200):
        on_cloud(_cloud(sec))
    assert not published


def test_unseen_link_dropped_during_grace_then_skipped():
    """Clouds are dropped during the grace, then the unseen link is skipped."""
    node, on_cloud, published = _filter(['pelvis', 'finger'], [['pelvis']] * 3)
    on_cloud(_cloud(100))
    on_cloud(_cloud(101))
    assert not published
    on_cloud(_cloud(102))
    assert len(published) == 1
    _, xyz = cloud_points(published[0])
    np.testing.assert_allclose(xyz, [[1.0, 0.0, 0.0]])


def test_lost_link_drops_cloud():
    """A link that had a TF and lost it drops the cloud."""
    node, on_cloud, published = _filter(['pelvis', 'arm'], [['pelvis', 'arm'], ['pelvis']])
    on_cloud(_cloud(100))
    assert len(published) == 1
    on_cloud(_cloud(103))
    assert len(published) == 1


def test_warning_once():
    """The unseen links are reported once while they stay the same."""
    node, on_cloud, published = _filter(['pelvis', 'finger'], [['pelvis']] * 4)
    for sec in (100, 103, 104, 105):
        on_cloud(_cloud(sec))
    assert len(published) == 3
    unseen_warnings = [w for w in node.logger.warnings if 'ever come' in w]
    assert len(unseen_warnings) == 1
    assert 'finger' in unseen_warnings[0]


def test_warning_again_when_the_set_changes():
    """The unseen links are reported again when the set changes."""
    node, on_cloud, _ = _filter(['pelvis', 'a', 'b'],
                                [['pelvis'], ['pelvis'], ['pelvis', 'a']])
    for sec in (100, 103, 104):
        on_cloud(_cloud(sec))
    unseen_warnings = [w for w in node.logger.warnings if 'ever come' in w]
    assert len(unseen_warnings) == 2
    assert unseen_warnings[0].endswith('a, b')
    assert unseen_warnings[1].endswith('b')
