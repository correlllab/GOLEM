"""SelfFilter skips links that never had a TF and drops clouds when a known link loses its."""
from types import SimpleNamespace

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import TransformException

from golem_slam.self_filter import cloud_points, SelfFilter

_BOX = (np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))


class _Logger:

    def __init__(self):
        self.warnings = []

    def warn(self, text, **kwargs):
        self.warnings.append(text)

    def info(self, text, **kwargs):
        pass


def _filter(links, resolvable):
    """Build a SelfFilter stand-in; `resolvable` holds, per cloud, the links that have a TF."""
    clouds = iter(resolvable)
    current = {}

    def lookup(frame, link, stamp):
        if link not in current['links']:
            raise TransformException(f'no TF for {link}')
        return SimpleNamespace(transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    published = []
    node = SimpleNamespace(
        boxes={link: _BOX for link in links}, seen_links=set(),
        reported_unseen=frozenset(), logger=_Logger(), _lookup=lookup,
        pub=SimpleNamespace(publish=published.append))
    node.get_logger = lambda: node.logger

    def on_cloud(msg):
        current['links'] = next(clouds)
        SelfFilter._on_cloud(node, msg)

    return node, on_cloud, published


def _cloud():
    msg = PointCloud2(header=Header(frame_id='odom'), height=1, width=2, point_step=12,
                      row_step=24, is_dense=True)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
    msg.data = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype='<f4').tobytes()
    return msg


def test_never_seen_links_are_skipped():
    node, on_cloud, published = _filter(['pelvis', 'finger'], [['pelvis']])
    on_cloud(_cloud())
    assert len(published) == 1
    _, xyz = cloud_points(published[0])
    np.testing.assert_allclose(xyz, [[1.0, 0.0, 0.0]])


def test_lost_link_drops_cloud():
    node, on_cloud, published = _filter(['pelvis', 'arm'], [['pelvis', 'arm'], ['pelvis']])
    on_cloud(_cloud())
    assert len(published) == 1
    on_cloud(_cloud())
    assert len(published) == 1


def test_warning_once():
    node, on_cloud, published = _filter(['pelvis', 'finger'], [['pelvis']] * 3)
    for _ in range(3):
        on_cloud(_cloud())
    assert len(published) == 3
    assert len(node.logger.warnings) == 1
    assert 'finger' in node.logger.warnings[0]


def test_warning_again_when_the_set_changes():
    node, on_cloud, _ = _filter(['pelvis', 'a', 'b'],
                                [['pelvis'], ['pelvis'], ['pelvis', 'a']])
    for _ in range(3):
        on_cloud(_cloud())
    assert len(node.logger.warnings) == 2
    assert node.logger.warnings[0].endswith('a, b')
    assert node.logger.warnings[1].endswith('b')
