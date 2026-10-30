"""Point-cloud leveling preserves stamps/fields and waits for exact TF."""
import importlib
from collections import deque
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import Buffer


def test_cloud_waits_for_transform_and_preserves_fields():
    assert importlib.util.find_spec('golem_slam.level_cloud'), 'level cloud adapter is missing'
    from golem_slam.level_cloud import LevelCloud
    from golem_slam.self_filter import cloud_points
    published = []
    node = SimpleNamespace(target_frame='base_footprint', buffer=Buffer(), pending=deque(),
                           publisher=SimpleNamespace(publish=published.append))
    msg = PointCloud2(header=Header(frame_id='body'), height=1, width=1, point_step=16,
                      row_step=16, is_dense=True)
    msg.header.stamp.sec = 5
    msg.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
    msg.data = np.array([1.0, 2.0, -0.5, 42.0], dtype='<f4').tobytes()
    node.pending.append(msg)
    LevelCloud._drain(node)
    assert not published and len(node.pending) == 1
    tf = TransformStamped(header=msg.header, child_frame_id='base_footprint')
    tf.transform.rotation.w = 1.0
    tf.transform.translation.z = -1.0
    node.buffer.set_transform(tf, 'test')
    LevelCloud._drain(node)
    assert not node.pending and len(published) == 1
    result = published[0]
    assert result.header.frame_id == 'base_footprint'
    assert result.header.stamp == msg.header.stamp
    fields, xyz = cloud_points(result)
    np.testing.assert_allclose(xyz, [[1.0, 2.0, 0.5]])
    assert fields['intensity'][0] == 42.0
    assert msg.header.frame_id == 'body'  # no mutation of the input cloud
