"""Marker and image projection tests without ROS initialization."""

import sys
import unittest
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import numpy as np

from model_server.aruco_geometry import load_layout
from model_server.aruco_visualization import BoardVisualization, annotate


class Marker:
    ADD, DELETE, LINE_LIST, TEXT_VIEW_FACING = 0, 2, 5, 9

    def __init__(self):
        self.header = NS(frame_id='', stamp=None)
        self.pose = NS(position=NS(x=0., y=0., z=0.), orientation=NS(w=0.))
        self.scale = NS(x=0., y=0., z=0.)
        self.lifetime = NS(sec=0, nanosec=0)


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        modules = {name: ModuleType(name) for name in ('visualization_msgs', 'visualization_msgs.msg',
                   'geometry_msgs', 'geometry_msgs.msg', 'std_msgs', 'std_msgs.msg')}
        modules['visualization_msgs.msg'].Marker = Marker
        modules['visualization_msgs.msg'].MarkerArray = lambda: NS(markers=[])
        modules['geometry_msgs.msg'].Point = lambda **kw: NS(**kw)
        modules['std_msgs.msg'].ColorRGBA = lambda **kw: NS(**kw)
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.node = MagicMock()
        self.visuals = BoardVisualization(self.node, load_layout())
        self.k = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1.]])
        self.header = NS(frame_id='head_color_optical_frame', stamp=NS(sec=10, nanosec=0))
        self.pose = np.eye(4)
        self.pose[2, 3] = .8
        self.visuals.observation = (self.header, self.pose, self.k, np.zeros(5), 640, 360)

    def test_geometry_and_frustum_use_observation_stamp_status_uses_now(self):
        now = NS(sec=11, nanosec=0)
        self.visuals.publish(True, 'accepted', now, 'pelvis')
        markers = self.visuals.publisher.publish.call_args.args[0].markers
        geometry = [m for m in markers if m.id and m.action == Marker.ADD]
        self.assertEqual(len(markers), len({m.id for m in markers}))
        self.assertEqual(len(geometry), 6)
        self.assertTrue(all(m.header.stamp is self.header.stamp for m in geometry))
        frustum = next(m for m in geometry if m.id == 6)
        self.assertEqual(frustum.header.frame_id, self.header.frame_id)
        self.assertEqual(len(frustum.points), 16)
        self.assertEqual(next(m for m in markers if m.id == 0).header.stamp, now)

    def test_invalid_status_removes_all_old_geometry(self):
        self.visuals.publish(False, 'stale', NS(sec=11), 'pelvis')
        markers = self.visuals.publisher.publish.call_args.args[0].markers
        self.assertEqual({m.id for m in markers if m.action == Marker.DELETE}, set(range(1, 7)))
        status = next(m for m in markers if m.id == 0)
        self.assertIn('INVALID: stale', status.text)
        self.assertEqual(status.color.r, 1.)

    def test_predicted_board_rectangle_annotates_copy(self):
        image = np.zeros((360, 640, 3), np.uint8)
        output = annotate(image, self.pose, self.k, np.zeros(5), [], None)
        self.assertTrue(output.any())
        self.assertFalse(image.any())


if __name__ == '__main__':
    unittest.main()
