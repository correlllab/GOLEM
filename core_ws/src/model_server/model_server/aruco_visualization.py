"""Observation-derived board geometry only; no TF broadcasts or simulator truth."""

import json
from pathlib import Path

import cv2
import numpy as np

from model_server.aruco_geometry import marker_points


BOARD_CORNERS = np.array([[-.16, -.125, .025], [.16, -.125, .025],
                          [.16, .125, .025], [-.16, .125, .025]], dtype=float)


def segments(points):
    return np.array([p for a, b in zip(points, np.roll(points, -1, axis=0)) for p in (a, b)])


def annotate(rgb, optical, k, d, corners, ids):
    result = rgb.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(result, corners, ids)
    camera = BOARD_CORNERS @ optical[:3, :3].T + optical[:3, 3]
    if np.all(camera[:, 2] > 0):
        pixels, _ = cv2.projectPoints(BOARD_CORNERS, cv2.Rodrigues(optical[:3, :3])[0],
                                     optical[:3, 3], k, d)
        if np.isfinite(pixels).all() and np.abs(pixels).max() < 1e6:
            cv2.polylines(result, [np.round(pixels).astype(np.int32)], True, (40, 255, 80), 2)
    return result


class BoardVisualization:
    def __init__(self, node, layout, catalog_path=''):
        from visualization_msgs.msg import MarkerArray
        self.publisher = node.create_publisher(MarkerArray, '/perception/pegboard/markers', 2)
        self.layout = layout
        self.hole = None
        if catalog_path:
            catalog = json.loads(Path(catalog_path).read_text())
            self.hole = np.array(catalog['holes']['cylinder_3']['entry'], dtype=float)
        self.observation = None

    def publish(self, valid, reason, now, target_frame):
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        message = MarkerArray()

        def marker(index, kind, frame, stamp):
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, stamp
            m.ns, m.id, m.type, m.action = 'pegboard_visualization', index, kind, Marker.ADD
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=0.15, g=1.0, b=0.3, a=1.0)
            message.markers.append(m)
            return m

        status = marker(0, Marker.TEXT_VIEW_FACING, target_frame, now)
        status.pose.position.z = .55
        status.scale.z = .035
        status.color = ColorRGBA(r=0.15 if valid else 1.0, g=1.0 if valid else .3, b=.2, a=1.0)
        status.text = (('VALID: ' if valid else 'INVALID: ') + reason[:180] +
                       ('\nAxes: cylinder_3 entry (catalog reference)' if self.hole is not None
                        else '\nAxes: board origin'))
        if valid and self.observation is not None:
            header, target, k, d, width, height = self.observation

            def lines(index, points, frame=target_frame, color=None):
                m = marker(index, Marker.LINE_LIST, frame, header.stamp)
                m.scale.x = .003
                m.lifetime.sec, m.lifetime.nanosec = 0, 500_000_000
                if color is not None:
                    m.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=1.0)
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]
                return m

            def transformed(points):
                return np.asarray(points) @ target[:3, :3].T + target[:3, 3]

            lines(1, transformed(segments(BOARD_CORNERS)))
            lines(2, transformed(np.concatenate([segments(marker_points(m)) for m in self.layout['markers']])),
                  color=(1., .8, .1))
            origin = np.zeros(3) if self.hole is None else self.hole
            for axis in range(3):
                end = origin.copy()
                end[axis] += .07
                lines(3 + axis, transformed([origin, end]), color=np.eye(3)[axis])
            pixels = np.array([[[0., 0.]], [[width-1., 0.]], [[width-1., height-1.]], [[0., height-1.]]])
            normalized = cv2.undistortPoints(pixels, k, d).reshape(-1, 2)
            rays = np.column_stack((normalized, np.ones(4))) * .5
            frustum = np.concatenate((segments(rays), np.array([p for r in rays for p in (np.zeros(3), r)])))
            lines(6, frustum, frame=header.frame_id, color=(.2, .7, 1.))
        else:
            # DELETE keeps RViz from presenting a rejected previous pose as valid.
            for index in range(1, 7):
                m = marker(index, Marker.LINE_LIST, target_frame, now)
                m.action = Marker.DELETE
        self.publisher.publish(message)
