"""
Draw /skill/grasp's candidates and target in RViz.

GraspMarkers is a mixin of GraspSkill: it publishes the candidates on
'graspgen_markers', coloured by their state in the approach walk, and
broadcasts TARGET_FRAME at the pose being driven to. Drawing is best-effort
and never fails a grasp.
"""

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from ..base import WORLD_FRAME
from ..grasp_synthesis.common import GRIPPER_BASE_TO_CONTACT_M
from ..perception_utils import pose_to_matrix


# TF frame (RViz) of the target being driven to
TARGET_FRAME = 'graspgenx_target_frame'
# most candidates drawn on 'graspgen_markers'
KEPT_MARKER_MAX = 20
# candidate state -> marker (r, g, b, a); untried candidates are 'pending'
GRASP_MARKER_COLORS = {
    'pending': (0.5, 0.5, 0.0, 0.4),   # dim yellow
    'trying': (0.0, 0.6, 1.0, 1.0),    # blue: approach in progress
    'failed': (1.0, 0.0, 0.0, 0.8),    # red: pre-grasp or contact not reached
    'chosen': (0.0, 1.0, 0.0, 1.0),    # green: committed grasp
}


class GraspMarkers:
    """Publish the grasp skill's RViz markers and target frame."""

    def _publish_target_tf(self, targets, pose_w, pose_p):
        """Broadcast TARGET_FRAME at `pose_w` with world-anchored targets, else at `pose_p`."""
        parent, pose = ((WORLD_FRAME, pose_w) if targets.have_world
                        else (targets.frame, pose_p))
        self.publish_tf(approach_target_tf(
            parent, pose, self.get_clock().now().to_msg()))

    def _publish_grasp_markers(self, frame, poses, scores, status=None):
        """
        Publish the grasp candidates on 'graspgen_markers' (latched), at most KEPT_MARKER_MAX.

        Every candidate in `status` (index -> GRASP_MARKER_COLORS state)
        first, then the rest best-score-first as 'pending'. Per grasp: an
        ARROW from the gripper-base origin along +Z to the finger contact
        point, a SPHERE there, and a TEXT label with the candidate index (the
        'grasp {i}' of the log lines). Marker ids are candidate indices, so a
        state change replaces that grasp's markers.
        """
        status = status or {}
        try:
            if not hasattr(self, '_grasp_marker_pub'):
                self._grasp_marker_pub = self.create_publisher(
                    MarkerArray, 'graspgen_markers',
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            order = sorted(range(len(poses)),
                           key=lambda i: (i not in status, -scores[i]))
            # zero stamp = latest available TF; a now() stamp races the
            # pelvis -> map link and RViz drops the markers
            stamp = Time()
            arr = MarkerArray()
            clear = Marker()
            clear.header.frame_id = frame
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            for i in order[:KEPT_MARKER_MAX]:
                T = pose_to_matrix(poses[i])
                base = T[:3, 3]
                contact = base + GRIPPER_BASE_TO_CONTACT_M * T[:3, 2]
                color = GRASP_MARKER_COLORS[status.get(i, 'pending')]
                arr.markers.extend(_candidate_markers(frame, stamp, i, base, contact, color))
            self._grasp_marker_pub.publish(arr)
        except Exception as e:                       # noqa: BLE001 - viz only
            self.get_logger().warn(f'grasp marker publish failed: {e}')


def _candidate_markers(frame, stamp, i, base, contact, color):
    """Return candidate `i`'s arrow (base to contact), contact sphere and index label."""
    r, g, b, a = color
    arrow = _marker(frame, stamp, 'graspgen_approach', i, Marker.ARROW, (r, g, b, a))
    arrow.points = [Point(x=float(base[0]), y=float(base[1]), z=float(base[2])),
                    Point(x=float(contact[0]), y=float(contact[1]), z=float(contact[2]))]
    arrow.scale.x = 0.008                  # m, shaft diameter
    arrow.scale.y = 0.018                  # m, head diameter
    arrow.scale.z = 0.03                   # m, head length

    dot = _marker(frame, stamp, 'graspgen_contact', i, Marker.SPHERE, (r, g, b, a))
    dot.pose.position = Point(x=float(contact[0]), y=float(contact[1]), z=float(contact[2]))
    dot.scale.x = dot.scale.y = dot.scale.z = 0.02    # m

    label = _marker(frame, stamp, 'graspgen_index', i, Marker.TEXT_VIEW_FACING, (r, g, b, 1.0))
    label.pose.position = Point(x=float(base[0]), y=float(base[1]), z=float(base[2]) + 0.02)
    label.scale.z = 0.025                  # m, text height
    label.text = str(i)
    return arrow, dot, label


def _marker(frame, stamp, ns, i, kind, color):
    """Return an ADD Marker of `kind` with identity orientation and `color` (r, g, b, a)."""
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = ns
    m.id = i
    m.type = kind
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def approach_target_tf(parent_frame, pose, stamp, child=TARGET_FRAME):
    """Build a debug TransformStamped placing `child` (default TARGET_FRAME) at `pose`."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child
    t.transform.translation.x = pose.position.x
    t.transform.translation.y = pose.position.y
    t.transform.translation.z = pose.position.z
    t.transform.rotation = pose.orientation
    return t
