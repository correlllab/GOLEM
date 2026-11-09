"""Catalog geometry and stamped board observations; no ROS graph or AI clients.

Catalog quaternions are wxyz. Peg body origin is its tip, not its centroid;
hole target is the fully seated tip (22 mm below the entry in this catalog).
"""

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from diagnostic_msgs.msg import DiagnosticStatus

from .grasp_synthesis.common import make_candidates
from .perception_utils import matrix_to_pose, pose_to_matrix


def rigid(matrix):
    """Reject malformed transforms rather than repairing unknown calibration."""
    m = np.asarray(matrix, dtype=float)
    if (m.shape != (4, 4) or not np.isfinite(m).all()
            or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError('invalid rigid transform')
    return m


def hole_transform(hole, site):
    q = np.asarray(hole['quat'], dtype=float)
    if q.shape != (4,) or not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6):
        raise ValueError('invalid catalog hole quaternion')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = hole[site]
    return rigid(t)


def fits(peg, hole):
    """Contain the actual peg polygon at the hole's catalog orientation.

    Edge sampling alone is insufficient for concave profiles: reject every
    boundary crossing as well as vertices outside the hole. Rows are NOT sizes.
    """
    p = np.asarray(peg['profile_xy'], dtype=float)
    h = np.asarray(hole['profile_xy'], dtype=float)
    t = hole_transform(hole, 'entry')
    p = p @ t[:2, :2].T + t[:2, 3]
    if not np.isfinite(p).all() or not np.isfinite(h).all():
        return False
    a, b = h, np.roll(h, -1, axis=0)
    for v in p:
        cross = ((a[:, 1] > v[1]) != (b[:, 1] > v[1]))
        x = a[:, 0] + (v[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (
            b[:, 1] - a[:, 1] + 1e-300)
        if not np.count_nonzero(cross & (v[0] < x)) % 2:
            return False
    for u, v in zip(p, np.roll(p, -1, axis=0)):
        d = v - u
        e = b - a
        den = d[0] * e[:, 1] - d[1] * e[:, 0]
        mask = np.abs(den) > 1e-14
        rel = a - u
        along = np.zeros(len(a))
        other = np.zeros(len(a))
        along[mask] = (rel[mask, 0] * e[mask, 1] - rel[mask, 1] * e[mask, 0]) / den[mask]
        other[mask] = (rel[mask, 0] * d[1] - rel[mask, 1] * d[0]) / den[mask]
        if np.any(mask & (along >= 0) & (along <= 1) & (other >= 0) & (other <= 1)):
            return False
    return True


@dataclass(frozen=True)
class TargetGeometry:
    """Board-local waypoint templates; use target_matrix for controlled pelvis poses.

    For top-down, template rotations describe the board-identity reference only;
    grasp rotation is also the fixed pelvis orientation used by the resolver.
    """
    peg_id: str
    source_hole: str
    destination: str
    arm: str
    source_tip: np.ndarray
    grasp: np.ndarray
    approach: np.ndarray
    extracted: np.ndarray
    preinsert: np.ndarray
    aperture_m: float
    length_m: float
    top_down: bool
    contact_offset: np.ndarray
    approach_m: float

    def target_matrix(self, board, name):
        """Resolve a named waypoint, preserving its contact point and frame convention."""
        if name == 'carry':
            target = self.preinsert.copy()
            target[2, 3] = max(self.extracted[2, 3], target[2, 3]) + 0.02
        elif name in ('grasp', 'approach', 'extracted', 'preinsert'):
            target = getattr(self, name)
        else:
            raise ValueError(f'unknown marker waypoint {name!r}')
        if not self.top_down:
            return board @ target
        if name == 'approach':
            resolved = self.target_matrix(board, 'grasp')
            resolved[:3, 3] -= self.approach_m * resolved[:3, 2]
            return resolved
        # Positions remain board-local, but the documented orientation is in
        # pelvis. Rotate the contact into pelvis BEFORE subtracting the G offset.
        contact = target[:3, 3] + target[:3, :3] @ self.contact_offset
        resolved = np.eye(4)
        resolved[:3, :3] = self.grasp[:3, :3]
        resolved[:3, 3] = (board[:3, :3] @ contact + board[:3, 3]
                           - resolved[:3, :3] @ self.contact_offset)
        return resolved

    def candidates(self, board):
        return make_candidates([matrix_to_pose(self.target_matrix(board, 'grasp'))],
                               [self.peg_id + ':catalog'], self.arm,
                               gripper_width=self.aperture_m)


class MarkerCatalog:
    def __init__(self, catalog_path, profile_path=None):
        if not catalog_path:
            raise ValueError('marker_catalog_path must name the ManipulationNet catalog')
        with open(catalog_path, encoding='utf-8') as stream:
            self.catalog = json.load(stream)
        with open(profile_path or Path(__file__).with_name('marker_profile.json'),
                  encoding='utf-8') as stream:
            self.profile = json.load(stream)
        p = self.profile
        for key in ('approach_m', 'extraction_m', 'preinsert_tip_height_m',
                    'step_m', 'step_rad', 'segment_sec', 'planned_motion_sec', 'linear_tolerance_m',
                    'angular_tolerance_rad', 'max_pose_age_sec', 'max_goal_sec',
                    'max_board_jump_m', 'max_board_jump_rad', 'aperture_margin_m'):
            if not math.isfinite(p[key]) or p[key] <= 0:
                raise ValueError(f'invalid marker profile {key}')
        if p['extraction_m'] <= max(h['seated_depth'] for h in self.catalog['holes'].values()):
            raise ValueError('extraction must clear the seated depth')

    def resolve(self, name):
        return self.profile['aliases'].get(name.strip(), name.strip())

    def compatible(self, peg_id, destination):
        """Check every catalog shape with its actual geometry, independently of grasp support."""
        holes = self.catalog['holes']
        if peg_id not in holes or destination not in holes:
            raise ValueError('unknown peg or hole catalog ID')
        source, hole = holes[peg_id], holes[destination]
        peg = self.catalog['variants'][source['shape']][str(source['size'])]
        if source['shape'] != hole['shape'] or not fits(peg, hole):
            raise ValueError('peg shape/dimensions do not fit destination hole')
        return peg

    def geometry(self, peg_id, destination, arm, *, top_down=False):
        peg_id, destination = self.resolve(peg_id), self.resolve(destination)
        peg = self.compatible(peg_id, destination)
        active = self.profile['active_pegs'].get(peg_id)
        if active is None:
            raise ValueError(f'peg {peg_id!r} is not present in active_pegs')
        source_id = active['source_hole']
        self.compatible(peg_id, source_id)
        if destination != source_id and any(
                value['source_hole'] == destination
                for key, value in self.profile['active_pegs'].items() if key != peg_id):
            raise ValueError('destination hole is occupied by another active peg')
        if self.catalog['holes'][peg_id]['shape'] != 'cylinder':
            raise ValueError('no calibrated grasp profile for this shape')
        if arm not in self.profile['arms']:
            raise ValueError('marker mode requires explicit left or right arm')
        source = self.catalog['holes'][source_id]
        target = self.catalog['holes'][destination]
        tip = hole_transform(source, 'target')
        grasp = np.eye(4)
        # workspace_analysis/head_camera.json specifies grasp_frame, whose +X
        # approaches. The per-arm profile converts its inward yaw to GRASPGENX
        # (+Z approach, +X closing); applying Euler pitch=90 to G is incorrect.
        rpy = (self.profile['top_down_rpy_pelvis_deg'][arm] if top_down
               else self.profile['grasp_rpy_board_deg'])
        grasp[:3, :3] = Rotation.from_euler(
            'xyz', rpy, degrees=True).as_matrix()
        # A cylinder side pinch must close across the shaft, not along its axis.
        if abs(grasp[2, 0]) > 0.1 or grasp[2, 2] > -0.5:
            raise ValueError('profile is not a downward GRASPGENX cylinder pinch')
        contact = tip @ np.r_[self.profile['cylinder_contact_from_tip_m'], 1.0]
        if not source['entry'][2] < contact[2] < tip[2, 3] + peg['length']:
            raise ValueError('profile contact must lie on the exposed cylinder above the source entry')
        offset = np.asarray(self.profile['arms'][arm]['contact_in_graspgenx_m'])
        if offset.shape != (3,) or not np.isfinite(offset).all() or offset[2] <= 0:
            raise ValueError('invalid per-arm GRASPGENX contact calibration')
        grasp[:3, 3] = contact[:3] - grasp[:3, :3] @ offset
        rigid(grasp)
        approach = grasp.copy()
        approach[:3, 3] -= self.profile['approach_m'] * grasp[:3, 2]
        extracted = grasp.copy()
        extracted[2, 3] += self.profile['extraction_m']
        # Preserve the rigid object-to-gripper transform when aligning its TIP,
        # not the gripper base, above the destination entry.
        destination_tip = hole_transform(target, 'entry')
        destination_tip[2, 3] += self.profile['preinsert_tip_height_m']
        preinsert = destination_tip @ np.linalg.inv(tip) @ grasp
        width = float(np.ptp(np.asarray(peg['bounds'])[:, :2], axis=0).max())
        aperture = width + self.profile['aperture_margin_m']
        if not 0 < aperture <= 0.106:
            raise ValueError('peg exceeds gripper aperture')
        return TargetGeometry(peg_id, source_id, destination, arm, tip, grasp,
                              approach, extracted, preinsert, aperture, peg['length'],
                              top_down, offset, self.profile['approach_m'])

    def check_workspace(self, board, geometry):
        limits = self.profile['arms'][geometry.arm]
        if geometry.top_down:
            approach = geometry.target_matrix(board, 'grasp')[:3, 2]
            pitch = math.degrees(math.atan2(-approach[2], np.linalg.norm(approach[:2])))
            # Validate the resolved pelvis orientation, not board tilt or motion tolerance.
            if pitch + 1e-10 < 90.0:
                raise ValueError('marker grasp violates requested downward pitch')
        for name in ('approach', 'grasp', 'extracted', 'preinsert', 'carry'):
            point = geometry.target_matrix(board, name)[:3, 3]
            if (np.any(point < limits['workspace_min_m'])
                    or np.any(point > limits['workspace_max_m'])):
                raise ValueError('target outside configured arm workspace; no motion permitted')


class BoardObservation:
    """Atomically match independently delivered pose and validity observations.

    An unmatched good update never replaces the last matched pair. Invalidity
    is ordered by observation stamp, not heartbeat/publication time, and blocks
    that observation and all older pairs even if their messages arrive later.
    """

    PENDING_LIMIT = 4

    def __init__(self, max_age, use_sim_time=False):
        self.max_age = max_age
        self.use_sim_time = use_sim_time
        self.lock = threading.Lock()
        self._poses = {}
        self._diagnostics = {}
        self._accepted = None  # observation stamp, matrix, diagnostic stamp, receipt time
        self._invalid_through = 0
        self._last_now_ns = None

    def _invalidate(self, stamp):
        # An unorderable invalid message cannot be ignored as an older packet.
        if stamp <= 0:
            stamp = max([1, self._accepted[0] if self._accepted else 0,
                         *self._poses, *self._diagnostics])
        self._invalid_through = max(self._invalid_through, stamp)
        for cache in (self._poses, self._diagnostics):
            for key in list(cache):
                if key <= self._invalid_through:
                    del cache[key]

    def _match(self):
        floor = max(self._invalid_through, self._accepted[0] if self._accepted else 0)
        matched = self._poses.keys() & self._diagnostics.keys()
        usable = [stamp for stamp in matched if stamp > floor]
        if usable:
            stamp = max(usable)
            matrix, pose_received = self._poses[stamp]
            diagnostic_stamp, diagnostic_received = self._diagnostics[stamp]
            self._accepted = (stamp, matrix, diagnostic_stamp,
                              min(pose_received, diagnostic_received))
            floor = stamp
        for cache in (self._poses, self._diagnostics):
            for key in list(cache):
                if key <= floor:
                    del cache[key]
            for key in sorted(cache)[:-self.PENDING_LIMIT]:
                del cache[key]

    def on_pose(self, msg):
        with self.lock:
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if self._accepted and stamp < self._accepted[0]:
                return
            try:
                if stamp <= 0 or msg.header.frame_id != 'pelvis':
                    raise ValueError('invalid observation stamp or board frame')
                q = msg.pose.orientation
                if not np.isclose(np.linalg.norm([q.x, q.y, q.z, q.w]), 1.0, atol=1e-6):
                    raise ValueError('invalid board quaternion')
                matrix = rigid(pose_to_matrix(msg.pose)).copy()
            except ValueError:
                self._invalidate(stamp)
                return
            if stamp <= max(self._invalid_through, self._accepted[0] if self._accepted else 0):
                return
            matrix.setflags(write=False)
            self._poses.setdefault(stamp, (matrix, time.monotonic()))
            self._match()

    def on_diagnostics(self, msg):
        with self.lock:
            statuses = [s for s in msg.status if s.name == 'pegboard_localization']
            if not statuses:
                return
            s = statuses[0]
            values = {v.key: v.value for v in s.values}
            try:
                observation_stamp = int(values.get('observation_stamp_ns', '0'))
            except (ValueError, TypeError):
                observation_stamp = 0
            if self._accepted and 0 < observation_stamp < self._accepted[0]:
                return
            diagnostic_stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if (observation_stamp <= 0 or diagnostic_stamp <= 0
                    or diagnostic_stamp < observation_stamp
                    or s.level != DiagnosticStatus.OK or values.get('valid') != 'true'):
                self._invalidate(observation_stamp)
                return
            if observation_stamp <= max(
                    self._invalid_through, self._accepted[0] if self._accepted else 0):
                return
            self._diagnostics.setdefault(observation_stamp, (diagnostic_stamp, time.monotonic()))
            self._match()

    def matrix(self, now_ns):
        with self.lock:
            reset = self._last_now_ns is not None and now_ns < self._last_now_ns
            self._last_now_ns = now_ns
            if reset:
                self._accepted = None
                self._poses.clear()
                self._diagnostics.clear()
                self._invalid_through = 0
                raise ValueError('board localization clock reset; waiting for new observations')
            if self._accepted is None:
                cause = ('invalid observation or diagnostics' if self._invalid_through
                         else 'no matched observation and valid diagnostics')
                raise ValueError(f'waiting for board pose and valid matching diagnostics: {cause}')
            stamp, matrix, diagnostic_stamp, received = self._accepted
            if stamp <= self._invalid_through:
                cause = 'observation invalidated by invalid pose or diagnostics'
            elif not 0 <= (now_ns - stamp) / 1e9 <= self.max_age:
                cause = ('observation timestamp is in the future' if stamp > now_ns
                         else 'stale observation ROS age')
            elif not 0 <= (now_ns - diagnostic_stamp) / 1e9 <= self.max_age:
                cause = ('diagnostic timestamp is in the future' if diagnostic_stamp > now_ns
                         else 'stale diagnostic ROS age')
            # Paused or slow simulation ages observations only in ROS time.
            elif not self.use_sim_time and time.monotonic() - received > self.max_age:
                cause = 'stale monotonic wall receipt age'
            elif any(key > now_ns for key in self._poses):
                cause = 'pending future observation timestamp'
            elif any(key > now_ns or value[0] > now_ns
                     for key, value in self._diagnostics.items()):
                cause = 'pending future diagnostic timestamp'
            else:
                return matrix.copy()
            raise ValueError(f'board localization invalid, stale, future, or clock reset: {cause}')
