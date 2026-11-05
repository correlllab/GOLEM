"""Marker-mode composition for the existing grasp and pick_place actions.

Only a known seated cylinder is graspable. Pick/place means extract and hold
at pre-insertion, never contact insertion or release. No cloud or AI is used.
"""

import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from custom_ros_messages.action import SkillGrasp, SkillPickPlace
from magpie_msgs.srv import SetGripperPosition, SetGripperForce
from std_srvs.srv import Trigger

from ..base import _Run, GRASP_FRAMES
from ..perception_utils import matrix_to_pose, pose_to_matrix


class MarkerManipulationSkill:
    def _marker_board(self, reference=None):
        board = self._board_observation.matrix(self.get_clock().now().nanoseconds)
        if reference is None:
            reference = getattr(self, '_marker_reference', None)
        if reference is not None:
            delta = np.linalg.inv(reference) @ board
            p = self._marker_catalog.profile
            if (np.linalg.norm(delta[:3, 3]) > p['max_board_jump_m']
                    or Rotation.from_matrix(delta[:3, :3]).magnitude() > p['max_board_jump_rad']):
                raise ValueError('board pose jumped beyond configured tracking limits')
        self._marker_reference = board
        return board

    def _marker_move(self, target, geometry, run, gh, label, planned=False):
        """Bounded board-relative waypoints; refresh the observation before EVERY send."""
        p = self._marker_catalog.profile
        board = self._marker_board()
        current = self._frame_pose_in_pelvis(GRASP_FRAMES[geometry.arm])
        if current is None:
            raise ValueError('grasp frame TF unavailable')
        start = np.linalg.inv(board) @ pose_to_matrix(current)
        angle = Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).magnitude()
        count = max(1, math.ceil(max(
            np.linalg.norm(target[:3, 3] - start[:3, 3]) / p['step_m'], angle / p['step_rad'])))
        # Free-space transit uses the existing collision planner once, rather
        # than replanning and settling at every centimetre. Contact/extraction
        # remain small, direct waypoints along the authored board axes.
        if planned:
            count = 1
        rotation = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))

        skill = self

        class MotionGuard:
            @property
            def is_cancel_requested(self):
                if gh.is_cancel_requested or run.remaining() <= 0 or skill.arm_halt_reason():
                    return True
                try:
                    skill._marker_board(board)
                except ValueError:
                    return True
                return False

        for k in range(1, count + 1):
            if gh.is_cancel_requested or run.remaining() <= 0 or self.arm_halt_reason():
                raise ValueError('motion canceled, timed out, or arm halted')
            board = self._marker_board(board)
            self._marker_catalog.check_workspace(board, geometry)
            t = k / count
            waypoint = np.eye(4)
            waypoint[:3, :3] = rotation([t]).as_matrix()[0]
            waypoint[:3, 3] = start[:3, 3] + t * (target[:3, 3] - start[:3, 3])
            if not self.move_frame_to(
                    GRASP_FRAMES[geometry.arm], matrix_to_pose(board @ waypoint),
                    outer_gh=MotionGuard(), duration_sec=min(
                        p['planned_motion_sec'] if planned else p['segment_sec'], run.remaining()),
                    do_plan=planned, slow_mode=True,
                    lin_tol=p['linear_tolerance_m'], ang_tol=p['angular_tolerance_rad'],
                    label=f'marker {label} {k}/{count}'):
                raise ValueError(f'{label} waypoint not reached; no subsequent close or release')
            measured = self._frame_pose_in_pelvis(GRASP_FRAMES[geometry.arm])
            if measured is None:
                raise ValueError('measured frame pose unavailable after motion')
            delta = np.linalg.inv(board @ waypoint) @ pose_to_matrix(measured)
            if (np.linalg.norm(delta[:3, 3]) > p['linear_tolerance_m']
                    or Rotation.from_matrix(delta[:3, :3]).magnitude() > p['angular_tolerance_rad']):
                raise ValueError('measured waypoint error exceeds marker profile tolerance')
        self._marker_board(board)

    def _marker_gripper(self, arm, kind, value, run, gh):
        """No retry/hold thread can send another close after cancellation."""
        if gh.is_cancel_requested or run.remaining() <= 0 or self.arm_halt_reason():
            raise ValueError('gripper command canceled, timed out, or arm halted')
        self._marker_board()
        if kind == 'position':
            self._stop_grip_hold(arm)
            request = SetGripperPosition.Request()
            request.position, request.speed = float(value), 1.0
            client = self.gripper_clis[arm]
        elif kind == 'force':
            request = SetGripperForce.Request()
            request.max_force = float(value)
            client = self.gripper_force_clis[arm]
        else:
            request = Trigger.Request()
            client = self.gripper_close_clis[arm]
        result = self._call_service(client, request, f'marker {kind}',
                                    timeout_sec=min(5.0, run.remaining()), outer_gh=gh)
        if result is None or not result.success:
            raise ValueError(f'marker gripper {kind} failed')

    def _marker_settle(self, run, gh):
        until = time.monotonic() + 1.0
        event = threading.Event()
        while time.monotonic() < until:
            if gh.is_cancel_requested or run.remaining() <= 0 or self.arm_halt_reason():
                raise ValueError('canceled or timed out while settling')
            self._marker_board()
            event.wait(min(0.02, max(0, until - time.monotonic())))

    def _exec_marker(self, gh, pick_place=False):
        from .grasp import GraspOutcome

        goal = gh.request
        action = SkillPickPlace if pick_place else SkillGrasp
        run = _Run(self, gh, action, 'marker_pick_place' if pick_place else 'marker_grasp')
        p = self._marker_catalog.profile
        run.deadline = min(run.deadline, time.monotonic() + p['max_goal_sec'])
        self._last_grasp_outcome = None
        self._marker_reference = None
        try:
            if goal.visual_servo or goal.top_down:
                raise ValueError('marker profile rejects AI visual_servo and top_down overrides')
            arm = self._validated_arm(goal)
            peg_id = self._marker_catalog.resolve(goal.target_object)
            active = p['active_pegs'].get(peg_id)
            if active is None:
                raise ValueError(f'peg {peg_id!r} not present in active_pegs')
            destination = goal.place_target if pick_place else active['source_hole']
            geometry = self._marker_catalog.geometry(peg_id, destination, arm)
            if peg_id in self._marker_displaced:
                raise ValueError('peg source pose consumed/uncertain; reset experiment before another grasp')
            board = self._marker_board()
            self._marker_catalog.check_workspace(board, geometry)
            candidates = geometry.candidates(board)
            pitch = math.degrees(math.asin(np.clip(-(board @ geometry.grasp)[2, 2], -1, 1)))
            min_pitch = float(getattr(goal, 'min_downward_pitch_deg', 0.0))
            if not 0 <= min_pitch <= 90 or pitch < min_pitch:
                raise ValueError('marker grasp violates requested downward pitch')
            if not run.phase('marker_pregrasp', 0.1):
                return run.result
            self._marker_gripper(arm, 'position', geometry.aperture_m * 1000, run, gh)
            self._marker_settle(run, gh)
            self._marker_move(geometry.approach, geometry, run, gh, 'pregrasp', planned=True)
            if not run.phase('marker_approach', 0.3):
                return run.result
            self._marker_move(geometry.grasp, geometry, run, gh, 'contact approach')
            self._marker_settle(run, gh)
            board = self._marker_board()
            measured = self._frame_pose_in_pelvis(GRASP_FRAMES[arm])
            if measured is None:
                raise ValueError('no measured grasp pose before close')
            delta = np.linalg.inv(board @ geometry.grasp) @ pose_to_matrix(measured)
            if (np.linalg.norm(delta[:3, 3]) > p['linear_tolerance_m']
                    or Rotation.from_matrix(delta[:3, :3]).magnitude() > p['angular_tolerance_rad']):
                raise ValueError('measured contact pose outside tolerance; will not close')
            self._marker_gripper(arm, 'force', 15.0, run, gh)
            if not run.phase('marker_close', 0.5):
                return run.result
            # Even a timed-out close may have moved the object. Never reuse its
            # assumed seated pose after dispatching that command.
            self._marker_displaced.add(peg_id)
            self._marker_gripper(arm, 'close', 0, run, gh)
            self._marker_settle(run, gh)
            grasp_pose = matrix_to_pose(board @ geometry.grasp)
            self._last_grasp_outcome = GraspOutcome(
                arm, grasp_pose, 'pelvis',
                (board @ geometry.source_tip @ np.array([0, 0, geometry.length_m / 2, 1]))[:3],
                geometry.length_m / 2, candidates.gripper_width, 1.0, 'marker_catalog')
            if pick_place:
                if not run.phase('marker_extract', 0.6):
                    return run.result
                self._marker_move(geometry.extracted, geometry, run, gh, 'board-normal extraction')
                if not run.phase('marker_preinsert', 0.8):
                    return run.result
                # Lift/transit before lowering; never carry through the board.
                carry = geometry.preinsert.copy()
                carry[2, 3] = max(geometry.extracted[2, 3], carry[2, 3]) + 0.02
                self._marker_move(carry, geometry, run, gh, 'carry', planned=True)
                self._marker_move(geometry.preinsert, geometry, run, gh, 'preinsert')
            if not run.phase('marker_hold', 0.95):
                return run.result
            self._marker_board()
            return run.succeed(
                f'{peg_id} from {geometry.source_hole}: ' +
                (f'held at pre-insertion above {geometry.destination}; no insertion or release performed'
                 if pick_place else 'grasp commanded at known seated pose; no extraction performed') +
                '; nominal geometry only, no contact/insertion or grip verification')
        except (ValueError, KeyError, TypeError) as exc:
            self._last_grasp_outcome = None
            return run.abort(str(exc) + '; no automatic retreat or release')
