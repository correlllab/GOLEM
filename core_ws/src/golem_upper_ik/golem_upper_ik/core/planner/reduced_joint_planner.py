"""Plan in the reduced (IK-joint) space of a RobotModel with OMPL: ReducedJointPlanner.

A path is valid when every joint is within its limits, the self-collision model reports no
collision, and the planner frames stay within the configured workspace bounds. The planner
needs a self-collision model; PlannerClient refuses requests for a robot without one.
"""
import numpy as np
from ompl import base as ob
from ompl import geometric as og
from dataclasses import dataclass

# two arms of seven IK joints each, left first: an arm whose start and goal differ by less
# than _ARM_MOVE_TOL (rad) on every joint is pinned when the caller gives no active mask
_ARM_REDUCED_NQ = 14
_ARM_MOVE_TOL = 2e-2
# a sparse path is resampled to this many points before its frame path length is measured
_LENGTH_ESTIMATE_STEPS = 50
# m: the grasp radius that turns a planner frame's rotation into swept arc length
_FRAME_ROTATION_RADIUS = 0.20


@dataclass
class PlannerConfig:
    """Hold the planner's settings (the controller config's planner section).

    Units: timeout, simplify_time and dt in s; range, validity_resolution and
    start_limit_tolerance in rad (joint space); moving_speed in m/s; the frame_* bounds and
    margins in m, in the pelvis frame, None for no bound.
    config/<robot>/example.controller.yaml in the source tree describes each key.
    """

    planner: str = 'RRTConnect'
    timeout: float = 1.0
    range: float = 0.0
    try_direct: bool = True
    simplify: bool = True
    simplify_time: float = 1.0
    shortcut: bool = True
    moving_speed: float = 0.25
    dt: float = 1.0 / 30.0
    min_interpolation_steps: int = 2
    max_interpolation_steps: int = 300
    validity_resolution: float = 0.0025
    constraint_check_steps: int = 10
    # frames bounded by the frame_* limits and timed by moving_speed; PlannerClient sets
    # them, and the planner refuses a config without them
    frame_names: tuple = None
    frame_x_max: float = 0.50
    frame_y_min: float = None
    frame_y_max: float = None
    frame_z_min: float = None
    frame_z_max: float = 0.50
    frame_limit_margin: float = 1e-3
    frame_z_corridor_margin: float = 0.05
    # the start is the robot's measured pose, which can sit marginally past a joint limit;
    # an overshoot up to this is clamped into the limits, a larger one fails the plan
    start_limit_tolerance: float = 0.05


@dataclass
class PlanResult:
    """Hold a plan's outcome: success, the (N, IK joints) path, and why."""

    success: bool
    path: np.ndarray
    reason: str = ''


class ReducedJointPlanner:
    """Plan valid paths between reduced (IK-joint) configurations with OMPL."""

    def __init__(self, robot_model, config: PlannerConfig = None):
        """Build the OMPL problem once for robot_model; ValueError for a bad model or config."""
        self.robot_model = robot_model
        self.config = config if config is not None else PlannerConfig()
        self._check_robot_model_ready()
        self._check_planner_config()

        self.nq = int(self.robot_model.model_body_reduced.nq)
        self.frame_names = tuple(self.config.frame_names)
        self.frame_min_z = self._make_frame_floor_map(self.config.frame_z_min)
        self.frame_max_z = self._make_frame_floor_map(self.config.frame_z_max)
        self.corridor_min_z = {}
        # the joints that may move, and where the others are pinned
        self.active_mask = np.ones(self.nq, dtype=bool)
        self.pinned_q = np.zeros(self.nq)

        # the OMPL problem is built once; plan() only resets the start and goal
        self.space = self._make_space()
        self.ss = og.SimpleSetup(self.space)
        self.si = self.ss.getSpaceInformation()
        self.ss.setStateValidityChecker(self._make_validity_checker())
        self.si.setStateValidityCheckingResolution(
            float(self.config.validity_resolution)
        )
        self.ss.setPlanner(self._make_planner())

    def _check_robot_model_ready(self):
        """Raise ValueError unless the model has a reduced and a self-collision model."""
        if getattr(self.robot_model, 'model_body_reduced', None) is None:
            raise ValueError('RobotModel reduced model is not initialized')
        if not self.robot_model.has_collision:
            raise ValueError('RobotModel collision model is not initialized')

    def _check_planner_config(self):
        """Raise ValueError for a config the planner cannot run with."""
        if self.config.frame_names is None:
            raise ValueError('Planner frame_names must be given')
        if self.config.moving_speed <= 0.0:
            raise ValueError('Planner moving_speed must be positive')
        if self.config.dt <= 0.0:
            raise ValueError('Planner dt must be positive')
        if self.config.min_interpolation_steps < 2:
            raise ValueError('Planner min_interpolation_steps must be at least 2')
        if (
            self.config.max_interpolation_steps <
            self.config.min_interpolation_steps
        ):
            raise ValueError(
                'Planner max_interpolation_steps must be at least '
                'min_interpolation_steps'
            )
        if self.config.frame_limit_margin < 0.0:
            raise ValueError('Planner frame_limit_margin must be non-negative')
        if self.config.frame_z_corridor_margin < 0.0:
            raise ValueError(
                'Planner frame_z_corridor_margin must be non-negative'
            )
        if (
            self.config.frame_y_min is not None and
            self.config.frame_y_max is not None and
            self.config.frame_y_min > self.config.frame_y_max
        ):
            raise ValueError('Planner frame_y_min must not exceed frame_y_max')

    def _make_space(self):
        """Return an OMPL real-vector space bounded by the reduced model's joint limits."""
        self.model_lower = np.asarray(
            self.robot_model.model_body_reduced.lowerPositionLimit,
            dtype=float,
        )
        self.model_upper = np.asarray(
            self.robot_model.model_body_reduced.upperPositionLimit,
            dtype=float,
        )
        lower, upper = self.model_lower, self.model_upper
        if lower.shape != upper.shape:
            raise ValueError('Reduced model lower and upper bounds differ')
        if lower.shape != (self.nq,):
            raise ValueError('Reduced model bounds do not match nq')
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError('Reduced model bounds must be finite')
        if np.any(lower >= upper):
            raise ValueError(
                'Reduced model lower bounds must be below upper bounds'
            )

        space = ob.RealVectorStateSpace(self.nq)
        space.setBounds(self._make_bounds(lower, upper))
        return space

    def _make_bounds(self, lower, upper):
        """Return OMPL bounds from per-joint lower and upper arrays."""
        bounds = ob.RealVectorBounds(self.nq)
        for idx in range(self.nq):
            bounds.setLow(idx, float(lower[idx]))
            bounds.setHigh(idx, float(upper[idx]))
        return bounds

    def _apply_active_bounds(self):
        """Collapse the OMPL bounds of the inactive joints to their pinned values."""
        lower = np.copy(self.model_lower)
        upper = np.copy(self.model_upper)
        lower[~self.active_mask] = self.pinned_q[~self.active_mask]
        upper[~self.active_mask] = self.pinned_q[~self.active_mask]
        self.space.setBounds(self._make_bounds(lower, upper))

    def _project_inactive(self, q):
        """Return q with the inactive joints at their pinned start values."""
        if np.all(self.active_mask):
            return q
        q = np.array(q, dtype=float)
        q[~self.active_mask] = self.pinned_q[~self.active_mask]
        return q

    def _make_planner(self):
        """Return the OMPL planner the config names; ValueError if this OMPL lacks it."""
        # resolved by name at run time: OMPL builds expose different sets of planners
        planner_name = self.config.planner.lower()
        planner_classes = {
            'rrtconnect': 'RRTConnect',
            'rrt': 'RRT',
            'rrtstar': 'RRTstar',
            'prm': 'PRM',
            'lazyprm': 'LazyPRM',
            'kpiece1': 'KPIECE1',
        }
        class_name = planner_classes.get(planner_name)
        planner_cls = getattr(og, class_name, None) if class_name else None
        if planner_cls is None:
            raise ValueError(
                f'Unsupported OMPL planner: {self.config.planner}'
            )

        planner = planner_cls(self.si)
        if self.config.range > 0.0 and hasattr(planner, 'setRange'):
            planner.setRange(float(self.config.range))
        return planner

    def _make_validity_checker(self):
        """Return the state validity checker in the form this OMPL binding takes."""
        if hasattr(ob, 'StateValidityCheckerFn'):
            return ob.StateValidityCheckerFn(self._is_state_valid)
        return self._is_state_valid

    def _as_reduced_array(self, q, name):
        """Return q as a finite reduced array; ValueError naming it otherwise."""
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.nq,):
            raise ValueError(f'{name} must have shape ({self.nq},)')
        if not np.all(np.isfinite(q)):
            raise ValueError(f'{name} must contain only finite values')
        return q

    def _empty_path(self):
        """Return a path with no waypoints."""
        return np.empty((0, self.nq), dtype=float)

    def _make_state(self, q):
        """Return an OMPL state at q, allocated the way this OMPL binding supports."""
        try:
            state = ob.State(self.space)
            values = state()
        except TypeError:
            state = self.space.allocState()
            values = state
        for idx, value in enumerate(q):
            values[idx] = float(value)
        return state

    def _state_to_array(self, state):
        """Return an OMPL state as reduced q, the inactive joints at their pinned values."""
        q = np.asarray([state[idx] for idx in range(self.nq)], dtype=float)
        return self._project_inactive(q)

    def _is_state_valid(self, state):
        """Return whether an OMPL state is valid (OMPL's validity callback)."""
        q = self._state_to_array(state)
        reason = self._validity_failure(q, 'state')
        return reason == ''

    def _validity_failure(self, q, name):
        """Return why q (called name) is invalid: limits, self-collision, workspace; or ''."""
        if not self.robot_model.check_within_limits_reduced(q):
            return (
                f'{name} is outside reduced joint limits: '
                f'{self._format_limit_violations(q)}'
            )
        if not self.robot_model.check_collision_free_reduced(q):
            return f'{name} is in self-collision'
        reason = self._workspace_constraint_failure(q, name)
        if reason:
            return reason
        return ''

    def _format_limit_violations(self, q):
        """Return the joints of q past their limits, as text."""
        violations = self.robot_model.reduced_limit_violations(q)
        if not violations:
            return '(no joint outside limits; check tolerance)'
        return ', '.join(
            f'{v["name"]}={v["value"]:.4f} '
            f'(limit {v["bound"]:.4f}, {v["overshoot"]:.4f} rad over)'
            for v in violations
        )

    def _clamp_start_into_limits(self, start):
        """Return start with overshoots within start_limit_tolerance clamped into the limits.

        Larger overshoots are left untouched, so plan() still rejects a bad pose. A clamp is
        printed.
        """
        tol = self.config.start_limit_tolerance
        if tol is None or tol <= 0.0:
            return start
        lower = self.robot_model.model_body_reduced.lowerPositionLimit
        upper = self.robot_model.model_body_reduced.upperPositionLimit
        clamped = np.clip(start, lower, upper)
        delta = clamped - start
        within = np.abs(delta) <= tol
        result = np.where(within, clamped, start)
        moved = within & (delta != 0.0)
        if np.any(moved):
            idx_to_name = self.robot_model._reduced_idx_to_name()
            detail = ', '.join(
                f'{idx_to_name.get(i, f"q[{i}]")} '
                f'{start[i]:.4f}->{result[i]:.4f}'
                for i in np.where(moved)[0]
            )
            print(
                f'[planner] clamped start into joint limits (tol={tol:.4f}): '
                f'{detail}',
                flush=True,
            )
        return result

    def _motor_q(self, q_reduced):
        """Return the measured body q with the IK joints at q_reduced."""
        q = np.copy(self.robot_model.state['q'])
        q[self.robot_model.reduced_mask] = q_reduced
        return q

    def _sample_path_for_length(self, path):
        """Return a sparse path resampled to _LENGTH_ESTIMATE_STEPS points, even in joint space."""
        if len(path) < 2 or len(path) >= _LENGTH_ESTIMATE_STEPS:
            return path

        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if distance[-1] <= 0.0:
            return path

        sample_distance = np.linspace(0.0, distance[-1], _LENGTH_ESTIMATE_STEPS)
        return np.asarray([
            np.interp(sample_distance, distance, path[:, idx])
            for idx in range(path.shape[1])
        ]).T

    def _rotation_angle(self, rotation_a, rotation_b):
        """Return the angle between two rotation matrices, rad."""
        rotation_delta = rotation_a.T @ rotation_b
        cos_angle = 0.5 * (np.trace(rotation_delta) - 1.0)
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _frame_path_length(self, path, frame_name):
        """Return (total, translation, rotation-as-arc) path lengths of a frame along path, m."""
        if len(path) < 2:
            return 0.0, 0.0, 0.0

        motor_path = [self._motor_q(q) for q in path]
        positions = np.asarray(
            [self.robot_model.get_frame_position(frame_name, q) for q in motor_path]
        )
        deltas = np.diff(positions, axis=0)
        translation_length = float(np.sum(np.linalg.norm(deltas, axis=1)))

        rotation_length = 0.0
        if hasattr(self.robot_model, 'get_frame_rotation'):
            rotations = [
                self.robot_model.get_frame_rotation(frame_name, q)
                for q in motor_path
            ]
            rotation_angle = sum(
                self._rotation_angle(rotations[idx], rotations[idx + 1])
                for idx in range(len(rotations) - 1)
            )
            rotation_length = _FRAME_ROTATION_RADIUS * rotation_angle

        return (
            translation_length + rotation_length,
            translation_length,
            rotation_length,
        )

    def _interpolation_metadata(self, path):
        """Return the path's waypoint count, duration and lengths at moving_speed and dt.

        The duration comes from the longest planner-frame path, clamped to the
        interpolation step limits.
        """
        sample_path = self._sample_path_for_length(path)
        frame_lengths = {
            frame_name: self._frame_path_length(sample_path, frame_name)
            for frame_name in self.frame_names
        }
        lengths = {
            frame_name: values[0]
            for frame_name, values in frame_lengths.items()
        }
        translation_lengths = {
            frame_name: values[1]
            for frame_name, values in frame_lengths.items()
        }
        rotation_lengths = {
            frame_name: values[2]
            for frame_name, values in frame_lengths.items()
        }
        path_length = max(lengths.values(), default=0.0)

        speed = float(self.config.moving_speed)
        dt = float(self.config.dt)
        steps = int(np.ceil(path_length / speed / dt)) + 1

        steps = max(int(self.config.min_interpolation_steps), steps)
        steps = min(int(self.config.max_interpolation_steps), steps)
        duration = max(steps - 1, 1) * dt
        metadata = {
            'waypoint_count': steps,
            'duration': duration,
            'effective_speed': path_length / duration,
            'path_length': path_length,
            'frame_path_lengths': lengths,
            'frame_translation_lengths': translation_lengths,
            'frame_rotation_lengths': rotation_lengths,
            'length_estimate_samples': len(sample_path),
        }
        return metadata

    def _print_interpolation_metadata(self, metadata, label):
        """Print a path's interpolation metadata, labelled."""
        frame_lengths = ', '.join(
            f'{name}={length:.4f}m '
            f'(pos={metadata["frame_translation_lengths"][name]:.4f}, '
            f'rot={metadata["frame_rotation_lengths"][name]:.4f})'
            for name, length in metadata['frame_path_lengths'].items()
        )
        print(
            f'[planner] {label}: '
            f'waypoints={metadata["waypoint_count"]}, '
            f'duration={metadata["duration"]:.3f}s, '
            f'effective_speed={metadata["effective_speed"]:.4f}m/s, '
            f'path_length={metadata["path_length"]:.4f}m, '
            f'samples={metadata["length_estimate_samples"]}, '
            f'{frame_lengths}',
            flush=True,
        )

    def _workspace_constraint_failure(self, q_reduced, name):
        """Return why a planner frame at q_reduced leaves the workspace bounds, or ''.

        Each bound is allowed frame_limit_margin; the z floor is the higher of frame_z_min
        and the plan's corridor floor.
        """
        if (
            not self.frame_min_z and
            not self.frame_max_z and
            not self.corridor_min_z and
            self.config.frame_x_max is None and
            self.config.frame_y_min is None and
            self.config.frame_y_max is None
        ):
            return ''

        margin = self.config.frame_limit_margin
        q = self._motor_q(q_reduced)
        for frame_name in self.frame_names:
            position = self.robot_model.get_frame_position(frame_name, q)
            x = position[0]
            y = position[1]
            z = position[2]
            min_z = self.frame_min_z.get(frame_name)
            corridor_z = self.corridor_min_z.get(frame_name)
            if corridor_z is not None:
                min_z = corridor_z if min_z is None else max(min_z, corridor_z)
            if min_z is not None and z + margin < min_z:
                return (
                    f'{name} moves {frame_name} below z={min_z:.4f} '
                    f'(z={z:.4f}, {min_z - z:.4f}m below floor)'
                )
            max_z = self.frame_max_z.get(frame_name)
            if max_z is not None and z - margin > max_z:
                return (
                    f'{name} moves {frame_name} above z={max_z:.4f} '
                    f'(z={z:.4f}, {z - max_z:.4f}m above ceiling)'
                )
            if (
                self.config.frame_x_max is not None and
                x - margin > self.config.frame_x_max
            ):
                return (
                    f'{name} moves {frame_name} beyond '
                    f'x={self.config.frame_x_max:.4f} '
                    f'(x={x:.4f}, {x - self.config.frame_x_max:.4f}m beyond)'
                )
            if (
                self.config.frame_y_min is not None and
                y + margin < self.config.frame_y_min
            ):
                return (
                    f'{name} moves {frame_name} below '
                    f'y={self.config.frame_y_min:.4f} '
                    f'(y={y:.4f}, {self.config.frame_y_min - y:.4f}m below)'
                )
            if (
                self.config.frame_y_max is not None and
                y - margin > self.config.frame_y_max
            ):
                return (
                    f'{name} moves {frame_name} above '
                    f'y={self.config.frame_y_max:.4f} '
                    f'(y={y:.4f}, {y - self.config.frame_y_max:.4f}m above)'
                )
        return ''

    def _make_frame_floor_map(self, min_z):
        """Return {frame: z} from a bound for every frame, a per-frame dict, or None ({})."""
        if min_z is None:
            return {}
        if isinstance(min_z, dict):
            return {name: float(z) for name, z in min_z.items()}
        return {name: float(min_z) for name in self.frame_names}

    def _set_workspace_corridor(self, start, goal):
        """Floor each planner frame at its lower end z, less frame_z_corridor_margin."""
        self.corridor_min_z = {}
        if not hasattr(self.robot_model, 'get_frame_position'):
            return

        margin = float(self.config.frame_z_corridor_margin)
        for frame_name in self.frame_names:
            start_pos = self.robot_model.get_frame_position(
                frame_name,
                self._motor_q(start),
            )
            goal_pos = self.robot_model.get_frame_position(
                frame_name,
                self._motor_q(goal),
            )
            self.corridor_min_z[frame_name] = (
                min(start_pos[2], goal_pos[2]) - margin
            )

    def _path_validity_failure(self, path):
        """Return why path is invalid at a waypoint or between two, or ''."""
        for idx, q in enumerate(path):
            reason = self._validity_failure(q, f'path waypoint {idx}')
            if reason:
                return reason

        # between waypoints only the workspace bounds are sampled, constraint_check_steps per
        # segment
        steps = max(1, int(self.config.constraint_check_steps))
        for idx in range(len(path) - 1):
            for sub_idx in range(1, steps):
                segment_alpha = sub_idx / steps
                q = (1.0 - segment_alpha) * path[idx]
                q += segment_alpha * path[idx + 1]
                reason = self._workspace_constraint_failure(
                    q,
                    f'path segment {idx}',
                )
                if reason:
                    return reason
        return ''

    def _infer_active_mask(self, start, goal):
        """Return a mask of the arms whose start and goal differ; None unless 14 IK joints."""
        if self.nq != _ARM_REDUCED_NQ:
            return None

        mask = np.zeros(self.nq, dtype=bool)
        if np.max(np.abs(start[:7] - goal[:7])) > _ARM_MOVE_TOL:
            mask[:7] = True
        if np.max(np.abs(start[7:] - goal[7:])) > _ARM_MOVE_TOL:
            mask[7:] = True
        return mask

    def plan(self, start, goal, active_mask=None):
        """Return a PlanResult: a valid path from start to goal (reduced q), or why there is none.

        active_mask (one bool per IK joint) says which joints may move; the others stay at
        start. Without one, a 14-joint model moves only the arms whose start and goal differ.
        The direct, smoothstep-timed path is tried first (try_direct), then OMPL.
        """
        start = self._as_reduced_array(start, 'start')
        goal = self._as_reduced_array(goal, 'goal')
        start = self._clamp_start_into_limits(start)
        if active_mask is None:
            active_mask = self._infer_active_mask(start, goal)

        self._set_active_mask(active_mask, start)
        goal = self._project_inactive(goal)
        self._set_workspace_corridor(start, goal)

        # invalid endpoints are refused before OMPL does any sampling
        for name, q in [('start', start), ('goal', goal)]:
            reason = self._validity_failure(q, name)
            if reason:
                return PlanResult(
                    success=False,
                    path=self._empty_path(),
                    reason=reason,
                )

        direct_failure = ''
        if self.config.try_direct:
            path = self._direct_path(start, goal)
            direct_failure = self._path_validity_failure(path)
            if not direct_failure:
                return PlanResult(
                    success=True,
                    path=path,
                    reason='direct path',
                )

        self.si.setStateValidityCheckingResolution(
            float(self.config.validity_resolution)
        )
        self._apply_active_bounds()
        self.ss.clear()
        self.ss.setStartAndGoalStates(
            self._make_state(start),
            self._make_state(goal),
        )

        solved = self.ss.solve(float(self.config.timeout))
        if not bool(solved):
            return PlanResult(
                success=False,
                path=self._empty_path(),
                reason=self._timeout_reason(direct_failure),
            )

        # bool(PlannerStatus) is true for APPROXIMATE_SOLUTION as well as EXACT_SOLUTION. An
        # approximate path ends at the closest state OMPL reached, not at the goal, so
        # executing it would drive the arm to a near-miss and leave the steady-state hold
        # chasing a gap it cannot close: it is a planning failure
        if not self.ss.haveExactSolutionPath():
            return PlanResult(
                success=False,
                path=self._empty_path(),
                reason=self._approximate_reason(direct_failure),
            )

        path = self.ss.getSolutionPath()
        self._shorten_path(path)
        sparse_path = self._path_to_array(path, cap=False)
        metadata = self._interpolation_metadata(sparse_path)
        self._print_interpolation_metadata(metadata, 'ompl path')
        steps = max(metadata['waypoint_count'], sparse_path.shape[0])
        path.interpolate(steps)
        path_array = self._path_to_array(path)
        path_array = self._smooth_path_timing(path_array)
        reason = self._path_validity_failure(path_array)
        if reason:
            return PlanResult(
                success=False,
                path=path_array,
                reason=reason,
            )

        return PlanResult(
            success=True,
            path=path_array,
            reason='success',
        )

    def _timeout_reason(self, direct_failure):
        """Return the reason for an OMPL timeout, with the direct path's failure."""
        reason = 'OMPL failed to find a solution before timeout'
        if direct_failure:
            reason = f'{reason}; direct path failed: {direct_failure}'
        return reason

    def _approximate_reason(self, direct_failure):
        """Return the reason for an approximate OMPL solution, with the direct path's failure."""
        reason = (
            'OMPL returned only an approximate solution '
            '(path does not reach the goal)'
        )
        if direct_failure:
            reason = f'{reason}; direct path failed: {direct_failure}'
        return reason

    def _path_to_array(self, path, cap=True):
        """Return an OMPL path as an (N, nq) array; with cap, at most max_interpolation_steps."""
        states = path.getStates()
        path_array = np.asarray(
            [self._state_to_array(state) for state in states],
            dtype=float,
        )
        max_steps = int(self.config.max_interpolation_steps)
        if cap and max_steps > 1 and path_array.shape[0] > max_steps:
            indices = np.linspace(0, path_array.shape[0] - 1, max_steps)
            indices = np.unique(indices.astype(int))
            path_array = path_array[indices]
        return path_array

    def _smooth_path_timing(self, path):
        """Return path retimed along a smoothstep, so it starts and stops slowly."""
        if len(path) < 3:
            return path

        t = np.linspace(0.0, 1.0, len(path))
        alpha = self._smoothstep(len(path))
        return np.asarray([
            np.interp(alpha, t, path[:, idx])
            for idx in range(path.shape[1])
        ]).T

    def _smoothstep(self, steps):
        """Return steps points of 3t^2 - 2t^3 over [0, 1]: zero slope at both ends."""
        t = np.linspace(0.0, 1.0, steps)
        return t * t * (3.0 - 2.0 * t)

    def _set_active_mask(self, active_mask, start):
        """Set which joints may move (None: all) and pin the rest at start."""
        if active_mask is None:
            self.active_mask = np.ones(self.nq, dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
            if active_mask.shape != (self.nq,):
                raise ValueError(f'active_mask must have shape ({self.nq},)')
            self.active_mask = active_mask
        self.pinned_q = np.array(start, dtype=float)

    def _direct_path(self, start, goal):
        """Return the straight joint path from start to goal, timed by a smoothstep."""
        coarse_path = np.vstack([start, goal])
        metadata = self._interpolation_metadata(coarse_path)
        steps = metadata['waypoint_count']
        alpha = self._smoothstep(steps).reshape(-1, 1)
        path = (1.0 - alpha) * start + alpha * goal
        path[:, ~self.active_mask] = self.pinned_q[~self.active_mask]
        self._print_interpolation_metadata(metadata, 'direct path')
        return path

    def _shorten_path(self, path, smooth=True):
        """Simplify an OMPL path (simplify), then shortcut and smooth it (shortcut).

        B-spline smoothing can overshoot into collision; plan() validates the result.
        """
        if not self.config.simplify:
            return

        self.ss.simplifySolution(float(self.config.simplify_time))

        if self.config.shortcut:
            simplifier = og.PathSimplifier(self.si)
            simplifier.ropeShortcutPath(path)
            simplifier.reduceVertices(path)
            if smooth:
                simplifier.smoothBSpline(path)
