"""Solve Pink differential IK on a RobotModel's reduced model (the IK joints).

IKSolver keeps two configurations: one on the reduced model, which the IK steps, and one on
the body model (no free-flyer), which carries every body joint's q. Each frame task pulls one
frame to a target pose. With a self-collision model the reduced model's IK keeps the
collision pairs controller.d_min apart (a self-collision barrier, on the BARRIER_PAIRS
closest pairs of each step); without one it runs without the barrier. solve_ik_reduced tries seeds in turn, the measured pose first.

The IK keeps every IK joint within the reduced model's position limits (RobotModel's range:
the controller's clip range). A configuration or seed given outside them is clamped into them
first, and so is every integrated step (the QP solver meets Pink's configuration limit only to
its tolerance, about 1e-7 rad, and Pink's acceleration limit takes the square root of the
distance to the limit, which is not a number past it), so every q the solver returns or
integrates to lies within them. The stepped moves (ik_step_reduced, goto_reduced_configuration)
brake into the limits: a joint closes on its limit at no more than its distance over
LIMIT_APPROACH_TIME; solve_ik_reduced's iterations keep Pink's default gain. A stepped move's
step is otherwise the whole step to its goal within the model's velocity limits: the
controller scales it to its speed and acceleration limits.
"""
import time
from dataclasses import dataclass

import numpy as np
import pink
import qpsolvers

from golem_upper_ik.core.robot_model import RobotModel

# named configurations solve_ik_reduced falls back to, in order, after the measured pose;
# every robot's profile must name them
SEED_CONFIGS = ('home', 't_pose', 'arms_front_45')
# solve_ik_reduced converges when the summed frame-task errors are below these
LINEAR_THRESHOLD = 1e-3  # m
ANGULAR_THRESHOLD = 1e-2  # rad
# The self-collision barrier's pairs: Pink computes every pair's distance and constrains
# this many closest ones in each IK step (their Jacobians are most of a step's work; all of
# both robots' ~210 pairs took ~22 ms a step). SAFETY: a pair outside them is farther than
# every constrained one; in the sphere models' postures at most 26 pairs lie within 0.15 m,
# and one step moves a pair a few mm.
BARRIER_PAIRS = 32
# s: the time constant of an IK joint's approach to its position limit in the stepped moves
# (ik_step_reduced, goto_reduced_configuration): each step closes at most 1 - exp(-dt / this)
# of the distance, so a joint slows into its limit at no more than distance / this rad/s
# (0.27 rad/s at 50 mrad, 0.11 at 20) instead of arriving at dq_lim. In the H1-2 RoboCasa sim,
# where the skills' failure-retreat lift brings the shoulder yaw's command to the clip, the
# yaw went 16.7 mrad past the clip without it and 9.9 to 14.4 mrad with it (8 lifts), and
# that lift took 3.1 s instead of 2.1 s. A command already parked on the clip is not braked:
# with it, one of 20 grasps still took the yaw 20.3 mrad past from there. Stepping the
# H1-2's IK on 30 direct goals that stay 0.2 rad inside the range took no longer with 0.2 s,
# 0.03 s longer at most with 0.3 s and 0.1 s with 0.5 s. Unmeasured on the real robots
LIMIT_APPROACH_TIME = 0.2


@dataclass
class IKResult:
    """Hold the outcome of the IK from one seed, as solve_ik_reduced returns it.

    q is per body joint (the frozen joints 0); the errors are the frame tasks' summed
    residuals, m and rad.
    """

    q: np.ndarray
    success: bool
    linear_error: float
    angular_error: float
    seed_name: str


class IKSolver:
    """Step and solve Pink IK on a RobotModel's reduced model, with frame and posture tasks."""

    def __init__(self,
                 robot_model: RobotModel,
                 named_configs: dict,
                 dt: float = 0.02,
                 d_min: float = 0.05):
        """Set up the tasks, the configurations at the measured state, and the solver.

        robot_model: the model the IK solves on; its measured state seeds the configurations.
        named_configs: name -> reduced q (one value per IK joint); must name SEED_CONFIGS,
            else ValueError before the model is touched.
        dt: s per IK step (ik_step_reduced, goto_reduced_configuration, integrate).
        d_min: m the self-collision barrier keeps between sphere pairs.
        """
        missing = [name for name in SEED_CONFIGS if name not in named_configs]
        if missing:
            raise ValueError(f'the IK seeds {missing} are not named configs')
        self.robot_model = robot_model
        self.named_configs = named_configs
        self.dt = dt
        self.d_min = d_min

        self.frame_tasks = {}  # task name -> pink FrameTask
        # a light regulariser holding the current posture, and the joint-space goal task
        self.posture_task = pink.tasks.PostureTask(cost=1e-3)
        self.config_task = pink.tasks.PostureTask(cost=30.0)

        self.configuration = pink.Configuration(
            robot_model.model_body,
            robot_model.data_body,
            robot_model.state['q']
        )

        self.rebuild_reduced()

        # the first of these QP solvers installed, else whichever qpsolvers has
        self.solver = qpsolvers.available_solvers[0]
        for preferred in ['daqp', 'proxqp', 'quadprog', 'osqp']:
            if preferred in qpsolvers.available_solvers:
                self.solver = preferred
                break

    def rebuild_reduced(self):
        """Rebind the reduced configuration, limits and barrier to the model's reduced model.

        Call after RobotModel.rebuild_reduced_model(), so the IK sees the frozen joints where
        they were re-locked. The reduced configuration restarts at the measured state. The
        self-collision barrier exists only with a self-collision model.
        """
        robot_model = self.robot_model
        collision_kwargs = {}
        self.collision_barrier_reduced = None
        if robot_model.has_collision:
            collision_kwargs = {
                'collision_model': robot_model.collision_model_body_reduced,
                'collision_data': robot_model.collision_data_body_reduced,
            }
            self.collision_barrier_reduced = pink.barriers.SelfCollisionBarrier(
                n_collision_pairs=min(
                    BARRIER_PAIRS, len(robot_model.collision_model_body_reduced.collisionPairs)),
                gain=20.0,
                safe_displacement_gain=1.0,
                d_min=self.d_min,
            )
        self.configuration_reduced = pink.Configuration(
            robot_model.model_body_reduced,
            robot_model.data_body_reduced,
            robot_model.state_reduced['q'],
            **collision_kwargs,
        )
        model = robot_model.model_body_reduced
        # solve_ik_reduced's iterations: a search for the goal, not a motion, so Pink's default
        # configuration-limit gain (0.5 of the distance per iteration), and each iteration's
        # step capped at 25 rad/s^2 times the iteration step squared
        self.limits_reduced = [
            pink.limits.ConfigurationLimit(model),
            pink.limits.VelocityLimit(model),
            pink.limits.AccelerationLimit(model, 25.0 * np.ones(model.nv)),
        ]
        # the stepped moves the controller commands: braked into the limits, and otherwise the
        # whole step to the goal the velocity limits allow, which the controller takes as the
        # distance left (UpperController's acceleration limit scales it)
        self.step_limits_reduced = [
            pink.limits.ConfigurationLimit(
                model, config_limit_gain=-np.expm1(-self.dt / LIMIT_APPROACH_TIME)),
            pink.limits.VelocityLimit(model),
        ]

    @property
    def q(self):
        """Return a copy of the body configuration's q, one value per body joint."""
        return np.copy(self.configuration.q)

    @property
    def q_reduced(self):
        """Return a copy of the reduced configuration's q, one value per IK joint."""
        return np.copy(self.configuration_reduced.q)

    def add_frame_task(self, task_name: str, frame_name: str,
                       position_cost: float = 50.0,
                       orientation_cost: float = 30.0,
                       lm_damping: float = 3.0,
                       target=None):
        """Add a task pulling frame_name to target (a pin.SE3 in the pelvis frame) when given.

        The task joins frame_tasks only once its target is set, so a reader in another thread
        never finds it without one.
        """
        task = pink.tasks.FrameTask(
            frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping
        )
        if target is not None:
            task.transform_target_to_world = target
        self.frame_tasks[task_name] = task

    def clear_frame_tasks(self):
        """Remove every frame task."""
        for task_name in list(self.frame_tasks.keys()):
            del self.frame_tasks[task_name]

    def _clamp_reduced(self, q_reduced):
        """Return reduced q clamped into the reduced model's position limits."""
        model = self.robot_model.model_body_reduced
        return np.clip(q_reduced, model.lowerPositionLimit, model.upperPositionLimit)

    def update_configurations(self, q=None):
        """Set both configurations from body q (default: the measured state).

        The IK joints are clamped into the position limits first: a measured pose past them
        (a joint pushed or overshooting) restarts the IK at the limit.
        """
        if q is None:
            q = self.robot_model.state['q']
        q = np.array(q, dtype=float)
        q[self.robot_model.reduced_mask] = self._clamp_reduced(q[self.robot_model.reduced_mask])
        self.configuration.update(q)
        self.configuration_reduced.update(q[self.robot_model.reduced_mask])

    def integrate(self, vel: np.ndarray):
        """Advance both configurations by body joint velocity vel (rad/s) over dt.

        The IK joints end clamped into the position limits.
        """
        mask = self.robot_model.reduced_mask
        q = self.configuration.integrate(vel, self.dt)
        q[mask] = self._clamp_reduced(q[mask])
        self.configuration.update(q)
        self.configuration_reduced.update(
            self._clamp_reduced(self.configuration_reduced.integrate(vel[mask], self.dt)))

    def _ik_reduced(self, tasks, dt, limits=None):
        """Return the reduced joint velocity Pink solves for tasks over dt, with the barrier.

        limits: the Pink limits; default limits_reduced (the solve's).
        """
        barriers = []
        if self.collision_barrier_reduced is not None:
            barriers = [self.collision_barrier_reduced]
        return pink.solve_ik(
            self.configuration_reduced,
            tasks,
            dt=dt,
            solver=self.solver,
            limits=self.limits_reduced if limits is None else limits,
            barriers=barriers,
            safety_break=False
        )

    def _frame_task_errors(self, configuration):
        """Return the frame tasks' summed position (m) and orientation (rad) errors."""
        linear_err = 0.0
        angular_err = 0.0
        for task in self.frame_tasks.values():
            err = task.compute_error(configuration)
            linear_err += np.linalg.norm(err[:3])
            angular_err += np.linalg.norm(err[3:])
        return linear_err, angular_err

    @staticmethod
    def _task_error_score(attempt):
        """Return an attempt's errors in units of the convergence thresholds, summed."""
        return (attempt.linear_error / LINEAR_THRESHOLD
                + attempt.angular_error / ANGULAR_THRESHOLD)

    def _reduced_q_to_full(self, q_reduced):
        """Return body q with the IK joints at q_reduced and every other joint at 0."""
        q = np.copy(self.robot_model.zero_q_body)
        q[self.robot_model.reduced_mask] = q_reduced
        return q

    def _default_seed_items(self):
        """Return (name, q) seeds: the measured pose ('current'), then SEED_CONFIGS.

        solve_ik_reduced returns the first seed that converges, so the measured pose goes
        first: it yields the solution nearest the arm, and the named configs are fallbacks.
        """
        return [('current', self.robot_model.state['q'])] + [
            (name, self.named_configs[name]) for name in SEED_CONFIGS
        ]

    def _normalize_reduced_seed(self, seed, seed_name):
        """Return a body or reduced seed as reduced q; ValueError if mis-shaped or non-finite."""
        q = np.asarray(seed, dtype=float).reshape(-1)
        if q.shape == (self.robot_model.model_body_reduced.nq,):
            normalized = q
        elif q.shape == (self.robot_model.model_body.nq,):
            normalized = q[self.robot_model.reduced_mask]
        else:
            raise ValueError(
                f'{seed_name} seed must have shape '
                f'({self.robot_model.model_body_reduced.nq},) or '
                f'({self.robot_model.model_body.nq},)'
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f'{seed_name} seed must contain only finite values')
        return normalized

    def _seed_list(self):
        """Return the seeds as (name, reduced q), each clamped into the position limits."""
        return [(seed_name, self._clamp_reduced(self._normalize_reduced_seed(seed, seed_name)))
                for seed_name, seed in self._default_seed_items()]

    def _best_attempt(self, attempts):
        """Return the converged attempt with the least error, else the attempt with the least."""
        candidates = [attempt for attempt in attempts if attempt.success]
        if not candidates:
            candidates = attempts
        return min(candidates, key=self._task_error_score)

    def _solve_ik_reduced_attempt(self, seed_name, q_seed, alpha, timeout):
        """Iterate the IK from q_seed in steps of alpha until it converges or timeout.

        timeout is s of steady time: a budget of host compute, not of robot time, so a sim's
        stopped /clock cannot make it endless.
        """
        self.configuration_reduced.update(q_seed)
        linear_err, angular_err = self._frame_task_errors(
            self.configuration_reduced,
        )
        start_time = time.monotonic()

        while (
            linear_err >= LINEAR_THRESHOLD or
            angular_err >= ANGULAR_THRESHOLD
        ) and time.monotonic() - start_time < timeout:
            self.posture_task.set_target_from_configuration(
                self.configuration_reduced,
            )
            tasks = list(self.frame_tasks.values()) + [self.posture_task]
            vel = self._ik_reduced(tasks, alpha)
            self.configuration_reduced.update(
                self._clamp_reduced(self.configuration_reduced.integrate(vel, alpha)))
            linear_err, angular_err = self._frame_task_errors(
                self.configuration_reduced,
            )

        return IKResult(
            q=self._reduced_q_to_full(self.configuration_reduced.q),
            success=(
                linear_err < LINEAR_THRESHOLD and
                angular_err < ANGULAR_THRESHOLD
            ),
            linear_error=linear_err,
            angular_error=angular_err,
            seed_name=seed_name,
        )

    def ik_step_reduced(self):
        """Return one IK step's body joint velocity (rad/s) for the frame tasks; others 0."""
        self.posture_task.set_target_from_configuration(self.configuration_reduced)
        tasks = list(self.frame_tasks.values()) + [self.posture_task]
        vel = self._ik_reduced(tasks, self.dt, self.step_limits_reduced)
        vel_full = self.robot_model.zero_q_body
        vel_full[self.robot_model.reduced_mask] = vel
        return vel_full

    def solve_ik_reduced(self, alpha=0.1, timeout=1.0):
        """Return an IKResult: the IK iterated to convergence from each seed in turn.

        alpha is the iteration step, s; timeout bounds each seed's iterations, s of steady
        time. A converged seed ends the search; without one, the reduced configuration is left
        at the best.
        """
        attempts = []
        for seed_name, q_seed in self._seed_list():
            attempt = self._solve_ik_reduced_attempt(seed_name, q_seed, alpha, timeout)
            attempts.append(attempt)
            if attempt.success:
                return self._best_attempt(attempts)

        result = self._best_attempt(attempts)
        self.configuration_reduced.update(result.q[self.robot_model.reduced_mask])
        return result

    def limited_step(self, step, low, high):
        """Return the reduced step (rad) nearest step within [low, high], or None if none is.

        The step also keeps the stepped moves' limits (step_limits_reduced: the position limits'
        brake and the velocity limits) and the self-collision barrier at the reduced
        configuration, as an IK step does; nearest is in joint space. None when no step keeps
        them all (the QP solver finds none).
        """
        barriers = []
        if self.collision_barrier_reduced is not None:
            barriers = [self.collision_barrier_reduced]
        problem = pink.build_ik(self.configuration_reduced, [], self.dt,
                                limits=self.step_limits_reduced, barriers=barriers)
        step = np.asarray(step, dtype=np.float64)
        return qpsolvers.solve_qp(np.eye(step.size), -step, problem.G, problem.h,
                                  lb=np.asarray(low, dtype=np.float64),
                                  ub=np.asarray(high, dtype=np.float64), solver=self.solver)

    def goto_reduced_configuration(self, q_reduced: np.ndarray):
        """Return one IK step's body joint velocity (rad/s) toward q_reduced; others 0."""
        self.config_task.set_target(q_reduced)
        vel = self._ik_reduced([self.config_task], self.dt, self.step_limits_reduced)
        vel_full = self.robot_model.zero_q_body
        vel_full[self.robot_model.reduced_mask] = vel
        return vel_full
