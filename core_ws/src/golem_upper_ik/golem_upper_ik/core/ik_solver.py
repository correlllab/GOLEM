'''
IKSolver: Pink differential IK on a RobotModel's body and reduced models.

Each frame task pulls one frame to a target pose. With a self-collision model the reduced
model's IK keeps the collision pairs apart (a self-collision barrier); without one it runs
without the barrier. solve_ik_reduced tries seeds in turn, the measured pose first.
'''
import time
from dataclasses import dataclass, field

import numpy as np
import pink
import qpsolvers

from golem_upper_ik.core.robot_model import RobotModel

# named configurations solve_ik_reduced falls back to, in order, after the measured pose
SEED_CONFIGS = ('home', 't_pose', 'arms_front_45')


@dataclass
class IKAttemptResult:
    '''Result for one IK seed attempt'''

    q: np.ndarray
    success: bool
    linear_error: float
    angular_error: float
    seed_name: str
    iterations: int
    elapsed_time: float


@dataclass
class IKResult:
    '''Final IK result with all seed attempts'''

    q: np.ndarray
    success: bool
    linear_error: float
    angular_error: float
    seed_name: str
    attempts: list = field(default_factory=list)


class IKSolver:
    def __init__(self,
                 robot_model: RobotModel,
                 named_configs: dict,
                 dt: float = 0.02,
                 d_min: float = 0.05):
        '''
        Initialize the IK solver with robot model and collision parameters
        Uses model_body (no free-flyer) for IK solving

        @param robot_model: the robot model instance
        @param named_configs: name -> reduced q (one value per IK joint); the seeds
            SEED_CONFIGS come from it
        @param dt: time step for IK solving
        @param d_min: minimum distance for collision avoidance
        '''
        missing = [name for name in SEED_CONFIGS if name not in named_configs]
        if missing:
            raise ValueError(f'the IK seeds {missing} are not named configs')
        self.robot_model = robot_model
        self.named_configs = named_configs
        self.dt = dt
        self.d_min = d_min

        # task management
        self.frame_tasks = {}  # maps task_name -> FrameTask
        # basic tasks
        self.posture_task = pink.tasks.PostureTask(cost=1e-3)
        self.config_task = pink.tasks.PostureTask(cost=30.0)

        # configuration using model_body (no free-flyer)
        self.configuration = pink.Configuration(
            robot_model.model_body,
            robot_model.data_body,
            robot_model.state['q']
        )

        # limits using model_body (no free-flyer)
        self.limits = [
            pink.limits.ConfigurationLimit(robot_model.model_body),
            pink.limits.VelocityLimit(robot_model.model_body),
            pink.limits.AccelerationLimit(robot_model.model_body,
                                          25.0 * np.ones(robot_model.model_body.nv))
        ]

        # reduced-model bindings (configuration, limits, collision barrier)
        self.rebuild_reduced()

        # solver selection
        self.solver = qpsolvers.available_solvers[0]
        for preferred in ['daqp', 'proxqp', 'quadprog', 'osqp']:
            if preferred in qpsolvers.available_solvers:
                self.solver = preferred
                break

    def rebuild_reduced(self):
        '''
        Rebind all objects that hold references to the reduced model.
        Call after RobotModel.rebuild_reduced_model() so the IK, limits and
        collision barrier all see the newly locked frozen-joint reference.
        The self-collision barrier exists only with a collision model.
        '''
        robot_model = self.robot_model
        collision_kwargs = {}
        self.collision_barrier_reduced = None
        if robot_model.has_collision:
            collision_kwargs = {
                'collision_model': robot_model.collision_model_body_reduced,
                'collision_data': robot_model.collision_data_body_reduced,
            }
            self.collision_barrier_reduced = pink.barriers.SelfCollisionBarrier(
                n_collision_pairs=len(robot_model.collision_model_body_reduced.collisionPairs),
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
        self.limits_reduced = [
            pink.limits.ConfigurationLimit(robot_model.model_body_reduced),
            pink.limits.VelocityLimit(robot_model.model_body_reduced),
            pink.limits.AccelerationLimit(robot_model.model_body_reduced,
                                          25.0 * np.ones(robot_model.model_body_reduced.nv))
        ]

    @property
    def q(self):
        '''Get current joint configuration (motor-only, one per body joint)'''
        return np.copy(self.configuration.q)

    @property
    def q_reduced(self):
        '''Get current reduced joint configuration'''
        return np.copy(self.configuration_reduced.q)

    def compute_error(self, task: pink.Task, q: np.ndarray) -> np.ndarray:
        '''Compute the error for a specific task (motor-only q)'''
        configuration = pink.Configuration(
            self.robot_model.model_body,
            self.robot_model.data_body,
            q
        )
        return task.compute_error(configuration)

    def compute_error_reduced(self, task: pink.Task, q_reduced: np.ndarray) -> np.ndarray:
        '''Compute the error for a specific task on reduced model'''
        configuration_reduced = pink.Configuration(
            self.robot_model.model_body_reduced,
            self.robot_model.data_body_reduced,
            q_reduced,
        )
        return task.compute_error(configuration_reduced)

    def add_frame_task(self, task_name: str, frame_name: str,
                       position_cost: float = 50.0,
                       orientation_cost: float = 30.0,
                       lm_damping: float = 3.0):
        '''Add a frame task to the IK solver'''
        self.frame_tasks[task_name] = pink.tasks.FrameTask(
            frame_name,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=lm_damping
        )

    def remove_frame_task(self, task_name: str):
        '''Remove a frame task from the IK solver; KeyError for an unknown task'''
        del self.frame_tasks[task_name]

    def clear_frame_tasks(self):
        '''Clear all frame tasks'''
        task_names = list(self.frame_tasks.keys())
        for task_name in task_names:
            self.remove_frame_task(task_name)

    def update_configurations(self, q=None):
        '''Update Pink configurations with q (motor-only; default: measured state)'''
        if q is None:
            q = self.robot_model.state['q']
        q = np.asarray(q, dtype=float)
        self.configuration.update(q)
        self.configuration_reduced.update(q[self.robot_model.reduced_mask])

    def integrate(self, vel: np.ndarray):
        '''Integrate the current configuration with given joint velocity (motor-only)'''
        vel_reduced = vel[self.robot_model.reduced_mask]
        self.configuration.integrate_inplace(vel, self.dt)
        self.configuration_reduced.integrate_inplace(vel_reduced, self.dt)

    def _ik(self, tasks, dt):
        '''Helper function solving IK on body model for all tasks'''
        return pink.solve_ik(
            self.configuration,
            tasks,
            dt=dt,
            solver=self.solver,
            limits=self.limits,
            barriers=[],
            safety_break=False
        )

    def _ik_reduced(self, tasks, dt):
        '''Helper function solving IK on reduced model for all tasks'''
        barriers = []
        if self.collision_barrier_reduced is not None:
            barriers = [self.collision_barrier_reduced]
        return pink.solve_ik(
            self.configuration_reduced,
            tasks,
            dt=dt,
            solver=self.solver,
            limits=self.limits_reduced,
            barriers=barriers,
            safety_break=False
        )

    def _frame_task_errors(self, configuration):
        '''Compute summed linear and angular frame-task residuals'''
        linear_err = 0.0
        angular_err = 0.0
        for task in self.frame_tasks.values():
            err = task.compute_error(configuration)
            linear_err += np.linalg.norm(err[:3])
            angular_err += np.linalg.norm(err[3:])
        return linear_err, angular_err

    def _task_error_score(self, attempt, linear_threshold, angular_threshold):
        linear_scale = max(float(linear_threshold), 1e-12)
        angular_scale = max(float(angular_threshold), 1e-12)
        return attempt.linear_error / linear_scale + attempt.angular_error / angular_scale

    def _reduced_q_to_full(self, q_reduced):
        q = np.copy(self.robot_model.zero_q_body)
        q[self.robot_model.reduced_mask] = q_reduced
        return q

    def _default_seed_items(self):
        # solve_ik_reduced returns the first seed that converges, so the measured
        # pose goes first: it yields the solution nearest the arm, and the
        # named configs are fallbacks for when it does not converge.
        return [('current', self.robot_model.state['q'])] + [
            (name, self.named_configs[name]) for name in SEED_CONFIGS
        ]

    def _extra_seed_items(self, initial_positions):
        if initial_positions is None:
            return []

        try:
            seeds = np.asarray(initial_positions, dtype=float)
            if seeds.ndim == 1:
                return [('initial_0', seeds)]
            if seeds.ndim == 2:
                return [(f'initial_{idx}', seed) for idx, seed in enumerate(seeds)]
        except (TypeError, ValueError):
            pass

        items = []
        for idx, item in enumerate(initial_positions):
            if (
                isinstance(item, tuple) and
                len(item) == 2 and
                isinstance(item[0], str)
            ):
                items.append((item[0], item[1]))
            else:
                items.append((f'initial_{idx}', item))
        return items

    def _normalize_reduced_seed(self, seed, seed_name):
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

    def _seed_list(self, initial_positions):
        seeds = []
        seed_items = self._default_seed_items() + self._extra_seed_items(
            initial_positions,
        )
        for seed_name, seed in seed_items:
            seeds.append((seed_name, self._normalize_reduced_seed(seed, seed_name)))
        return seeds

    def _result_from_attempts(self, attempts, linear_threshold, angular_threshold):
        candidates = [attempt for attempt in attempts if attempt.success]
        if not candidates:
            candidates = attempts
        best = min(
            candidates,
            key=lambda attempt: self._task_error_score(
                attempt,
                linear_threshold,
                angular_threshold,
            ),
        )
        return IKResult(
            q=np.copy(best.q),
            success=best.success,
            linear_error=best.linear_error,
            angular_error=best.angular_error,
            seed_name=best.seed_name,
            attempts=attempts,
        )

    def _solve_ik_reduced_attempt(self, seed_name, q_seed, alpha, timeout,
                                  linear_threshold, angular_threshold):
        self.configuration_reduced.update(q_seed)
        linear_err, angular_err = self._frame_task_errors(
            self.configuration_reduced,
        )
        start_time = time.time()
        iterations = 0

        while (
            linear_err >= linear_threshold or
            angular_err >= angular_threshold
        ) and time.time() - start_time < timeout:
            self.posture_task.set_target_from_configuration(
                self.configuration_reduced,
            )
            tasks = list(self.frame_tasks.values()) + [self.posture_task]
            vel = self._ik_reduced(tasks, alpha)
            self.configuration_reduced.integrate_inplace(vel, alpha)
            iterations += 1
            linear_err, angular_err = self._frame_task_errors(
                self.configuration_reduced,
            )

        return IKAttemptResult(
            q=self._reduced_q_to_full(self.configuration_reduced.q),
            success=(
                linear_err < linear_threshold and
                angular_err < angular_threshold
            ),
            linear_error=linear_err,
            angular_error=angular_err,
            seed_name=seed_name,
            iterations=iterations,
            elapsed_time=time.time() - start_time,
        )

    def ik_step(self):
        '''Solve one step of IK for all tasks and return joint velocity (motor-only)'''
        self.posture_task.set_target_from_configuration(self.configuration)
        tasks = list(self.frame_tasks.values()) + [self.posture_task]
        return self._ik(tasks, self.dt)

    def ik_step_reduced(self):
        '''Solve one step of IK on reduced model for all tasks and return motor-only velocity'''
        self.posture_task.set_target_from_configuration(self.configuration_reduced)
        tasks = list(self.frame_tasks.values()) + [self.posture_task]
        # convert reduced vel to full motor vel
        vel = self._ik_reduced(tasks, self.dt)
        vel_full = self.robot_model.zero_q_body
        vel_full[self.robot_model.reduced_mask] = vel
        return vel_full

    def solve_ik_reduced(self, alpha=0.1, timeout=1.0,
                         linear_threshold=1e-3, angular_threshold=1e-2,
                         initial_positions=None):
        '''Solve IK on reduced model to convergence and return motor-only q'''
        attempts = []
        for seed_name, q_seed in self._seed_list(initial_positions):
            attempt = self._solve_ik_reduced_attempt(
                seed_name,
                q_seed,
                alpha,
                timeout,
                linear_threshold,
                angular_threshold,
            )
            attempts.append(attempt)
            if attempt.success:
                return self._result_from_attempts(
                    attempts,
                    linear_threshold,
                    angular_threshold,
                )

        result = self._result_from_attempts(
            attempts,
            linear_threshold,
            angular_threshold,
        )
        self.configuration_reduced.update(result.q[self.robot_model.reduced_mask])
        return result

    def goto_configuration(self, q: np.ndarray):
        '''Solve one step of IK to reach a target joint configuration (motor-only)'''
        self.config_task.set_target(q)
        return self._ik([self.config_task], self.dt)

    def goto_reduced_configuration(self, q_reduced: np.ndarray):
        '''Solve one step of IK to reach a target reduced joint configuration'''
        self.config_task.set_target(q_reduced)
        vel = self._ik_reduced([self.config_task], self.dt)
        # convert reduced vel to full motor vel
        vel_full = self.robot_model.zero_q_body
        vel_full[self.robot_model.reduced_mask] = vel
        return vel_full
