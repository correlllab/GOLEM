"""Run the upper-body (torso or waist, and arms) IK controller of one robot on a ROS node.

UpperController reads /lowstate through a StateSubscriber, solves the Pink IK on the robot
model's reduced model, plans with the OMPL planner where the profile has a self-collision
model, and commands golem_safety's input through a LowCmdHandler.

Every rate, wait and timeout is in s of robot time (golem_upper_ik.core.robot_clock, the
controller's clock): the sim's /clock under use_sim_time, the steady clock otherwise.

Start-up, before any command: under use_sim_time a first /clock within controller.state_timeout
of steady time, then a fresh /lowstate with finite joint positions within
controller.state_timeout, else RuntimeError and nothing published. The first command is the
measured pose with its gravity torque. With init, the controller then waits for golem_safety
to relay its commands (the follow gate opening) within controller.startup_timeout, and moves
the arms through the profile's start-up postures and the held joints to
controller.hold_targets within controller.startup_move_timeout of the gate opening. A
start-up that fails shuts the controller down (the shutdown command, then silence:
LowCmdHandler.shutdown) and raises.

Each goal re-locks the reduced model's frozen joints at the measured pose and starts the IK
from the last command (update_ik_solver). On the real robots' ROS domain (0) every control
step first checks that /lowstate is no staler than controller.max_state_lag_ms
(StaleStateError otherwise; the last command keeps going out). Commands carry the gravity
torque plus the steady-state hold's integral bias, which fades over a move.
"""
import numpy as np
from pink.exceptions import NoSolutionFound

from golem_upper_ik.config import SAFETY_OUTPUT_TOPIC
from golem_upper_ik.core.ik_solver import IKSolver
from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
from golem_upper_ik.core.planner import PlannerClient, PlannerConfig, PlannerUnavailableError
from golem_upper_ik.core.robot_clock import RobotClock
from golem_upper_ik.core.robot_model import RobotModel
from golem_upper_ik.core.ros_io import StaleStateError, StateSubscriber
from golem_upper_ik.profile import RobotProfile

__all__ = ['UpperController', 'ros_domain_id']

# The real robots' command bus; the staleness check runs only there, since the sims stamp
# /lowstate's tick in sim steps, not in ms of the robot's clock
REAL_ROBOT_DOMAIN_ID = 0
# s of robot time: /lowstate older than this is too old to seed the first command from
STATE_MAX_AGE = 0.1
# rad: the largest gap on any IK joint between the last command and the measured position at
# which a new move starts from the command. A larger gap means the arm is blocked or was
# pushed, so the move restarts from the measured position and drops the held torque bias,
# releasing the force instead of continuing it.
COMMAND_SEED_MAX_GAP = 0.1
# s: the time constant over which the steady-state torque bias fades once a move commands
# positions. Carrying it over keeps the arm from sagging at the start of a move; fading it
# hands the load back to gravity compensation as the arm leaves the pose the bias was
# integrated for.
TAU_BIAS_FADE_SEC = 1.0
# rad/s: the start-up move of the held joints to their targets, at most controller.dq_lim
HOLD_SPEED = 0.3
# rad: a start-up posture is reached when the IK's command is this close on every IK joint
STARTUP_THRESHOLD = 1e-3


def ros_domain_id(node) -> int:
    """Return the ROS domain id of node's context."""
    return node.context.get_domain_id()


class UpperController:
    """Command one robot's upper body by IK through golem_safety (module doc)."""

    def __init__(self, node, profile: RobotProfile, config: dict,
                 robot_model: RobotModel = None, init: bool = True):
        """Wait for /lowstate, command the measured pose, then run the start-up (with init).

        node: the ROS node the controller's subscriptions, publisher and logger live on; an
            executor must spin it, since the start-up waits on its /lowstate subscription
            (and, under use_sim_time, on its /clock). Its use_sim_time picks the controller's
            clock (RobotClock).
        profile, config: the robot's profile and controller config
            (golem_upper_ik.config.load_controller_config).
        robot_model: the profile's RobotModel at controller.imu_frame, default built here;
            its state source is the controller's /lowstate subscriber until shutdown().
        init: run the start-up moves.
        Raises ValueError for a config or model of another robot or IMU frame, and
        RuntimeError (after the shutdown command) when the start-up fails.
        """
        if config['robot'] != profile.robot:
            raise ValueError(f"the controller config is for robot {config['robot']}, not "
                             f'{profile.robot}')
        controller = config['controller']
        if robot_model is None:
            robot_model = RobotModel(profile, imu_frame=controller['imu_frame'])
        elif robot_model.profile.robot != profile.robot:
            raise ValueError(f'the robot model is of robot {robot_model.profile.robot}, not '
                             f'{profile.robot}')
        elif robot_model.imu_frame != controller['imu_frame']:
            raise ValueError(f'the robot model takes the IMU in {robot_model.imu_frame}, '
                             f"not in controller.imu_frame {controller['imu_frame']}")
        self.node = node
        self.profile = profile
        self.config = config
        self.robot_model = robot_model
        self._logger = node.get_logger()
        # robot time: every rate, wait and timeout of the controller counts it
        self.clock = RobotClock(node)

        self.dt = 1.0 / float(config['frequency']['ctrl_hz'])
        self.v_lim = controller['v_lim']
        self.w_lim = controller['w_lim']
        self.dq_lim = controller['dq_lim']
        self.d_min = controller['d_min']
        # multiplier on commanded arm speed: 1.0 is full speed, a goal's slow mode lowers it
        self.speed_scale = 1.0
        # N*m per body joint: the steady-state hold's integral torque bias, integrated while
        # holding and faded by _apply_joint_position while moving
        self._tau_bias = np.zeros(robot_model.num_motor)
        self._tau_bias_decay = float(np.exp(-self.dt / TAU_BIAS_FADE_SEC))
        # ms. Controlling on stale state winds up the hold's integral term and seeds moves
        # from where the arm was, so motion stops when the state is staler than this; 0 (off
        # the real robots' domain) turns the check off
        self.max_state_lag_ms = (controller['max_state_lag_ms']
                                 if ros_domain_id(node) == REAL_ROBOT_DOMAIN_ID else 0.0)
        self.state_timeout = controller['state_timeout']
        self.startup_timeout = controller['startup_timeout']
        self.startup_move_timeout = controller['startup_move_timeout']
        self.hold_targets = np.array(
            [controller['hold_targets'][joint] for joint in profile.held_joints])
        body = profile.body_joints
        self.enabled_ids = [body.index(joint) for joint in profile.enabled_joints]
        self.upper_ids = list(profile.upper_slots)
        self.left_ee_name, self.right_ee_name = profile.ee_frames
        # (joint name, q, limit) for the joints the steady-state hold is pinned against
        self.steady_state_limited_joints = []
        self._steady = None

        self.low_cmd_handler = None
        self.planner = None
        self.state_topic = config['topics']['low_state']
        self.state_subscriber = StateSubscriber(node, profile, topic=self.state_topic,
                                                clock=self.clock)
        robot_model.state_subscriber = self.state_subscriber
        try:
            # SAFETY: nothing is commanded before a fresh /lowstate, so no command starts from
            # the model's all-zero rest pose
            self._wait_for_state()
        except BaseException:
            robot_model.shutdown()
            self.clock.close()
            raise
        try:
            robot_model.update_kinematics()
            # the first command is where the robot is, gravity compensated
            self.low_cmd_handler = LowCmdHandler(node, profile, config, robot_model,
                                                 clock=self.clock)
            q_measured = np.asarray(robot_model.state['q'], dtype=np.float64)
            self.low_cmd_handler.start(
                q_init=q_measured,
                tau=robot_model.dynamics.get_gravity_compensation(q_measured))
            self.ik_solver = IKSolver(robot_model, profile.named_configs, dt=self.dt,
                                      d_min=self.d_min)
            self.planner = PlannerClient(profile, self._planner_config(),
                                         frame_names=config['planner']['frame_names'])
            if init:
                self._startup_routine()
        except BaseException:
            # SAFETY: a failed start-up after the first command sends the shutdown command,
            # then falls silent
            self.shutdown()
            raise

    def _planner_config(self):
        """Return the PlannerConfig of the planner section; dt defaults to the control period."""
        settings = {key: value for key, value in self.config['planner'].items()
                    if key not in ('enabled', 'frame_names')}
        settings.setdefault('dt', self.dt)
        return PlannerConfig(**settings)

    def _wait_for_state(self):
        """Block until a fresh /lowstate with finite joint positions; RuntimeError at timeout.

        Under use_sim_time the first /clock comes first, within state_timeout of steady time,
        since robot time starts with it.
        """
        clock = self.clock
        clock.wait_for_clock(self.state_timeout)
        subscriber = self.state_subscriber
        deadline = clock.now() + self.state_timeout
        while clock.now() < deadline:
            age = subscriber.age_s()
            if age is not None and age < STATE_MAX_AGE and \
                    np.all(np.isfinite(subscriber.state['q'])):
                return
            clock.sleep(0.01)
        raise RuntimeError(
            f'No fresh {self.state_topic} with finite joint positions within '
            f'{self.state_timeout:.1f} s on ROS domain {ros_domain_id(self.node)}; not '
            f'enabling any motor. Is the robot (or the sim) up and publishing '
            f'{self.state_topic} on this domain (ROS_DOMAIN_ID), and does the CycloneDDS '
            'configuration (CYCLONEDDS_URI) use the interface wired to it?')

    def _startup_routine(self):
        """Hold until golem_safety relays the commands, then run the start-up moves.

        The relay must start within startup_timeout; the moves (the arms through the start-up
        postures, then the held joints to their targets) must end within
        startup_move_timeout of the gate opening. RuntimeError otherwise.
        """
        self._wait_until_following(self.clock.now() + self.startup_timeout)
        # the moves get their own budget from the gate opening on: a late engagement must not
        # cut them short once golem_safety has taken over
        deadline = self.clock.now() + self.startup_move_timeout
        for name in self.profile.startup_sequence:
            self._move_to_reduced_configuration(
                self.profile.named_configs[name], deadline, desc=f'Moving {name}')
        self._move_held_joints_to_targets(deadline)

    def _wait_until_following(self, deadline):
        """Re-send the gated command, near the measured pose, until the follow gate opens."""
        handler = self.low_cmd_handler
        if handler.following:
            return
        topic = SAFETY_OUTPUT_TOPIC
        self._logger.info(f'Waiting for golem_safety to relay the commands on {topic}...')
        clock = self.clock
        while not handler.following:
            if clock.now() > deadline:
                raise RuntimeError(self._not_relayed_error(topic))
            start = clock.now()
            # the IK and the hold sit on the gated command; each step refreshes the gravity
            # torque
            self._apply_joint_position(self.ik_solver.q)
            self.update_robot_model()
            clock.sleep(max(0.0, self.dt - (clock.now() - start)))

    def _not_relayed_error(self, topic):
        """Return the start-up error for commands golem_safety never relayed, with its causes."""
        return (
            'golem_safety did not relay the commands within controller.startup_timeout '
            f'({self.startup_timeout:.0f} s): no {topic} sample carried them. golem_safety '
            f'relays only once it sees {self.state_topic} report mode_machine '
            f'{self.profile.mode_machine} (the {self.profile.robot} profile\'s) and, when started '
            'with use_estop:=true, a first sample on its estop topics; it must run the safety '
            'config this controller config pairs with, and must not have latched its estop, '
            'at this start-up (it checks its '
            f'estop limits from its first {self.state_topic} sample, so a joint already past '
            'limits.estop latches it) or in an earlier run; a latch holds until golem_safety '
            'restarts. Stopping the controller')

    def _startup_timeout_error(self, desc):
        """Return the error for a start-up move not done in time, naming the furthest joint."""
        q_cmd = self.low_cmd_handler.q_cmd[self.upper_ids]
        q = self.robot_model.state['q'][self.upper_ids]
        worst = int(np.argmax(np.abs(q_cmd - q)))
        return RuntimeError(
            f'{desc}: not done within controller.startup_move_timeout '
            f'({self.startup_move_timeout:.0f} s) of the follow gate opening '
            f'({self.profile.body_joints[self.upper_ids[worst]]} commanded '
            f'{q_cmd[worst]:.3f} rad, measured {q[worst]:.3f} rad). Stopping the controller')

    def _move_to_reduced_configuration(self, q_reduced, deadline, desc):
        """Step the IK until the command reaches q_reduced; RuntimeError at the deadline.

        Each step sleeps a whole control period after its work, as do the held joints' steps.
        """
        q_reduced = np.asarray(q_reduced, dtype=np.float64)
        self._logger.info(f'{desc}...')
        while True:
            try:
                self.goto_reduced_configuration(q_reduced)
            except NoSolutionFound as error:
                raise self._no_ik_solution_error(desc, error) from error
            if np.max(np.abs(self.ik_solver.q_reduced - q_reduced)) < STARTUP_THRESHOLD:
                return
            if self.clock.now() > deadline:
                raise self._startup_timeout_error(desc)
            self.clock.sleep(self.dt)

    def _no_ik_solution_error(self, desc, error):
        """Return the error for a start-up IK step with no solution, and log the closest pairs.

        The usual cause is a self-collision pair the IK starts well inside controller.d_min,
        which the barrier cannot push out in one step.
        """
        pairs = self.robot_model.closest_collision_pairs(self.ik_solver.q_reduced)
        if pairs:
            closest = '; '.join(f'{first} - {second} {distance * 1000:.1f} mm'
                                for first, second, distance in pairs)
            inside = sum(distance < self.d_min for _, _, distance in pairs)
            self._logger.error(
                f'{desc}: the IK has no solution; closest self-collision pairs (controller.d_min '
                f'{self.d_min * 1000:.0f} mm): {closest}')
            if inside:
                cause = (f'{inside} of the closest self-collision pairs are inside '
                         f'controller.d_min ({self.d_min * 1000:.0f} mm), closest '
                         f'{pairs[0][0]} - {pairs[0][1]} at {pairs[0][2] * 1000:.1f} mm: the '
                         'arms start too close to the body or each other; move them clear')
            else:
                cause = (f'every self-collision pair keeps controller.d_min '
                         f'({self.d_min * 1000:.0f} mm), closest {pairs[0][2] * 1000:.1f} mm')
        else:
            cause = 'the robot has no self-collision model'
        return RuntimeError(f'{desc}: the IK has no solution ({error}); {cause}. Stopping the '
                            'controller')

    def _move_held_joints_to_targets(self, deadline):
        """Ease the held joints (the torso, the waist) to exactly their hold targets."""
        held = ', '.join(self.profile.held_joints)
        desc = f'Moving {held} to the hold targets'
        handler = self.low_cmd_handler
        step = max(min(HOLD_SPEED, self.dq_lim) * self.dt, 1e-6)
        targets = self.hold_targets
        self._logger.info(f'{desc}...')
        while np.any(handler.hold != targets):
            if self.clock.now() > deadline:
                raise self._startup_timeout_error(desc)
            remaining = targets - handler.hold
            # the last step lands on the targets themselves
            handler.set_hold(np.where(np.abs(remaining) <= step, targets,
                                      handler.hold + np.clip(remaining, -step, step)))
            self._apply_joint_position(self.ik_solver.q)
            self.update_robot_model()
            self.clock.sleep(self.dt)

    def ee_transformations(self):
        """Return the left and right end effectors' 4x4 placements in the pelvis frame.

        From the reduced model at one measured state, on data of their own, so a call from
        another thread leaves the IK's data alone.
        """
        q_reduced = self.robot_model.state_reduced['q']
        return tuple(self.robot_model.get_frame_transformation_reduced(name, q_reduced)
                     for name in (self.left_ee_name, self.right_ee_name))

    def check_state_fresh(self):
        """Raise StaleStateError if /lowstate is staler than max_state_lag_ms (when on)."""
        if self.max_state_lag_ms <= 0.0:
            return
        staleness_ms = self.robot_model.state_staleness_ms
        if staleness_ms > self.max_state_lag_ms:
            raise StaleStateError(
                f'{self.state_topic} is {staleness_ms:.0f} ms stale '
                f'(> {self.max_state_lag_ms:.0f} ms); holding the last command')

    def update_ik_solver(self):
        """Ready the IK for a new goal: re-lock the frozen joints and seed from the command."""
        self.check_state_fresh()
        # re-lock the frozen joints (the torso or waist, the legs) at their measured angles so
        # reduced-model FK agrees with the robot (and the TF tree): frozen at zero, a torso
        # holding -0.04 rad offsets every arm frame by a constant ~15 mm the IK cannot remove
        self.robot_model.rebuild_reduced_model()
        self.ik_solver.rebuild_reduced()
        # start the IK from the last command, not the measured state: the arm sits below its
        # command (gravity, PD lag), and restarting from the measured position would step the
        # command down and drop the arm
        q_seed, from_command = self._command_seed()
        if not from_command:
            self._tau_bias[:] = 0.0
        self.ik_solver.update_configurations(q_seed)

    def _command_seed(self):
        """Return (q, from_command): the body configuration a new move starts from.

        The IK joints take the last commanded position when it is within COMMAND_SEED_MAX_GAP
        of the measured position on every one of them; otherwise the whole seed is the
        measured state.
        """
        q_seed = np.array(self.robot_model.state['q'], dtype=np.float64)
        q_cmd = self.low_cmd_handler.q_cmd
        ids = self.enabled_ids
        gap = np.max(np.abs(q_cmd[ids] - q_seed[ids]))
        if not np.isfinite(gap) or gap > COMMAND_SEED_MAX_GAP:
            self._logger.info(f'Command-to-measured gap {gap:.3f} rad > '
                              f'{COMMAND_SEED_MAX_GAP} rad; starting from the measured state')
            return q_seed, False
        q_seed[ids] = q_cmd[ids]
        return q_seed, True

    @property
    def planning_refusal(self):
        """Return why this controller cannot plan, or None."""
        if not self.planner.available:
            return self.planner.unavailable_reason
        if not self.config['planner']['enabled']:
            return (f"Planning is turned off: the controller config {self.config['source']} "
                    'sets planner.enabled: false; send the goal with plan=false')
        return None

    def _check_planning(self):
        """Raise PlannerUnavailableError when this controller cannot plan."""
        refusal = self.planning_refusal
        if refusal is not None:
            raise PlannerUnavailableError(refusal)

    def _as_reduced_path(self, path):
        """Return path as an (N, IK joints) array of finite waypoints; ValueError otherwise."""
        path = np.asarray(path, dtype=float)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        nq = self.robot_model.model_body_reduced.nq
        if path.ndim != 2 or path.shape[1] != nq:
            raise ValueError(f'Path must have shape (N, {nq})')
        if not np.all(np.isfinite(path)):
            raise ValueError('Path must contain only finite values')
        return path

    @staticmethod
    def _path_or_raise(result):
        """Return a PlanResult's path; RuntimeError with its reason if planning failed."""
        if not result.success:
            raise RuntimeError(f'Planning failed: {result.reason}')
        return result.path

    def _planner_speed_overrides(self):
        """Return (moving_speed, max_interpolation_steps) for speed_scale; (None, None) at 1.

        A planned path is slowed by planning it at a lower planner-frame speed: the planner
        then emits proportionally more waypoints and execute_path still clocks them out at
        dt. Stretching the step period instead keeps the original waypoint spacing and turns
        a smooth ramp into a staircase. At full speed the planner process keeps the config it
        was started with.
        """
        scale = float(self.speed_scale)
        if scale == 1.0:
            return None, None
        config = self.planner.planner_config
        # the waypoint cap bounds a path's duration, so it has to grow with the step count or
        # a long slow path clamps back to near full speed
        return (config.moving_speed * scale,
                int(np.ceil(config.max_interpolation_steps / scale)))

    def plan_to_configuration(self, q_reduced, start=None, active_mask=None):
        """Return a planned reduced path from start (default: the command seed) to q_reduced.

        PlannerUnavailableError when the controller cannot plan; RuntimeError when planning
        fails.
        """
        self._check_planning()
        if start is None:
            start = self._command_seed()[0][self.robot_model.reduced_mask]
        moving_speed, max_interpolation_steps = self._planner_speed_overrides()
        result = self.planner.plan_configuration(
            current_q=self.robot_model.state['q'],
            start=start,
            goal=np.asarray(q_reduced, dtype=float),
            active_mask=active_mask,
            moving_speed=moving_speed,
            max_interpolation_steps=max_interpolation_steps,
        )
        return self._path_or_raise(result)

    def plan_to_ik_target(self, start=None, ik_timeout=1.0, ik_alpha=0.1):
        """Return a planned reduced path to the IK solution of the current frame tasks.

        PlannerUnavailableError, before any IK, when the controller cannot plan; RuntimeError
        when the IK or the planning fails. Only the joints that move the tasks' frames plan.
        """
        self._check_planning()
        # restrict planning to the joints supporting the active tasks, so untasked limbs
        # (the other arm) never move during the plan
        active_mask = self._active_task_mask()
        self.update_ik_solver()
        ik_result = self.ik_solver.solve_ik_reduced(alpha=ik_alpha, timeout=ik_timeout)
        if not ik_result.success:
            raise RuntimeError('IK failed to resolve current targets')
        q_goal = ik_result.q[self.robot_model.reduced_mask]
        return self.plan_to_configuration(q_goal, start=start, active_mask=active_mask)

    def _active_task_mask(self):
        """Return the reduced-joint mask of the joints that move any frame task's frame."""
        mask = np.zeros(self.robot_model.model_body_reduced.nq, dtype=bool)
        for task in self.ik_solver.frame_tasks.values():
            mask |= self._reduced_support_mask(task.frame)
        return mask

    def _reduced_support_mask(self, frame_name):
        """Return the reduced-joint mask of the joints between the root and frame_name."""
        model = self.robot_model.model_body_reduced
        joint_id = model.frames[model.getFrameId(frame_name)].parentJoint
        mask = np.zeros(model.nq, dtype=bool)
        while joint_id > 0:
            idx_q = model.joints[joint_id].idx_q
            mask[idx_q:idx_q + model.joints[joint_id].nq] = True
            joint_id = model.parents[joint_id]
        return mask

    def execute_path(self, path, validate=True):
        """Command reduced waypoints as positions, one per control period.

        validate checks each waypoint's limits and self-collision first (ValueError). Refused
        (RuntimeError) while the follow gate is shut.
        """
        if not self.low_cmd_handler.following:
            # SAFETY: absolute waypoints would run ahead of a robot that is not following
            raise RuntimeError("Not executing a path before golem_safety relays the "
                               "controller's commands (follow gate shut)")
        path = self._as_reduced_path(path)
        # The path is time-parameterized at dt by the planner (slow moves are planned at a
        # lower moving_speed and come back with more waypoints), so it always plays back at
        # the control period: each step sleeps the period less the work already done, or
        # the checks and FK would stretch a path past its planned duration
        clock = self.clock
        for q_reduced in path:
            start = clock.now()
            self.check_state_fresh()
            if validate:
                if not self.robot_model.check_within_limits_reduced(q_reduced):
                    raise ValueError('Path waypoint is outside joint limits')
                if not self.robot_model.check_collision_free_reduced(q_reduced):
                    raise ValueError('Path waypoint is in self-collision')
            q = np.array(self.robot_model.state['q'], dtype=np.float64)
            q[self.robot_model.reduced_mask] = q_reduced
            self._apply_joint_position(q)
            self.update_robot_model()
            clock.sleep(max(0.0, self.dt - (clock.now() - start)))

    def update_robot_model(self):
        """Update the robot model's kinematics from the measured state."""
        self.robot_model.update_kinematics()

    @property
    def reduced_configuration_error(self):
        """Return the measured IK joints' error from the joint-space goal, rad."""
        return self.ik_solver.compute_error_reduced(
            self.ik_solver.config_task, self.robot_model.state_reduced['q'])

    def set_reduced_configuration_target(self, q_reduced):
        """Set the joint-space goal that reduced_configuration_error measures against."""
        self.ik_solver.config_task.set_target(q_reduced)

    def goto_reduced_configuration(self, q_reduced):
        """Command one IK step toward q_reduced; return the body joint velocity, rad/s."""
        self.check_state_fresh()
        vel = self.ik_solver.goto_reduced_configuration(q_reduced)
        return self._apply_velocity_command(self._limit_joint_vel(vel))

    def control_step_reduced(self):
        """Command one IK step on every frame task; return the body joint velocity, rad/s."""
        self.check_state_fresh()
        vel = self.ik_solver.ik_step_reduced()
        return self._apply_velocity_command(self._limit_joint_vel(vel))

    def init_steady_state(self):
        """Hold the last command with integral control from the next steady_state_step.

        The integration resumes from the current torque bias, so the hand-off from a move to
        a hold is continuous.
        """
        limits = self.config['limits']
        self._steady = {
            'q': self.low_cmd_handler.q_cmd.astype(np.float64),
            'dq': np.zeros(self.robot_model.num_motor),
            'ki': np.asarray(self.config['gains']['ki'], dtype=np.float64),
            'tau_bias_limit': np.asarray(limits['tau_clip_limits'], dtype=np.float64),
            'q_clip_limits': np.asarray(limits['q_clip_limits'], dtype=np.float64),
        }
        self.steady_state_limited_joints = []

    def _steady_state_limited_joints(self, q_state, tau_bias_increment, threshold):
        """Return (joint, q, limit) for each upper joint the integral term pushes into a limit.

        A joint counts within threshold (rad) of its position clip limit.
        """
        limits = self._steady['q_clip_limits']
        limited = []
        for i in self.upper_ids:
            low, high = limits[i]
            if q_state[i] >= high - threshold and tau_bias_increment[i] > 0.0:
                limited.append((self.profile.body_joints[i], float(q_state[i]), float(high)))
            elif q_state[i] <= low + threshold and tau_bias_increment[i] < 0.0:
                limited.append((self.profile.body_joints[i], float(q_state[i]), float(low)))
        return limited

    def steady_state_step(self, threshold=1e-3):
        """Command one integral-control tick of the hold; return whether it has converged.

        Converged: every upper joint within threshold (rad) of its held command. A hold pinned
        against a joint position limit cannot converge: the integral term is frozen,
        steady_state_limited_joints names the joints, and this returns False.
        """
        self.check_state_fresh()
        if self._steady is None:
            self.init_steady_state()
        steady = self._steady
        q_state = np.asarray(self.robot_model.state['q'], dtype=np.float64)
        tau_bias_increment = steady['ki'] * (steady['q'] - q_state) * self.dt
        self.steady_state_limited_joints = self._steady_state_limited_joints(
            q_state, tau_bias_increment, threshold)
        if not self.steady_state_limited_joints:
            self._tau_bias = np.clip(self._tau_bias + tau_bias_increment,
                                     -steady['tau_bias_limit'], steady['tau_bias_limit'])
        tau = self.robot_model.dynamics.get_gravity_compensation(q_state) + self._tau_bias
        self._command(steady['q'], steady['dq'], tau)
        self.update_robot_model()
        if self.steady_state_limited_joints:
            return False
        q_error = self._steady['q'] - self.robot_model.state['q']
        return bool(np.max(np.abs(q_error[self.upper_ids])) < threshold)

    def _limit_joint_vel(self, vel):
        """Return vel with each arm scaled under dq_lim, v_lim and w_lim, then by speed_scale.

        Each arm's joints are scaled together, so its direction is kept; the other joints
        pass through.
        """
        dynamics = self.robot_model.dynamics
        vel_scaled = np.array(vel, dtype=np.float64)
        for frame, arm in ((self.left_ee_name, self.robot_model.left_arm_index),
                           (self.right_ee_name, self.robot_model.right_arm_index)):
            twist = dynamics.compute_frame_twist(frame, vel)
            scaler = min([
                1.0,
                np.min(self.dq_lim / (np.abs(vel[arm]) + 1e-6)),
                self.v_lim / (np.linalg.norm(twist[:3]) + 1e-6),
                self.w_lim / (np.linalg.norm(twist[3:]) + 1e-6),
            ])
            vel_scaled[arm] = self.speed_scale * scaler * vel[arm]
        return vel_scaled

    def _apply_joint_position(self, q):
        """Command q with dq 0 and the gravity torque plus the bias, which fades by one step."""
        tau = self.robot_model.dynamics.get_gravity_compensation(self.robot_model.state['q'])
        tau = tau + self._tau_bias
        self._tau_bias *= self._tau_bias_decay
        self._command(q, np.zeros(self.robot_model.num_motor), tau)

    def _apply_velocity_command(self, vel):
        """Integrate vel into the IK, command the result, and return vel."""
        self.ik_solver.integrate(vel)
        self._apply_joint_position(self.ik_solver.q)
        self.update_robot_model()
        return vel

    def _command(self, q, dq, tau):
        """Send q, dq and tau to the handler, then sync the IK to what the gate let through."""
        self.low_cmd_handler.set_joint_commands(q, dq, tau)
        self._sync_to_command()

    def _sync_to_command(self):
        """Restart the IK and the hold target from the gated command until golem_safety relays.

        So none of them runs ahead of the robot while the follow gate is shut (the held
        joints' hold restarts in the handler); once it is open, nothing is synced.
        """
        if self.low_cmd_handler.following:
            return
        q_cmd = self.low_cmd_handler.q_cmd.astype(np.float64)
        self.ik_solver.configuration.update(q_cmd)
        self.ik_solver.configuration_reduced.update(q_cmd[self.robot_model.reduced_mask])
        if self._steady is not None:
            self._steady['q'] = q_cmd

    def shutdown(self):
        """Send the shutdown command and stop publishing, then stop the planner and the state.

        SAFETY: the shutdown command (LowCmdHandler.shutdown) goes first, since the planner's
        own shutdown may wait out a plan in progress; publishing stops before the state source
        is detached, since a detached source would leave the follow gate nothing to hold the
        commands near. The planner and the state source are stopped even when an earlier step
        raises, so no planner subprocess outlives the controller; the error propagates.
        Then the robot clock closes, after the publisher has stopped: a goal waiting on a
        stopped /clock ends with an error instead of waiting for ever. Calling it again does
        nothing more.
        """
        try:
            if self.low_cmd_handler is not None:
                self.low_cmd_handler.shutdown()
        finally:
            try:
                if self.planner is not None:
                    self.planner.shutdown()
            finally:
                try:
                    self.robot_model.shutdown()
                finally:
                    self.clock.close()
