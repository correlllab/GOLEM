'''
UpperController: the upper-body (waist or torso, and arms) IK controller of one robot, on a
ROS node. It reads /lowstate through a StateSubscriber, solves the Pink IK on the robot
model's reduced model, plans with the OMPL planner where the profile has a self-collision
model, and commands golem_safety's input through a LowCmdHandler.

Start-up, before any command: a fresh /lowstate with finite joint positions within
controller.state_timeout, else RuntimeError and nothing published. The first command is the
measured pose with its gravity torque. With init, the controller then waits for golem_safety
to relay its commands (the follow gate opening) within controller.startup_timeout, moves the
arms through the profile's start-up postures and the held joints to controller.hold_targets,
within controller.startup_move_timeout of the gate opening. A start-up that fails shuts the
controller down (one zero-gain command, then silence) and raises.

Each goal re-locks the reduced model's frozen joints at the measured pose and starts the IK
from the last command (update_ik_solver). On the real robots' ROS domain (0) every control
step first checks that /lowstate is no staler than controller.max_state_lag_ms
(StaleStateError otherwise; the last command keeps going out). Commands carry the gravity
torque plus the steady-state hold's integral bias, which fades over a move.
'''
import time

import numpy as np

from golem_upper_ik.config import SAFETY_OUTPUT_TOPICS
from golem_upper_ik.core.ik_solver import IKSolver
from golem_upper_ik.core.low_cmd_handler import LowCmdHandler
from golem_upper_ik.core.planner import PlannerClient, PlannerConfig, PlannerUnavailableError
from golem_upper_ik.core.robot_model import RobotModel
from golem_upper_ik.core.ros_io import StaleStateError, StateSubscriber
from golem_upper_ik.profile import RobotProfile

__all__ = ['UpperController', 'ros_domain_id']

# The real robots' command bus; the tick-lag check runs only there, since the sims stamp
# /lowstate's tick in sim steps, not in ms of the robot's clock
REAL_ROBOT_DOMAIN_ID = 0
# /lowstate older than this (s) is too old to seed the first command from
STATE_MAX_AGE = 0.1
# Max per-joint gap [rad] between the last commanded and the measured arm position for a new
# move to start from the command. A larger gap means the arm is blocked or was pushed, so the
# move restarts from the measured position and drops the held torque bias, releasing the
# force instead of continuing it.
COMMAND_SEED_MAX_GAP = 0.1
# Time constant [s] over which the steady-state torque bias fades once a new move starts
# commanding positions. Carrying it over keeps the arm from sagging at the start of a move;
# fading it hands the load back to gravity compensation as the arm leaves the pose the bias
# was integrated for.
TAU_BIAS_FADE_SEC = 1.0
# speed of the start-up move of the held joints to their targets, rad/s, at most
# controller.dq_lim
HOLD_SPEED = 0.3
# a start-up posture is reached when the IK's command is this close to it on every joint, rad
STARTUP_THRESHOLD = 1e-3


def ros_domain_id(node) -> int:
    '''The ROS domain node is on'''
    return node.context.get_domain_id()


class UpperController:
    def __init__(self, node, profile: RobotProfile, config: dict,
                 robot_model: RobotModel = None, init: bool = True):
        '''
        node: the ROS node the controller's subscriptions, publisher and logger live on; an
            executor must spin it, since the start-up waits on its /lowstate subscription
        profile, config: the robot's profile and controller config
            (golem_upper_ik.config.load_controller_config)
        robot_model: the profile's RobotModel at controller.imu_frame, default built here;
            its state source is the controller's /lowstate subscriber until shutdown()
        init: run the start-up moves
        '''
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

        self.dt = 1.0 / float(config['frequency']['ctrl_hz'])
        self.v_lim = controller['v_lim']
        self.w_lim = controller['w_lim']
        self.dq_lim = controller['dq_lim']
        self.d_min = controller['d_min']
        # multiplier on commanded arm speed: 1.0 is full speed, a goal's slow mode lowers it
        self.speed_scale = 1.0
        # steady-state integral torque bias [N*m, per body joint]: integrated while holding,
        # faded by _apply_joint_position while moving
        self._tau_bias = np.zeros(robot_model.num_motor)
        self._tau_bias_decay = float(np.exp(-self.dt / TAU_BIAS_FADE_SEC))
        # controlling on stale state winds up the hold's integral term and seeds moves from
        # where the arm was, so motion stops when the state is staler than this
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
        self.state_subscriber = StateSubscriber(node, profile, topic=self.state_topic)
        robot_model.state_subscriber = self.state_subscriber
        try:
            # nothing is commanded before a fresh /lowstate: no command starts from the model's
            # all-zero rest pose
            self._wait_for_state()
        except BaseException:
            robot_model.shutdown()
            raise
        try:
            robot_model.update_kinematics()
            # the first command is where the robot is, gravity compensated
            self.low_cmd_handler = LowCmdHandler(node, profile, config, robot_model)
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
            # one zero-gain command, then silence
            self.shutdown()
            raise

    def _planner_config(self):
        '''The planner section's settings over PlannerConfig's defaults; dt the control period'''
        settings = {key: value for key, value in self.config['planner'].items()
                    if key not in ('enabled', 'frame_names')}
        settings.setdefault('dt', self.dt)
        return PlannerConfig(**settings)

    def _wait_for_state(self):
        '''Block until a fresh /lowstate with finite joint positions arrives'''
        subscriber = self.state_subscriber
        deadline = time.monotonic() + self.state_timeout
        while time.monotonic() < deadline:
            age = subscriber.age_s()
            if age is not None and age < STATE_MAX_AGE and \
                    np.all(np.isfinite(subscriber.state['q'])):
                return
            time.sleep(0.01)
        raise RuntimeError(
            f'No fresh {self.state_topic} with finite joint positions within '
            f'{self.state_timeout:.1f} s on ROS domain {ros_domain_id(self.node)}; not '
            f'enabling any motor. Is the robot (or the sim) up and publishing '
            f'{self.state_topic} on this domain, and is DDS on the interface wired to it '
            '(on the G1, G1_DDS_INTERFACE)?')

    # the start-up

    def _startup_routine(self):
        '''
        Hold the measured pose until golem_safety relays the commands (within
        startup_timeout), then move the arms through the start-up postures and the held
        joints to their targets (within startup_move_timeout of that)
        '''
        self._wait_until_following(time.monotonic() + self.startup_timeout)
        # the moves get their own budget from the gate opening on: a late engagement must not
        # cut them short once golem_safety has taken over
        deadline = time.monotonic() + self.startup_move_timeout
        for name in self.profile.startup_sequence:
            self._move_to_reduced_configuration(
                self.profile.named_configs[name], deadline, desc=f'Moving {name}')
        self._move_held_joints_to_targets(deadline)

    def _wait_until_following(self, deadline):
        '''Re-send the gated hold, near the measured pose, until the follow gate opens'''
        handler = self.low_cmd_handler
        if handler.following:
            return
        topic = handler.safety_output_topic
        self._logger.info(f'Waiting for golem_safety to relay the commands on {topic}...')
        while not handler.following:
            if time.monotonic() > deadline:
                raise RuntimeError(self._not_relayed_error(topic))
            start = time.monotonic()
            # the IK and the hold sit on the gated command; fresh gravity torque
            self._apply_joint_position(self.ik_solver.q)
            self.update_robot_model()
            time.sleep(max(0.0, self.dt - (time.monotonic() - start)))

    def _not_relayed_error(self, topic):
        at = ' at weight 1' if topic == SAFETY_OUTPUT_TOPICS['arm_sdk_mode'] else ''
        return (
            'golem_safety did not relay the commands within controller.startup_timeout '
            f'({self.startup_timeout:.0f} s): no {topic} sample{at} carried them. golem_safety '
            f'relays only once it sees {self.state_topic} report mode_machine '
            f'{self.profile.mode_machine} (the {self.profile.robot} profile\'s) and, when started '
            'with use_estop:=true, a first sample on its estop topics; it must run the safety '
            'config this controller config pairs with (writing '
            f'{topic}), not latched in an estop from an earlier run (restart it), and under '
            'an engage gate (the G1\'s configs) every input of its mode must command: in split '
            'mode the walking node too. Stopping the controller')

    def _startup_timeout_error(self, desc):
        '''The error for a start-up move not done in time'''
        q_cmd = self.low_cmd_handler.q_cmd[self.upper_ids]
        q = self.robot_model.state['q'][self.upper_ids]
        worst = int(np.argmax(np.abs(q_cmd - q)))
        return RuntimeError(
            f'{desc}: not done within controller.startup_move_timeout '
            f'({self.startup_move_timeout:.0f} s) of the follow gate opening '
            f'({self.profile.body_joints[self.upper_ids[worst]]} commanded '
            f'{q_cmd[worst]:.3f} rad, measured {q[worst]:.3f} rad). Stopping the controller')

    def _move_to_reduced_configuration(self, q_reduced, deadline, desc):
        '''Step the IK until the command reaches q_reduced; raise at the deadline'''
        q_reduced = np.asarray(q_reduced, dtype=np.float64)
        self._logger.info(f'{desc}...')
        while True:
            start = time.monotonic()
            self.goto_reduced_configuration(q_reduced)
            if np.max(np.abs(self.ik_solver.q_reduced - q_reduced)) < STARTUP_THRESHOLD:
                return
            if time.monotonic() > deadline:
                raise self._startup_timeout_error(desc)
            time.sleep(max(0.0, self.dt - (time.monotonic() - start)))

    def _move_held_joints_to_targets(self, deadline):
        '''Ease the held joints (the torso, the waist) to exactly their hold targets'''
        held = ', '.join(self.profile.held_joints)
        desc = f'Moving {held} to the hold targets'
        handler = self.low_cmd_handler
        step = max(min(HOLD_SPEED, self.dq_lim) * self.dt, 1e-6)
        targets = self.hold_targets
        self._logger.info(f'{desc}...')
        while np.any(handler.hold != targets):
            if time.monotonic() > deadline:
                raise self._startup_timeout_error(desc)
            start = time.monotonic()
            remaining = targets - handler.hold
            # the last step lands on the targets themselves
            handler.set_hold(np.where(np.abs(remaining) <= step, targets,
                                      handler.hold + np.clip(remaining, -step, step)))
            self._apply_joint_position(self.ik_solver.q)
            self.update_robot_model()
            time.sleep(max(0.0, self.dt - (time.monotonic() - start)))

    # end effectors

    def ee_transformations(self):
        '''
        The left and right end effectors' 4x4 placements in the pelvis frame, from the
        reduced model at one measured state. Computed on data of their own, so a call from
        another thread leaves the IK's data alone
        '''
        q_reduced = self.robot_model.state_reduced['q']
        return tuple(self.robot_model.get_frame_transformation_reduced(name, q_reduced)
                     for name in (self.left_ee_name, self.right_ee_name))

    # goals

    def check_state_fresh(self):
        '''Raise StaleStateError if /lowstate is too stale to control on'''
        if self.max_state_lag_ms <= 0.0:
            return
        staleness_ms = self.robot_model.state_staleness_ms
        if staleness_ms > self.max_state_lag_ms:
            raise StaleStateError(
                f'{self.state_topic} is {staleness_ms:.0f} ms stale '
                f'(> {self.max_state_lag_ms:.0f} ms); holding the last command')

    def update_ik_solver(self):
        '''Ready the IK for a new goal'''
        self.check_state_fresh()
        # re-lock the frozen joints (the torso or waist, the legs) at their measured angles
        # so reduced-model FK agrees with the robot's actual kinematics (and the TF tree);
        # frozen at zero, a torso holding e.g. -0.04 rad offsets every arm frame by a
        # constant ~15 mm that the IK can never converge away
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
        '''
        (q, from_command): the configuration (per body joint) a new move starts from. The IK
        joints take the last commanded position when it is within COMMAND_SEED_MAX_GAP of the
        measured position on every one of them; otherwise the whole seed is the measured state
        '''
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
        '''Why this controller cannot plan, or None'''
        if not self.planner.available:
            return self.planner.unavailable_reason
        if not self.config['planner']['enabled']:
            return (f"Planning is turned off: the controller config {self.config['source']} "
                    'sets planner.enabled: false; send the goal with plan=false')
        return None

    def _check_planning(self):
        refusal = self.planning_refusal
        if refusal is not None:
            raise PlannerUnavailableError(refusal)

    def _as_reduced_path(self, path):
        '''path as an (N, IK joints) array of finite waypoints'''
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
        if not result.success:
            raise RuntimeError(f'Planning failed: {result.reason}')
        return result.path

    def _planner_speed_overrides(self):
        '''
        Per-plan speed settings implied by the current speed_scale.

        A planned path is slowed by planning it at a lower planner-frame speed: the planner
        then emits proportionally more waypoints and execute_path still clocks them out at
        dt. Stretching the step period instead keeps the original waypoint spacing and turns
        a smooth ramp into a staircase. (None, None) at full speed, so the planner process
        keeps the config it was started with.
        '''
        scale = float(self.speed_scale)
        if scale == 1.0:
            return None, None
        config = self.planner.planner_config
        # the waypoint cap bounds a path's duration, so it has to grow with the step count or
        # a long slow path clamps back to near full speed
        return (config.moving_speed * scale,
                int(np.ceil(config.max_interpolation_steps / scale)))

    def plan_to_configuration(self, q_reduced, start=None, active_mask=None):
        '''
        Plan from the move's start configuration (default: the command seed) to a reduced
        joint target. PlannerUnavailableError when the controller cannot plan
        '''
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
        '''
        Solve the current IK tasks, then plan to the IK's joint solution.
        PlannerUnavailableError, before any IK, when the controller cannot plan
        '''
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
        '''Reduced joint mask supporting all active frame tasks'''
        mask = np.zeros(self.robot_model.model_body_reduced.nq, dtype=bool)
        for task in self.ik_solver.frame_tasks.values():
            mask |= self._reduced_support_mask(task.frame)
        return mask

    def _reduced_support_mask(self, frame_name):
        '''Reduced joint mask supporting a frame in the kinematic chain'''
        model = self.robot_model.model_body_reduced
        joint_id = model.frames[model.getFrameId(frame_name)].parentJoint
        # walk up the tree to the root, marking supporting joints
        mask = np.zeros(model.nq, dtype=bool)
        while joint_id > 0:
            idx_q = model.joints[joint_id].idx_q
            mask[idx_q:idx_q + model.joints[joint_id].nq] = True
            joint_id = model.parents[joint_id]
        return mask

    def execute_path(self, path, validate=True):
        '''Execute reduced joint waypoints as direct position commands, one per control period'''
        if not self.low_cmd_handler.following:
            # absolute waypoints would run ahead of a robot that is not following
            raise RuntimeError("Not executing a path before golem_safety relays the "
                               "controller's commands (follow gate shut)")
        path = self._as_reduced_path(path)
        # The path is time-parameterized at dt by the planner (slow moves are planned at a
        # lower moving_speed and come back with more waypoints), so it always plays back at
        # the control period: each step sleeps the period less the work already done, or
        # the checks and FK would stretch a path past its planned duration
        for q_reduced in path:
            start = time.monotonic()
            self.check_state_fresh()
            if validate:
                # keep execution guarded by the same reduced validity checks
                if not self.robot_model.check_within_limits_reduced(q_reduced):
                    raise ValueError('Path waypoint is outside joint limits')
                if not self.robot_model.check_collision_free_reduced(q_reduced):
                    raise ValueError('Path waypoint is in self-collision')
            q = np.array(self.robot_model.state['q'], dtype=np.float64)
            q[self.robot_model.reduced_mask] = q_reduced
            self._apply_joint_position(q)
            self.update_robot_model()
            time.sleep(max(0.0, self.dt - (time.monotonic() - start)))

    def update_robot_model(self):
        self.robot_model.update_kinematics()

    @property
    def reduced_configuration_error(self):
        return self.ik_solver.compute_error_reduced(
            self.ik_solver.config_task, self.robot_model.state_reduced['q'])

    def set_reduced_configuration_target(self, q_reduced):
        '''Set the reduced joint target used for configuration feedback'''
        self.ik_solver.config_task.set_target(q_reduced)

    def goto_reduced_configuration(self, q_reduced):
        '''One IK step toward a reduced joint configuration; the joint velocity commanded'''
        self.check_state_fresh()
        vel = self.ik_solver.goto_reduced_configuration(q_reduced)
        return self._apply_velocity_command(self._limit_joint_vel(vel))

    def control_step_reduced(self):
        '''One IK step on every frame task, on the reduced model; the joint velocity commanded'''
        self.check_state_fresh()
        vel = self.ik_solver.ik_step_reduced()
        return self._apply_velocity_command(self._limit_joint_vel(vel))

    # the steady-state hold

    def init_steady_state(self):
        '''
        Hold the last command with integral control. The integration resumes from the current
        torque bias, so the hand-off from a move to a hold is continuous
        '''
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
        '''
        (joint name, q, limit) for each upper-body joint within threshold of a position clip
        limit that the integral term is pushing further into
        '''
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
        '''
        One steady-state integral-control tick; whether the hold has converged. A hold pinned
        against a joint position limit cannot converge: the integral term is frozen,
        steady_state_limited_joints names the joints, and this returns False
        '''
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

    # commands

    def _limit_joint_vel(self, vel):
        '''
        Scale each arm's joint velocity so its joints stay under dq_lim and its end effector
        under v_lim and w_lim, then by speed_scale; the other joints pass through
        '''
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
        '''Command q with dq 0 and the gravity torque plus the bias, which fades by one step'''
        tau = self.robot_model.dynamics.get_gravity_compensation(self.robot_model.state['q'])
        tau = tau + self._tau_bias
        self._tau_bias *= self._tau_bias_decay
        self._command(q, np.zeros(self.robot_model.num_motor), tau)

    def _apply_velocity_command(self, vel):
        self.ik_solver.integrate(vel)
        self._apply_joint_position(self.ik_solver.q)
        self.update_robot_model()
        return vel

    def _command(self, q, dq, tau):
        self.low_cmd_handler.set_joint_commands(q, dq, tau)
        self._sync_to_command()

    def _sync_to_command(self):
        '''
        Until golem_safety relays the commands, restart the IK and the steady-state target
        from the command the follow gate let through (the hold restarts in the handler), so
        none of them runs ahead of the robot; never after that
        '''
        if self.low_cmd_handler.following:
            return
        q_cmd = self.low_cmd_handler.q_cmd.astype(np.float64)
        self.ik_solver.configuration.update(q_cmd)
        self.ik_solver.configuration_reduced.update(q_cmd[self.robot_model.reduced_mask])
        if self._steady is not None:
            self._steady['q'] = q_cmd

    def shutdown(self):
        '''
        Stop the planner, send the upper slots one zero-gain command and stop publishing,
        then detach the state source. Publishing stops first: a detached source would leave
        the follow gate nothing to hold the commands near. Idempotent
        '''
        if self.planner is not None:
            self.planner.shutdown()
        if self.low_cmd_handler is not None:
            self.low_cmd_handler.shutdown()
        self.robot_model.shutdown()
