import time
import numpy as np
import pinocchio as pin
from tqdm import tqdm

from g1_control.core.ik_solver import IKSolver
from g1_control.core.robot_model import RobotModel
from g1_control.core.low_cmd_handler import LowCmdHandler
from g1_control.core.planner import PlannerConfig, PlannerClient
from g1_control.utility.named_config import NAMED_CONFIGS, STARTUP_SEQUENCE
from g1_control.utility.controller_config import (
    ARM_SDK_TOPIC,
    _channel_factory_settings,
    load_controller_config,
)
from g1_control.utility.joint_definition import (
    ENABLED_JOINTS,
    BODY_JOINTS,
    LEFT_ARM_INDEX,
    LEFT_EE_FRAME,
    RIGHT_ARM_INDEX,
    RIGHT_EE_FRAME,
    UPPER_BODY_INDEX,
    WAIST_INDEX,
)

PLANNER_UNAVAILABLE = (
    'Planning is unavailable: the G1 has no self-collision model yet '
    '(set planner.enabled with a collision URDF/SRDF); send the goal with plan=false'
)
# rt/lowstate older than this (s) is too old to seed the first command from
STATE_MAX_AGE = 0.1
# speed of the start-up waist move (rad/s), at most controller.dq_lim
WAIST_SPEED = 0.3

class UpperController:
    def __init__(self,
                 urdf_path: str,
                 init: bool=True,
                 visualize=False,
                 config=None,
                 collision_urdf_path: str=None,
                 collision_srdf_path: str=None):
        self.config = config if config is not None else load_controller_config()
        controller_cfg = self.config.get('controller', {})

        # initialize robot model
        self.robot_model = RobotModel(
            urdf_path, imu_frame=controller_cfg.get('imu_frame', 'imu_in_pelvis')
        )
        self.dt = 1.0 / float(self.config.get('frequency', {}).get('ctrl_hz', 50.0))
        self.v_lim = float(controller_cfg.get('v_lim', 1.0))
        self.w_lim = float(controller_cfg.get('w_lim', 2.0))
        self.dq_lim = float(controller_cfg.get('dq_lim', 1.0))
        self.d_min = float(controller_cfg.get('d_min', 0.02))
        self.waist_target = np.asarray(
            controller_cfg.get('waist_target', [0.0] * len(WAIST_INDEX)), dtype=np.float64
        )
        self.state_timeout = float(controller_cfg.get('state_timeout', 5.0))
        self.startup_timeout = float(controller_cfg.get('startup_timeout', 30.0))
        self.startup_move_timeout = float(controller_cfg.get('startup_move_timeout', 30.0))
        self.follow_tolerance = float(controller_cfg.get('follow_tolerance', 0.05))
        self.visualize = visualize

        # initialize subscriber in robot model; nothing is commanded before a
        # fresh rt/lowstate arrives, so no target starts from the all-zero pose
        low_state_topic = self.config.get('topics', {}).get('low_state', 'rt/lowstate')
        self.robot_model.init_subscriber(low_state_topic=low_state_topic)
        try:
            self._wait_for_state(low_state_topic)
        except BaseException:
            self.robot_model.shutdown()
            raise
        self.robot_model.update_kinematics()

        # init reduced model; the self-collision model is optional (the G1 has none yet)
        self.robot_model.init_reduced_model(ENABLED_JOINTS)
        has_collision_model = collision_urdf_path is not None and collision_srdf_path is not None
        if has_collision_model:
            self.robot_model.init_collision_model(collision_urdf_path, collision_srdf_path)
        self.enabled_ids = [BODY_JOINTS.index(joint) for joint in ENABLED_JOINTS]
        self.upper_ids = UPPER_BODY_INDEX

        # intialize low cmd publisher
        self.low_cmd_handler = LowCmdHandler(self.robot_model,
                                             config=self.config)

        # seed every slot from the measured pose, gravity compensated: the
        # first command is where the robot is
        q_measured = np.asarray(self.robot_model.state['q'], dtype=np.float64)
        self.low_cmd_handler.set_joint_commands(
            q=q_measured,
            dq=np.zeros(self.robot_model.model_body.nv),
            tau=self.robot_model.dynamics.get_gravity_compensation(q_measured),
        )
        q_seed = self.low_cmd_handler.q_cmd

        # enable upper body motors
        self.low_cmd_handler.enable_motors(self.enabled_ids, q_seed[self.enabled_ids])

        # enable the waist motors at their current position; they are held, not
        # IK-driven, and every command carries the hold angles
        self._waist_hold = np.asarray(q_seed[WAIST_INDEX], dtype=np.float64)
        self.low_cmd_handler.enable_motors(WAIST_INDEX, self._waist_hold)

        # until g1_safety_layer is seen relaying the commands (read on
        # topics.safety_out), no waist or arm command leads its measured joint
        # by more than this
        self.low_cmd_handler.arm_follow_gate(
            self.follow_tolerance, self.config.get('topics', {}).get('safety_out', ARM_SDK_TOPIC)
        )

        # start publisher
        self.low_cmd_handler.start()

        # initialize IK solver
        self.ik_solver = IKSolver(
            robot_model=self.robot_model,
            dt=self.dt,
            d_min=self.d_min
        )
        # OMPL planning needs the self-collision model
        self.planner = None
        if has_collision_model and bool(self.config.get('planner', {}).get('enabled', False)):
            self.planner = PlannerClient(
                urdf_path=urdf_path,
                collision_urdf_path=collision_urdf_path,
                collision_srdf_path=collision_srdf_path,
                planner_config=self._load_planner_config(),
            )

        if self.visualize:
            self.robot_model.init_visualizer()
            self.robot_model.config_visualizer(show_com=True, show_zmp=True)

        # default end effector frame names for velocity limiting (Dex3-1 palms)
        self.left_ee_name = LEFT_EE_FRAME
        self.right_ee_name = RIGHT_EE_FRAME

        if init:
            try:
                self._startup_routine(self.waist_target)
            except BaseException:
                # stop publishing: g1_safety_layer then hands back or estops
                self.shutdown()
                raise

    def _wait_for_state(self, low_state_topic):
        '''Block until a fresh rt/lowstate with finite joint positions arrives'''
        deadline = time.monotonic() + self.state_timeout
        while time.monotonic() < deadline:
            state = self.robot_model.state
            age = time.time() - state['time_stamp']
            fresh = state['time_stamp'] > 0.0 and age < STATE_MAX_AGE
            if fresh and np.all(np.isfinite(state['q'])):
                return
            time.sleep(0.01)
        domain_id, interface = _channel_factory_settings(self.config)
        raise RuntimeError(
            f'No fresh {low_state_topic} with finite joint positions within '
            f'{self.state_timeout:.1f} s on DDS domain {domain_id} '
            f'(interface {interface or "auto"}); not enabling any motor. Is the robot '
            '(or the sim) up, and does G1_DDS_INTERFACE name the port wired to it?'
        )

    def _startup_routine(self, waist_target):
        '''
        Hold the measured pose until g1_safety_layer relays the commands (within
        startup_timeout), then move the arms through STARTUP_SEQUENCE and the
        waist to its hold angles (within startup_move_timeout of that)
        '''
        self._wait_until_following(time.monotonic() + self.startup_timeout)
        # the moves get their own budget from the gate opening on: a late
        # engagement must not cut them short once the safety layer has taken over
        deadline = time.monotonic() + self.startup_move_timeout
        for config_name in STARTUP_SEQUENCE:
            self._move_to_reduced_configuration(
                NAMED_CONFIGS[config_name],
                deadline,
                desc=f'Moving {config_name}',
            )
        self._move_waist_to_target(waist_target, deadline)

    def _wait_until_following(self, deadline):
        '''Re-send the gated hold, near the measured pose, until the follow gate opens'''
        if self.low_cmd_handler.following:
            return
        topic = self.low_cmd_handler.safety_output_topic
        print(f'Waiting for g1_safety_layer to relay the commands on {topic}...', flush=True)
        while not self.low_cmd_handler.following:
            if time.monotonic() > deadline:
                at = ' at weight 1' if topic == ARM_SDK_TOPIC else ''
                raise RuntimeError(
                    'g1_safety_layer did not engage within controller.startup_timeout '
                    f'({self.startup_timeout:.0f} s): no {topic} sample{at} carried the '
                    'commands. Is it running with the safety config this controller config '
                    'pairs with (topics.low_cmd_out the same as topics.safety_out), not '
                    'latched in an estop from an earlier run (restart it), and in split '
                    'mode is the walking node commanding too? Stopping the controller'
                )
            start_time = time.time()
            # the IK and the waist hold sit on the gated command; fresh gravity torque
            self._apply_joint_position(self.ik_solver.q)
            self.update_robot_model()
            time.sleep(max(0.0, self.dt - (time.time() - start_time)))

    def _startup_timeout_error(self, desc):
        '''The error for a start-up move not done in time'''
        q_cmd = self.low_cmd_handler.q_cmd[UPPER_BODY_INDEX]
        q = self.robot_model.state['q'][UPPER_BODY_INDEX]
        worst = int(np.argmax(np.abs(q_cmd - q)))
        return RuntimeError(
            f'{desc}: not done within controller.startup_move_timeout '
            f'({self.startup_move_timeout:.0f} s) of the follow gate opening '
            f'({BODY_JOINTS[UPPER_BODY_INDEX[worst]]} commanded {q_cmd[worst]:.3f} rad, '
            f'measured {q[worst]:.3f} rad). Stopping the controller'
        )

    def _move_to_reduced_configuration(
        self, q_reduced, deadline,
        threshold=1e-3,
        desc='Moving configuration',
    ):
        '''Step the IK until the command reaches q_reduced; raise at the deadline'''
        q_reduced = np.asarray(q_reduced, dtype=np.float64)
        print(f'{desc}...', flush=True)
        while True:
            start_time = time.time()
            self.goto_reduced_configuration(q_reduced)
            if np.max(np.abs(self.ik_solver.q_reduced - q_reduced)) < threshold:
                return
            if time.monotonic() > deadline:
                raise self._startup_timeout_error(desc)
            time.sleep(max(0.0, self.dt - (time.time() - start_time)))

    def _load_planner_config(self):
        '''Load OMPL planner config from controller config'''
        # merge optional yaml planner settings with dataclass defaults
        cfg = self.config.get('planner', {})
        defaults = PlannerConfig()
        return PlannerConfig(
            planner=cfg.get('planner', defaults.planner),
            timeout=float(cfg.get('timeout', defaults.timeout)),
            range=float(cfg.get('range', defaults.range)),
            try_direct=bool(cfg.get('try_direct', defaults.try_direct)),
            simplify=bool(cfg.get('simplify', defaults.simplify)),
            simplify_time=float(cfg.get('simplify_time', defaults.simplify_time)),
            shortcut=bool(cfg.get('shortcut', defaults.shortcut)),
            moving_speed=float(cfg.get('moving_speed', defaults.moving_speed)),
            dt=float(cfg.get('dt', self.dt)),
            min_interpolation_steps=int(
                cfg.get(
                    'min_interpolation_steps',
                    defaults.min_interpolation_steps,
                )
            ),
            max_interpolation_steps=int(
                cfg.get(
                    'max_interpolation_steps',
                    defaults.max_interpolation_steps,
                )
            ),
            validity_resolution=float(
                cfg.get('validity_resolution', defaults.validity_resolution)
            ),
            constraint_check_steps=int(
                cfg.get('constraint_check_steps', defaults.constraint_check_steps)
            ),
            frame_names=tuple(cfg.get('frame_names', defaults.frame_names)),
            frame_z_min=cfg.get('frame_z_min', defaults.frame_z_min),
            frame_z_min_margin=float(
                cfg.get('frame_z_min_margin', defaults.frame_z_min_margin)
            ),
            frame_z_corridor_margin=float(
                cfg.get(
                    'frame_z_corridor_margin',
                    defaults.frame_z_corridor_margin,
                )
            ),
        )

    def _move_waist_to_target(self, waist_target, deadline):
        '''Ease the three waist joints (yaw, roll, pitch) to exactly their hold angles'''
        desc = 'Moving waist to target'
        step_size = max(min(WAIST_SPEED, self.dq_lim) * self.dt, 1e-6)
        waist_target = np.asarray(waist_target, dtype=np.float64)
        print(f'{desc}...', flush=True)
        while np.any(self._waist_hold != waist_target):
            if time.monotonic() > deadline:
                raise self._startup_timeout_error(desc)
            start_time = time.time()
            remaining = waist_target - self._waist_hold
            # the last step lands on the target itself
            self._waist_hold = np.where(
                np.abs(remaining) <= step_size,
                waist_target,
                self._waist_hold + np.clip(remaining, -step_size, step_size),
            )
            self._apply_joint_position(self.ik_solver.q)
            self.update_robot_model()
            time.sleep(max(0.0, self.dt - (time.time() - start_time)))

    '''
    joint position for left and right arms
    '''
    @property
    def left_arm_q(self):
        return np.copy(self.robot_model.state['q'][LEFT_ARM_INDEX])

    @property
    def right_arm_q(self):
        return np.copy(self.robot_model.state['q'][RIGHT_ARM_INDEX])

    '''
    left end effector properties
    left_ee_transformation: transformation matrix of the left end effector
    left_ee_position: position of the left end effector
    left_ee_rotation: rotation matrix of the left end effector
    left_ee_rpy: roll, pitch, yaw of the left end effector
    left_ee_pose: pose of the left end effector (x, y, z, roll, pitch, yaw)
    '''
    @property
    def left_ee_transformation(self):
        return self.robot_model.get_frame_transformation_reduced(self.left_ee_name)

    @property
    def left_ee_position(self):
        return self.robot_model.get_frame_position_reduced(self.left_ee_name)

    @property
    def left_ee_rotation(self):
        return self.robot_model.get_frame_rotation_reduced(self.left_ee_name)

    @property
    def left_ee_rpy(self):
        return pin.rpy.matrixToRpy(self.left_ee_rotation)

    @property
    def left_ee_pose(self):
        return np.concatenate([self.left_ee_position, self.left_ee_rpy])

    '''
    right end effector properties
    right_ee_transformation: transformation matrix of the right end effector
    right_ee_position: position of the right end effector
    right_ee_rotation: rotation matrix of the right end effector
    right_ee_rpy: roll, pitch, yaw of the right end effector
    right_ee_pose: pose of the right end effector (x, y, z, roll, pitch, yaw)
    '''
    @property
    def right_ee_transformation(self):
        return self.robot_model.get_frame_transformation_reduced(self.right_ee_name)

    @property
    def right_ee_position(self):
        return self.robot_model.get_frame_position_reduced(self.right_ee_name)

    @property
    def right_ee_rotation(self):
        return self.robot_model.get_frame_rotation_reduced(self.right_ee_name)

    @property
    def right_ee_rpy(self):
        return pin.rpy.matrixToRpy(self.right_ee_rotation)

    @property
    def right_ee_pose(self):
        return np.concatenate([self.right_ee_position, self.right_ee_rpy])

    def update_ik_solver(self):
        # update IK solver with the latest robot configuration
        self.ik_solver.update_configurations()

    def _as_reduced_path(self, path):
        # normalize path input and enforce reduced-model waypoint shape
        path = np.asarray(path, dtype=float)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        nq = self.robot_model.model_body_reduced.nq
        if path.ndim != 2 or path.shape[1] != nq:
            raise ValueError(f'Path must have shape (N, {nq})')
        if not np.all(np.isfinite(path)):
            raise ValueError('Path must contain only finite values')
        return path

    def _path_or_raise(self, result):
        if not result.success:
            raise RuntimeError(f'Planning failed: {result.reason}')
        return result.path

    def plan_to_configuration(self, q_reduced, start=None, active_mask=None):
        '''Plan from current reduced state to a reduced joint target'''
        if self.planner is None:
            raise RuntimeError(PLANNER_UNAVAILABLE)
        if start is None:
            start = self.robot_model.state_reduced['q']
        q_reduced = np.asarray(q_reduced, dtype=float)
        result = self.planner.plan_configuration(
            current_q=self.robot_model.state['q'],
            start=start,
            goal=q_reduced,
            active_mask=active_mask,
        )
        return self._path_or_raise(result)

    def plan_to_ik_target(self, start=None, ik_timeout=1.0, ik_alpha=0.1):
        '''Solve current IK tasks, then plan to the IK joint solution'''
        if self.planner is None:
            raise RuntimeError(PLANNER_UNAVAILABLE)
        # restrict planning to joints supporting the active tasks so that
        # untasked limbs (e.g. the other arm) never move during the plan
        active_mask = self._active_task_mask()
        self.update_ik_solver()
        ik_result = self.ik_solver.solve_ik_reduced(
            alpha=ik_alpha,
            timeout=ik_timeout,
        )
        if not ik_result.success:
            raise RuntimeError('IK failed to resolve current targets')

        q_goal = ik_result.q[self.robot_model.reduced_mask]
        start_q = start if start is not None else self.robot_model.state_reduced['q']
        return self.plan_to_configuration(
            q_goal,
            start=start_q,
            active_mask=active_mask,
        )

    def _active_task_mask(self):
        '''Reduced joint mask supporting all active frame tasks'''
        mask = np.zeros(self.robot_model.model_body_reduced.nq, dtype=bool)
        for task in self.ik_solver.frame_tasks.values():
            mask |= self._reduced_support_mask(task.frame)
        return mask

    def _reduced_support_mask(self, frame_name):
        '''Reduced joint mask supporting a frame in the kinematic chain'''
        model = self.robot_model.model_body_reduced
        frame_id = model.getFrameId(frame_name)
        joint_id = model.frames[frame_id].parentJoint

        # walk up the tree to the root, marking supporting joints
        mask = np.zeros(model.nq, dtype=bool)
        while joint_id > 0:
            idx_q = model.joints[joint_id].idx_q
            nq = model.joints[joint_id].nq
            mask[idx_q:idx_q + nq] = True
            joint_id = model.parents[joint_id]
        return mask

    def visualize_path(self, path):
        '''Draw a reduced joint-space path with Meshcat'''
        if self.robot_model.visualizer is None:
            self.robot_model.init_visualizer()
        self.robot_model.visualizer.visualize_reduced_path(path)

    def play_path(self, path, delay=0.05):
        '''Animate a reduced joint-space path with Meshcat'''
        if self.robot_model.visualizer is None:
            self.robot_model.init_visualizer()
        self.robot_model.visualizer.play_reduced_path(path, delay=delay)

    def execute_path(self, path, validate=True):
        '''Execute reduced joint waypoints as direct position commands'''
        if not self.low_cmd_handler.following:
            # absolute waypoints would run ahead of a robot that is not following
            raise RuntimeError(
                'Not executing a path before g1_safety_layer relays the controller\'s '
                'commands (follow gate shut)'
            )
        path = self._as_reduced_path(path)

        # apply each reduced waypoint as a direct full motor position command
        for q_reduced in tqdm(path, desc='Executing path', leave=False):
            if validate:
                # keep execution guarded by the same reduced validity checks
                if not self.robot_model.check_within_limits_reduced(q_reduced):
                    raise ValueError('Path waypoint is outside joint limits')
                if not self.robot_model.check_collision_free_reduced(q_reduced):
                    raise ValueError('Path waypoint is in self-collision')

            q = np.copy(self.robot_model.state['q'])
            q[self.robot_model.reduced_mask] = q_reduced
            self._apply_joint_position(q)
            self.update_robot_model()
            time.sleep(self.dt)

    def update_robot_model(self):
        # update kinematics
        self.robot_model.update_kinematics()
        # update visualizer if needed
        if self.visualize:
            self.ik_solver.update_visualizer()
            self.robot_model.update_visualizer()

    @property
    def configuration_error(self):
        return self.ik_solver.compute_error(
            self.ik_solver.config_task,
            self.robot_model.state['q']
        )

    def goto_configuration(self, q):
        # solve IK and apply control
        vel = self.ik_solver.goto_configuration(q)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver and command the joint position
        self.ik_solver.integrate(vel)
        self._apply_joint_position(self.ik_solver.q)
        self.update_robot_model()
        return vel

    def sim_goto_configuration(self, q):
        # solve IK and apply control
        vel = self.ik_solver.goto_configuration(q)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver
        self.ik_solver.integrate(vel)
        # force robot model to use local variable tracking states
        self.robot_model.state_subscriber = None
        self.robot_model._q = self.robot_model.state['q'] + vel * self.dt
        self.update_robot_model()
        return vel

    @property
    def reduced_configuration_error(self):
        return self.ik_solver.compute_error_reduced(
            self.ik_solver.config_task,
            self.robot_model.state_reduced['q']
        )

    def set_reduced_configuration_target(self, q_reduced):
        '''Set the reduced joint target used for configuration feedback'''
        self.ik_solver.config_task.set_target(q_reduced)

    def goto_reduced_configuration(self, q_reduced):
        # solve IK and apply control
        vel = self.ik_solver.goto_reduced_configuration(q_reduced)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver and command the joint position
        self.ik_solver.integrate(vel)
        self._apply_joint_position(self.ik_solver.q)
        self.update_robot_model()
        return vel

    def sim_goto_reduced_configuration(self, q_reduced):
        # solve IK and apply control
        vel = self.ik_solver.goto_reduced_configuration(q_reduced)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver
        self.ik_solver.integrate(vel)
        # force robot model to use local variable tracking states
        self.robot_model.state_subscriber = None
        self.robot_model._q = self.ik_solver.q
        self.update_robot_model()
        return vel

    def lock_configuration(self, q):
        self._apply_joint_position(q)
        self.update_robot_model()

    def control_step(self, com=False):
        '''Solve IK for all tasks'''
        # solve IK and apply the control
        vel = self.ik_solver.ik_step(com=com)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver and command the joint position
        self.ik_solver.integrate(vel)
        self._apply_joint_position(self.ik_solver.q)
        self.update_robot_model()
        return vel

    def control_step_reduced(self, com=False):
        '''Solve IK for all tasks with the reduced model'''
        # solve IK and apply the control
        vel = self.ik_solver.ik_step_reduced(com=com)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver and command the joint position
        self.ik_solver.integrate(vel)
        self._apply_joint_position(self.ik_solver.q)
        self.update_robot_model()
        return vel

    def sim_step(self, com=False):
        # solve IK and apply the control
        vel = self.ik_solver.ik_step(com=com)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver
        self.ik_solver.integrate(vel)
        # force robot model to use local variable tracking states
        self.robot_model.state_subscriber = None
        self.robot_model._q = self.ik_solver.q
        self.update_robot_model()
        return vel

    def sim_step_reduced(self, com=False):
        # solve IK and apply the control
        vel = self.ik_solver.ik_step_reduced(com=com)
        vel = self._limit_joint_vel(vel)
        # integrate IK solver
        self.ik_solver.integrate(vel)
        # force robot model to use local variable tracking states
        self.robot_model.state_subscriber = None
        self.robot_model._q = self.ik_solver.q
        self.update_robot_model()
        return vel

    def init_steady_state(self):
        '''Initialize steady-state I control from the final IK command'''
        self._steady_q_cmd = np.copy(self.low_cmd_handler.q_cmd)
        self._steady_dq_cmd = np.zeros(self.robot_model.model_body.nv)
        self._steady_tau_bias = np.zeros(self.robot_model.model_body.nv)
        ki = self.config.get('gains', {}).get('ki')
        if ki is None:
            self._steady_ki = np.zeros(self.robot_model.model_body.nv)
        else:
            self._steady_ki = np.asarray(ki, dtype=np.float64)

        self._steady_tau_bias_limit = np.zeros_like(self.robot_model.model_body.nv)
        tau_clip_limits = self.config.get('limits', {}).get('tau_clip_limits')
        if tau_clip_limits is not None:
            self._steady_tau_bias_limit = np.asarray(tau_clip_limits, dtype=np.float64)

        self._steady_q_clip_limits = None
        q_clip_limits = self.config.get('limits', {}).get('q_clip_limits')
        if q_clip_limits is not None:
            self._steady_q_clip_limits = np.asarray(q_clip_limits, dtype=np.float64)

    def _steady_state_hits_position_clip(self, q_state, tau_bias_increment, threshold):
        if self._steady_q_clip_limits is None:
            return False

        q_limits = self._steady_q_clip_limits[self.upper_ids]
        q_state = q_state[self.upper_ids]
        tau_bias_increment = tau_bias_increment[self.upper_ids]

        near_upper = q_state >= (q_limits[:, 1] - threshold)
        near_lower = q_state <= (q_limits[:, 0] + threshold)
        push_upper = tau_bias_increment > 0.0
        push_lower = tau_bias_increment < 0.0

        return bool(np.any((near_upper & push_upper) | (near_lower & push_lower)))

    def steady_state_step(self, threshold=1e-3):
        '''Run one steady-state I-control tick and return convergence'''
        if not hasattr(self, '_steady_q_cmd'):
            self.init_steady_state()

        # get positional error
        q_state = self.robot_model.state['q']
        q_error = self._steady_q_cmd - q_state
        tau_bias_increment = self._steady_ki * q_error * self.dt
        # close to clipping limit
        if self._steady_state_hits_position_clip(
            q_state,
            tau_bias_increment,
            threshold,
        ):
            tau_gravity = self.robot_model.dynamics.get_gravity_compensation(
                self.robot_model.state['q']
            )
            tau_cmd = tau_gravity + self._steady_tau_bias
            self.low_cmd_handler.set_joint_commands(
                self._steady_q_cmd,
                self._steady_dq_cmd,
                tau_cmd,
            )
            self._sync_to_command()
            self.update_robot_model()
            return True
        else:
            self._steady_tau_bias += tau_bias_increment
            self._steady_tau_bias = np.clip(
                self._steady_tau_bias,
                -self._steady_tau_bias_limit,
                self._steady_tau_bias_limit,
            )

            # compute total torque to command
            tau_gravity = self.robot_model.dynamics.get_gravity_compensation(
                self.robot_model.state['q']
            )
            tau_cmd = tau_gravity + self._steady_tau_bias
            self.low_cmd_handler.set_joint_commands(
                self._steady_q_cmd,
                self._steady_dq_cmd,
                tau_cmd,
            )
            self._sync_to_command()

            # check convergence
            self.update_robot_model()
            q_error = self._steady_q_cmd - self.robot_model.state['q']
            return bool(np.max(np.abs(q_error[self.upper_ids])) < threshold)

    def _limit_joint_vel(self, vel):
        # get end effector twist
        twist_left = self.robot_model.dynamics.compute_frame_twist(self.left_ee_name, vel)
        twist_right = self.robot_model.dynamics.compute_frame_twist(self.right_ee_name, vel)
        # compute end effector velocity and angular velocity
        v_left, w_left = twist_left[:3], twist_left[3:]
        v_right, w_right = twist_right[:3], twist_right[3:]

        # joint-level limit
        left_dq_scaler = np.min(self.dq_lim / (np.abs(vel[LEFT_ARM_INDEX]) + 1e-6))
        right_dq_scaler = np.min(self.dq_lim / (np.abs(vel[RIGHT_ARM_INDEX]) + 1e-6))

        # scale left and right end effectors
        left_scaler = min([
            1.0,
            left_dq_scaler,
            self.v_lim / (np.linalg.norm(v_left) + 1e-6),
            self.w_lim / (np.linalg.norm(w_left) + 1e-6)
        ])
        right_scaler = min([
            1.0,
            right_dq_scaler,
            self.v_lim / (np.linalg.norm(v_right) + 1e-6),
            self.w_lim / (np.linalg.norm(w_right) + 1e-6)
        ])

        # legs and waist pass through; only the arms are scaled
        vel_scaled = np.zeros_like(vel)
        vel_scaled[:LEFT_ARM_INDEX[0]] = vel[:LEFT_ARM_INDEX[0]]
        vel_scaled[LEFT_ARM_INDEX] = left_scaler * vel[LEFT_ARM_INDEX]
        vel_scaled[RIGHT_ARM_INDEX] = right_scaler * vel[RIGHT_ARM_INDEX]

        return vel_scaled

    def _apply_joint_position(self, q):
        # the waist is held, not IK-driven: every command carries its hold
        # angles, never the measured (sagging) waist the IK was seeded from
        q = np.array(q, dtype=np.float64)
        q[WAIST_INDEX] = self._waist_hold
        # get gravity compensation torque
        tau = self.robot_model.dynamics.get_gravity_compensation(self.robot_model.state['q'])
        dq = np.zeros(self.robot_model.model_body.nv)
        self.low_cmd_handler.set_joint_commands(q, dq, tau)
        self._sync_to_command()

    def _sync_to_command(self):
        '''
        Until the safety layer relays the commands, restart the IK, the waist
        hold and the steady-state target from the command the follow gate let
        through, so none of them runs ahead of the robot; never after that
        '''
        if self.low_cmd_handler.following:
            return
        q_cmd = np.asarray(self.low_cmd_handler.q_cmd, dtype=np.float64)
        self.ik_solver.configuration.update(q_cmd)
        self.ik_solver.configuration_reduced.update(q_cmd[self.robot_model.reduced_mask])
        self._waist_hold = q_cmd[WAIST_INDEX]
        if hasattr(self, '_steady_q_cmd'):
            self._steady_q_cmd = q_cmd

    def estop(self):
        self.low_cmd_handler.estop()

    def shutdown(self):
        self.stop_recording()
        if self.planner is not None:
            self.planner.shutdown()
        # stop publishing before rt/lowstate goes: without it the robot model
        # falls back to its local all-zero pose, and the follow gate would pull
        # the published command toward it
        self.low_cmd_handler.shutdown()
        self.robot_model.shutdown()

    def damp_mode(self, kd=3.0):
        '''Damp the arms (kp = 0); the waist keeps holding so the torso cannot fold'''
        ids = self.enabled_ids
        self.low_cmd_handler.set_joint_gains(
            kp=np.zeros(len(ids)), kd=kd * np.ones(len(ids)), joint_ids=ids
        )
        self.low_cmd_handler.set_joint_commands(dq=np.zeros(len(ids)), joint_ids=ids)
        print(f'Set arm kp to zero, kd to {kd} and dq to 0; waist still holding')

    def start_recording(self, save_path, filename, record_interval=1.0):
        '''Start recording with background saving'''
        self.low_cmd_handler.start_recording(self.upper_ids, save_path, filename, record_interval)

    def stop_recording(self):
        '''Stop recording'''
        self.low_cmd_handler.stop_recording()

    def clear_recording(self):
        '''Clear recorded'''
        self.low_cmd_handler.clear_recording()
