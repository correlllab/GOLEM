"""The two-agent MARL controller, without ROS: robot state in, 27 joint targets out.

One step() is one env step of LocoManip-Marl-Flat-IK-Direct-v0, in the env's
order (locomanip_marl_env.py _get_observations, then _pre_physics_step):

  1. pelvis odometry: estimate how the pelvis moved over the last step and
     move the arm command by it (ArmTargetsCommand.apply_pelvis_motion);
  2. commands: advance the settle timer, switch mode;
  3. observations: the legs' (96) and the arms' (124), one fewer each without
     height_drop_obs;
  4. actions: clamp to +-clip, legs = scale * a + default, arms = q + IK step
     + residual_scale * r, both bounded to the hard limits +- target_margin
     (negative: inside them), narrowed to target_limits where it names a joint
     (the run's GOLEM position e-stop range, from export.yaml). r is the
     arms' action, low-passed when arms.residual_cutoff_hz is set
     (mdp.IKResidualArmAction), and is what the arms observe as their last
     action. With torque_headroom set, each target
     is then kept where the PD torque it asks for at the measured state,
     kp (target - q) - kd qd, is within torque_headroom x effort_limits
     (mdp.actions.torque_bounds).

Odometry is cfg["odometry"]:
  learned  the pelvis-motion estimator on its history window and the action;
  legs     rotation from the IMU attitude (taken back to the pelvis through
           the torso joint when the IMU sits on the torso), translation from
           the planted feet: a foot that stays put moves in the pelvis frame
           by the pelvis's motion, p(t-1) - dR p(t) (LocoManipMarlEnv.
           _leg_odometry). Training called a foot planted from its contact
           force; with no foot force sensor, a foot counts as planted here when
           it is within stance_height (m, gravity axis) of the lowest foot.

Modes follow the training command schedule: navigation (velocity command
live, arms holding the rest pose) or an arm goal (velocity zero, one wrist
pose per arm fixed in the world), with a zero-velocity settle at each switch.
The operator's arm goal is a pelvis-frame pose taken at request time; the
odometry estimate keeps it on its world point from then on, through the
settle too.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from lmg_deploy.kinematics import ArmKinematics
from lmg_deploy.rotations import (
    compute_pose_error,
    projected_gravity,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_from_rotvec,
    quat_unique,
    subtract_frame_transforms,
    wxyz_to_xyzw,
)

NUM_JOINTS = 27
NUM_LEGS = 12
TORSO = 12
ARM_START = 13
NUM_ARM_JOINTS = 14


@dataclass
class RobotState:
    q: np.ndarray       # (27,) rad, policy order
    dq: np.ndarray      # (27,) rad/s
    tau: np.ndarray     # (27,) Nm, measured
    quat: np.ndarray    # (4,) IMU orientation, wxyz
    gyro: np.ndarray    # (3,) rad/s, IMU frame
    acc: np.ndarray     # (3,) m/s^2, IMU frame, accelerometer convention (+9.81 up at rest)


@dataclass
class JointCommand:
    q: np.ndarray       # (27,) targets
    kp: np.ndarray
    kd: np.ndarray


class Mode(Enum):
    NAV = "nav"
    SETTLE_TO_ARM = "settle_to_arm"
    ARM = "arm_goal"
    SETTLE_TO_NAV = "settle_to_nav"


class MarlController:
    def __init__(self, cfg: dict, kinematics: ArmKinematics, legs, arms, estimator):
        """legs / arms / estimator: callables numpy -> numpy (networks.py)."""
        self.cfg = cfg
        self.kin = kinematics
        self.legs, self.arms, self.estimator = legs, arms, estimator
        self.dt = float(cfg["control_dt"])
        self.default = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
        self.kp = np.asarray(cfg["kp"], dtype=np.float64)
        self.kd = np.asarray(cfg["kd"], dtype=np.float64)
        self.clip = float(cfg["clip_actions"])
        self.lower, self.upper = self._position_bounds(kinematics.joint_limits, float(cfg["target_margin"]),
                                                       cfg.get("target_limits"))
        self.height_drop_obs = bool(cfg.get("height_drop_obs", True))
        self.legs_scale = float(cfg["legs"]["action_scale"])
        self.arms_scale = float(cfg["arms"]["residual_scale"])
        self.ik_damping = float(cfg["arms"]["ik_damping"])
        self.max_ik_step = float(cfg["arms"]["max_ik_step"])
        self.history = int(cfg["estimator"]["history"])
        self.gait_period = float(cfg["gait_period"])
        self.zero_cmd_eps = float(cfg["zero_cmd_eps"])
        ranges = cfg["command_ranges"]
        self.cmd_low = np.array([ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_z"][0]])
        self.cmd_high = np.array([ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]])
        self.settle_to_arm_s = float(cfg["settle_to_arm_s"])
        self.settle_to_nav_s = float(cfg["settle_to_nav_s"])
        self.num_arms = len(cfg["arms"]["ee_frames"])
        self.torque_headroom = cfg.get("torque_headroom")
        if self.torque_headroom is not None:
            self.effort_limits = np.asarray(cfg["effort_limits"], dtype=np.float64)
        cutoff = cfg["arms"].get("residual_cutoff_hz")
        self.filter_alpha = 1.0 if cutoff is None else 1.0 - np.exp(-2.0 * np.pi * float(cutoff) * self.dt)
        self.odometry = cfg.get("odometry", "learned")
        if self.odometry not in ("learned", "legs"):
            raise ValueError(f"odometry must be 'learned' or 'legs', not {self.odometry!r}")
        legs_odom = cfg.get("legs_odometry", {})
        self.foot_frames = legs_odom.get("foot_frames", [])
        self.imu_frame = legs_odom.get("imu_frame", "pelvis")
        self.stance_height = float(legs_odom.get("stance_height", 0.02))
        if self.odometry == "legs" and not self.foot_frames:
            raise ValueError("odometry 'legs' needs legs_odometry.foot_frames")
        if len(self.default) != NUM_JOINTS or len(self.kp) != NUM_JOINTS or len(self.kd) != NUM_JOINTS:
            raise ValueError("default_joint_pos, kp and kd need one entry per policy joint (27)")
        if cfg.get("rest_pose") is not None:
            self.rest_pose = np.asarray(cfg["rest_pose"], dtype=np.float64).reshape(self.num_arms, 7)
        else:
            self.rest_pose = self._rest_pose_from_fk()
        for arm in range(self.num_arms):
            self.rest_pose[arm, 3:] = quat_unique(self.rest_pose[arm, 3:])
        self.cmd_vel = np.zeros(3)
        self.reset(None)

    # -- lifecycle -------------------------------------------------------------
    def reset(self, state: RobotState | None) -> None:
        """Start an episode at the measured state: navigation, standing, empty histories."""
        self.mode = Mode.NAV
        self.timer = 0.0
        self.pending_goal: np.ndarray | None = None
        self.believed = self.rest_pose.copy()   # (arms, 7) pelvis frame, wxyz: the arm command
        self.steps = 0                          # since reset: the gait clock
        self._frames: list[np.ndarray] = []     # estimator history, oldest first
        self._prev_action: np.ndarray | None = None
        # the action terms' state after a reset: legs see the applied target in action units, arms zero
        q_legs = state.q[:NUM_LEGS] if state is not None else self.default[:NUM_LEGS]
        self.legs_last = (q_legs - self.default[:NUM_LEGS]) / self.legs_scale
        self.arms_last = np.zeros(NUM_ARM_JOINTS)
        self.residual = np.zeros(NUM_ARM_JOINTS)    # the arms' action after the low-pass
        self._prev_pelvis_quat: np.ndarray | None = None
        self._prev_feet: np.ndarray | None = None
        self.last_obs: dict[str, np.ndarray] = {}

    # -- operator commands -------------------------------------------------------
    def set_velocity_command(self, cmd: np.ndarray) -> None:
        """(vx, vy, wz), body frame, clipped to the training ranges. Applies while navigating."""
        self.cmd_vel = np.clip(np.asarray(cmd, dtype=np.float64), self.cmd_low, self.cmd_high)

    def request_arm_goal(self, poses: np.ndarray) -> None:
        """poses: (arms, 7) wrist targets (x, y, z, qw, qx, qy, qz) in the pelvis frame, now."""
        goal = np.asarray(poses, dtype=np.float64).reshape(self.num_arms, 7).copy()
        for arm in range(self.num_arms):
            goal[arm, 3:] = quat_unique(goal[arm, 3:] / np.linalg.norm(goal[arm, 3:]))
        if self.mode == Mode.ARM:
            self.believed = goal           # a goal replacing a goal: no settle, as in training
        elif self.settle_to_arm_s > 0.0:
            self.pending_goal = goal
            self.mode, self.timer = Mode.SETTLE_TO_ARM, self.settle_to_arm_s
        else:
            self._start_arm_goal(goal)

    def end_arm_goal(self) -> None:
        if self.mode == Mode.ARM:
            self.believed = self.rest_pose.copy()
            if self.settle_to_nav_s > 0.0:
                self.mode, self.timer = Mode.SETTLE_TO_NAV, self.settle_to_nav_s
            else:
                self.mode = Mode.NAV
        elif self.mode == Mode.SETTLE_TO_ARM:
            self.pending_goal = None
            self.mode = Mode.NAV

    # -- the env step ----------------------------------------------------------------
    def step(self, state: RobotState) -> JointCommand:
        self.kin.update(state.q)
        self._update_odometry(state)
        self._update_mode()

        common = self._common_obs(state)
        legs_obs = np.concatenate([common, self.legs_last])
        arms_obs = np.concatenate([common, self.arms_last, self._wrist_poses(), self._wrist_errors()])
        self.last_obs = {"legs": legs_obs, "arms": arms_obs}

        legs_action = np.clip(self.legs(legs_obs), -self.clip, self.clip)
        arms_action = np.clip(self.arms(arms_obs), -self.clip, self.clip)

        target = self.default.copy()
        legs_target = legs_action * self.legs_scale + self.default[:NUM_LEGS]
        target[:NUM_LEGS] = self._torque_bound(self._bound(legs_target, slice(0, NUM_LEGS)), state, slice(0, NUM_LEGS))
        ik = state.q[ARM_START:].copy()
        for arm in range(self.num_arms):
            cols = self.kin.arm_cols[arm] - ARM_START
            ik[cols] += self.kin.ik_step(arm, self.believed[arm, :3], self.believed[arm, 3:],
                                         self.ik_damping, self.max_ik_step)
        self.residual = self.residual + self.filter_alpha * (arms_action - self.residual)
        arms = slice(ARM_START, NUM_JOINTS)
        target[arms] = self._torque_bound(self._bound(ik + self.residual * self.arms_scale, arms), state, arms)

        # what each agent observes of its action next step (mdp.applied_action)
        self.legs_last = (target[:NUM_LEGS] - self.default[:NUM_LEGS]) / self.legs_scale
        self.arms_last = self.residual.copy()
        self._prev_action = np.concatenate([legs_action, arms_action])
        self.steps += 1
        return JointCommand(q=target, kp=self.kp.copy(), kd=self.kd.copy())

    # -- internals ----------------------------------------------------------------------
    def _position_bounds(self, hard: np.ndarray, margin: float, target_limits: dict | None):
        """(lower, upper) per joint: the hard limits widened by margin, narrowed to target_limits[name]."""
        lower, upper = hard[:, 0] - margin, hard[:, 1] + margin
        names = list(self.cfg["joint_names"])
        for name, (low, high) in (target_limits or {}).items():
            if name not in names:
                raise ValueError(f"target_limits names {name!r}, not a policy joint")
            i = names.index(name)
            lower[i], upper[i] = max(lower[i], float(low)), min(upper[i], float(high))
        if np.any(lower > upper):
            raise ValueError("target bounds are empty for some joint")
        return lower, upper

    def _bound(self, target: np.ndarray, joints: slice) -> np.ndarray:
        return np.minimum(np.maximum(target, self.lower[joints]), self.upper[joints])

    def _torque_bound(self, target: np.ndarray, state: RobotState, joints: slice) -> np.ndarray:
        if self.torque_headroom is None:
            return target
        q, dq = state.q[joints], state.dq[joints]
        kp, kd = self.kp[joints], self.kd[joints]
        tau = self.torque_headroom * self.effort_limits[joints]
        return np.minimum(np.maximum(target, q + (kd * dq - tau) / kp), q + (kd * dq + tau) / kp)

    def _start_arm_goal(self, goal: np.ndarray) -> None:
        self.believed = goal
        self.pending_goal = None
        self.mode, self.timer = Mode.ARM, 0.0

    def _update_mode(self) -> None:
        if self.mode not in (Mode.SETTLE_TO_ARM, Mode.SETTLE_TO_NAV):
            return
        self.timer -= self.dt
        if self.timer > 1e-9:
            return
        if self.mode == Mode.SETTLE_TO_ARM:
            self._start_arm_goal(self.pending_goal)
        else:
            self.mode = Mode.NAV

    def velocity_command(self) -> np.ndarray:
        return self.cmd_vel.copy() if self.mode == Mode.NAV else np.zeros(3)

    def _gait_phase(self, cmd: np.ndarray) -> np.ndarray:
        if np.linalg.norm(cmd) < self.zero_cmd_eps:
            return np.zeros(2)
        phase = (self.steps * self.dt % self.gait_period) / self.gait_period
        return np.sin(2.0 * np.pi * np.array([phase, (phase + 0.5) % 1.0]))

    def _common_obs(self, state: RobotState) -> np.ndarray:
        cmd = self.velocity_command()
        ee_targets = np.concatenate([np.concatenate([p[:3], wxyz_to_xyzw(p[3:])]) for p in self.believed])
        return np.concatenate([
            state.acc,
            state.gyro,
            projected_gravity(state.quat),
            cmd,
            [1.0 if self.mode == Mode.ARM else 0.0],
            ee_targets,
            [0.0] if self.height_drop_obs else [],   # height_drop: always shown as zero
            state.q - self.default,
            state.dq,
            self._gait_phase(cmd),
        ])

    def _wrist_poses(self) -> np.ndarray:
        out = []
        for arm in range(self.num_arms):
            pos, quat = self.kin.wrist_pose(arm)
            out.append(np.concatenate([pos, wxyz_to_xyzw(quat)]))
        return np.concatenate(out)

    def _wrist_errors(self) -> np.ndarray:
        out = []
        for arm in range(self.num_arms):
            pos, quat = self.kin.wrist_pose(arm)
            pos_err, rot_err = compute_pose_error(pos, quat, self.believed[arm, :3], self.believed[arm, 3:])
            out.append(np.concatenate([pos_err, rot_err]))
        return np.concatenate(out)

    def wrist_errors(self) -> np.ndarray:
        """(arms,) position error to the command (m), for logging."""
        return np.linalg.norm(self._wrist_errors().reshape(self.num_arms, 6)[:, :3], axis=1)

    def _update_odometry(self, state: RobotState) -> None:
        if self.odometry == "legs":
            motion = self._leg_odometry(state)
            if motion is not None:
                self._move_goal(*motion)
            return
        frame = np.concatenate([
            state.acc, state.gyro, projected_gravity(state.quat),
            state.q - self.default, state.dq, state.tau[:NUM_LEGS],
        ])
        if not self._frames:
            self._frames = [frame] * self.history   # the first push fills the window, as CircularBuffer does
        else:
            self._frames = self._frames[1:] + [frame]
        if self._prev_action is None:
            return
        motion = self.estimator(np.concatenate([self._history_input(), self._prev_action]))
        self._move_goal(motion[:3] * self.dt, quat_from_rotvec(motion[3:] * self.dt))

    def _pelvis_quat(self, state: RobotState) -> np.ndarray:
        """The pelvis's world orientation from the IMU's: through the torso joint when the IMU is on the torso."""
        if self.imu_frame == "pelvis":
            return state.quat
        _, quat_imu_in_pelvis = self.kin.frame_pose(self.imu_frame)
        return quat_mul(state.quat, quat_conjugate(quat_imu_in_pelvis))

    def _leg_odometry(self, state: RobotState) -> tuple[np.ndarray, np.ndarray] | None:
        """The pelvis's motion since the last step, in the previous pelvis frame: (delta_pos, delta_quat)."""
        quat = self._pelvis_quat(state)
        feet = np.stack([self.kin.frame_pose(f)[0] for f in self.foot_frames])  # pelvis frame
        prev_quat, prev_feet = self._prev_pelvis_quat, self._prev_feet
        self._prev_pelvis_quat, self._prev_feet = quat, feet
        if prev_quat is None:
            return None
        delta_quat = quat_mul(quat_conjugate(prev_quat), quat)
        moved = np.stack([prev_feet[i] - quat_apply(delta_quat, feet[i]) for i in range(len(feet))])
        height = np.array([quat_apply(quat, foot)[2] for foot in feet])  # along gravity, from the pelvis
        planted = height <= height.min() + self.stance_height
        return moved[planted].mean(axis=0), delta_quat

    def _move_goal(self, delta_pos: np.ndarray, delta_quat: np.ndarray) -> None:
        if self.mode not in (Mode.ARM, Mode.SETTLE_TO_ARM):
            return
        # a world-fixed target: T_prev o c_prev = T_new o c_new, so c_new = delta^-1 o c_prev
        poses = self.believed if self.mode == Mode.ARM else self.pending_goal
        for arm in range(self.num_arms):
            pos, quat = subtract_frame_transforms(delta_pos, delta_quat, poses[arm, :3], poses[arm, 3:])
            poses[arm, :3], poses[arm, 3:] = pos, quat_unique(quat)

    def _history_input(self) -> np.ndarray:
        """The odometry group's layout: term by term, each term's frames oldest first."""
        frames = np.stack(self._frames)              # (history, frame)
        sizes = [3, 3, 3, NUM_JOINTS, NUM_JOINTS, NUM_LEGS]
        bounds = np.cumsum([0] + sizes)
        return np.concatenate([frames[:, lo:hi].reshape(-1) for lo, hi in zip(bounds[:-1], bounds[1:])])

    def _rest_pose_from_fk(self) -> np.ndarray:
        self.kin.update(self.default)
        return np.stack([np.concatenate(self.kin.wrist_pose(arm)) for arm in range(self.num_arms)])
