"""Run the two-agent MARL controller without ROS: robot state in, 27 joint targets out.

One step() is one env step of LocoManip-Marl-Direct-v0, in the env's order
(locomanip_marl_env.py _get_observations, then _pre_physics_step):

  1. commands: the velocity command's low-pass;
  2. observations: each agent's terms in export.yaml's legs_obs_terms /
     arms_obs_terms (OBS_TERMS' order, any of them left out; gait_phase_cos,
     each leg's phase cos after its sin, only with cfg["gait_phase_cos"]);
  3. actions: clamp to +-clip; legs = scale * a + default (one scale, or per
     joint from export.yaml's leg_action_scales), the arms as
     mdp.IKPostureArmAction does (below); both bounded to the hard limits +-
     target_margin (negative: inside them), narrowed to target_limits where it
     names a joint (the run's GOLEM position e-stop range, from export.yaml).
     With torque_headroom set, each target is then kept where the PD torque it
     asks for at the measured state, kp (target - q) - kd qd, is within
     torque_headroom x effort_limits (mdp.actions.torque_bounds).

The arms (export.yaml's arm_ik: mdp.IKPostureArmAction, LocoManip-Marl-Direct-v0
runs; Locomanipulation_game's mujoco_evals/mujoco_sim.py MarlMujoco is the
numpy twin). Per step, from the arms' 14-wide action a:
  - residual: integrated (residual_integrate: leak exp(-dt / leak_s), plus
    gain x a, clamped to +-clamp); held while frozen; the arms observe it as
    their last action;
  - posture: per arm seed_q (during an arm goal the joint angles of
    arm_seed_table's row nearest the arm command, position and rotation at
    seed_rot_weight m per rad, with seed_solve solved for the command itself;
    set once per goal, as training draws it; a rest-pose goal and navigation:
    the default angles), inside the target bounds narrowed by limit_margin;
  - trajectory: the target ik_q moves from its own last value (not the
    measured angles) to the posture, each arm's joints together at
    trajectory_max_vel / trajectory_max_acc;
  - target: ik_q + gravity offset (gravity_offset: minus each arm joint's
    gravity torque at ik_q over its kp, gravity in the pelvis frame from the
    IMU attitude carried through imu.frame's pose) + integral + residual scale
    (arm_residual_scales) x residual, then the bounds above;
  - integral: leaks (integral.leak_s); once ik_q has arrived at the posture,
    integral.gain x each arm's rot_weight-weighted DLS step on the measured
    wrist's error is added; clamped to +-integral.clamp;
  - freeze: the residual is latched once ik_q has arrived and both wrists have
    been within freeze.pos_tol / rot_tol of the arm command for settle_s;
    released when the posture moves or the arm goal ends.
reset() starts ik_q at the measured arm angles with everything else zero.

The export format is the shipped policy's (_check_format): an older export
(no arm_ik or another arm_ik variant, world-frame arm goals on SLAM odometry,
arm goals that stop walking or none at all, no term lists, an IMU other than
the calibrated torso IMU) is refused, not run as if it were this one.

With command_filter_tau (export.yaml; s), the velocity command the policies
see and the gait clock reads is the operator's through a first-order low-pass
stepped once per control step and zeroed at reset, as ModalVelocityCommand
filters it in training. It stays live through an arm goal
(arm_goal_keeps_walking).

Recurrent actors (networks.RecurrentTorchScriptNetwork) carry their LSTM
state from step to step; reset() zeroes it, as training did at each episode
start.

Engage blend: with export.yaml's engage_blend_s (s; absent or 0: none), over the
first engage_blend_s of an episode the legs' target goes linearly from the
target the legs held before it (reset()'s held_target: the last command sent,
the pre-pose's hold) to the policy's, then is bounded as in step 3; the legs
observe the blended target as their last action (mdp.BoundedJointPositionAction's
engage_blend_s). It keeps the first commands continuous with the held ones when
the policy's first targets are far from them (an engage from a sagged pose).

Leg target low-pass: with export.yaml's leg_action_lowpass_hz (Hz; absent or 0:
none), after the engage blend the legs' target goes through a one-pole low-pass
per control step, y += alpha (target - y) with alpha = 1 - exp(-2 pi f
control_dt), and is then bounded as in step 3, so the bounds hold every step
(mdp.BoundedJointPositionAction's target_lowpass_hz). Its state y is the filtered
target before the bounds; reset() starts it at held_target (the engage), else
at the measured legs (the default pose without a state). The legs observe the
filtered, bounded target as their last action.

IMU calibration (imu_calibration.py): the policy trained on a level torso IMU
(export.yaml imu.orientation calibrated), so the controller removes the
robot's static torso-IMU offset (MarlController's imu_calibration) from every
state it reads; the observations, the arms' gravity offset and step() all see
the corrected attitude.

Modes follow the training command schedule: navigation (arms holding the rest
pose) until the first arm goal, then the arm goal (one wrist pose per arm, in
the pelvis frame, riding on the pelvis as arm_goal_frame pelvis trained it).
A goal may target one arm; the other keeps its command and posture seed.
hold_arms() stops arms where they are (a canceled or timed-out goal).
"""

import logging
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from lmg_deploy.imu_calibration import correct_imu, resolve_calibration
from lmg_deploy.kinematics import ArmKinematics
from lmg_deploy.rotations import compute_pose_error, projected_gravity, quat_apply, quat_unique, wxyz_to_xyzw

NUM_JOINTS = 27
NUM_LEGS = 12
ARM_START = 13
NUM_ARM_JOINTS = 14
# The observation terms step() can build, in order (gait_phase_cos only with cfg["gait_phase_cos"]). An export's
# legs_obs_terms / arms_obs_terms name the ones its actors saw, in this order, any left out.
COMMON_OBS_TERMS = ["base_ang_vel", "projected_gravity", "velocity_commands", "ee_targets", "joint_pos", "joint_vel",
                    "gait_phase", "gait_phase_cos"]
LEGS_OBS_TERMS = COMMON_OBS_TERMS + ["actions"]
ARMS_OBS_TERMS = COMMON_OBS_TERMS + ["actions", "wrist_poses", "wrist_errors"]
GRAVITY = 9.81             # m/s^2, the training sim's
SOFT_LIMIT_FACTOR = 0.9    # the training asset's soft_joint_pos_limit_factor (H1_2_MAGPIE_CFG): the seed solve's range
LOG = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Hold the measured state one step reads."""

    q: np.ndarray       # (27,) rad, policy order
    dq: np.ndarray      # (27,) rad/s
    quat: np.ndarray    # (4,) IMU orientation, wxyz
    gyro: np.ndarray    # (3,) rad/s, IMU frame


@dataclass
class JointCommand:
    """Hold one step's joint targets and gains, in policy order."""

    q: np.ndarray       # (27,) rad
    kp: np.ndarray      # (27,) Nm/rad
    kd: np.ndarray      # (27,) Nm s/rad


class Mode(Enum):
    """Name the controller's command modes; the values are what ~/state publishes."""

    NAV = "nav"
    ARM = "arm_goal"


def _check_format(cfg: dict) -> None:
    """Raise ValueError, saying what and why, on a config of an export format this controller does not run.

    It runs the shipped policy's format (module doc); each refused key or value selects behaviour older exports
    trained with and this controller no longer has.
    """
    def refuse(what: str) -> None:
        """Raise the refusal for what."""
        raise ValueError(f"export.yaml {what}: an older export format, no longer supported by this lmg_deploy (it "
                         "runs LocoManip-Marl-Direct-v0 exports from 2026-11-20 on: README 'Exporting a checkpoint')")

    for key in ("settle_to_arm_s", "settle_to_nav_s"):
        if float(cfg.get(key) or 0.0) != 0.0:
            raise ValueError(f"{key} {cfg[key]}: this lmg_deploy switches between navigation and an arm goal at "
                             "once, as the shipped policy trained; a settle at the switch is no longer supported")
    if cfg.get("arm_goals", True) is not True:
        refuse(f"arm_goals {cfg['arm_goals']!r} (a run that never trained an arm goal)")
    if cfg.get("arm_goal_frame") != "pelvis":
        refuse(f"arm_goal_frame {cfg.get('arm_goal_frame')!r} (world-frame arm goals kept by SLAM odometry; "
               "needs 'pelvis')")
    if cfg.get("arm_goal_keeps_walking") is not True:
        refuse(f"arm_goal_keeps_walking {cfg.get('arm_goal_keeps_walking')!r} (arm goals that stop walking; "
               "needs true)")
    for agent in ("legs", "arms"):
        if cfg.get(f"{agent}_obs_terms") is None:
            refuse(f"without {agent}_obs_terms (every observation term)")
    if cfg.get("rest_pose") is None:
        refuse("without rest_pose (no arm target tables)")
    ik = cfg.get("arm_ik")
    if not ik:
        refuse("without arm_ik (mdp.IKResidualArmAction's arms)")
    for key, needed in (("posture_actions", False), ("trajectory", True), ("posture_from_command", True)):
        if bool(ik.get(key)) != needed:
            refuse(f"arm_ik.{key} {ik.get(key)!r} (needs {str(needed).lower()})")
    for key in ("trajectory_max_vel", "trajectory_max_acc", "residual_integrate"):
        if ik.get(key) is None:
            refuse(f"arm_ik without {key}")


class MarlController:
    """Step the legs and arms policies and turn their actions into bounded joint targets."""

    def __init__(self, cfg: dict, kinematics: ArmKinematics, legs, arms, imu_calibration: dict | None = None):
        """Read the deploy config (config/example.yaml, export.yaml's keys over it) and start an episode.

        legs / arms: callables numpy -> numpy (networks.py). kinematics needs the IMU frame (imu.frame) among its extra
        frames for arm_ik's gravity offset. imu_calibration: the robot's torso-IMU calibration, {"roll_deg",
        "pitch_deg", "yaw_deg", "rotate_gyro"} (config/imu_calibration.yaml) (imu_calibration.resolve_calibration).
        Raises ValueError on an export format it does not run (_check_format), on a config whose sizes or target
        bounds are wrong, and without a calibration.
        """
        _check_format(cfg)
        self.cfg = cfg
        # the calibration applied to every state read, and the line saying so
        self.imu_calibration, self.imu_calibration_note = resolve_calibration(cfg, imu_calibration)
        LOG.info(self.imu_calibration_note)
        self.kin = kinematics
        self.legs, self.arms = legs, arms
        self.dt = float(cfg["control_dt"])
        self.default = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
        self.kp = np.asarray(cfg["kp"], dtype=np.float64)
        self.kd = np.asarray(cfg["kd"], dtype=np.float64)
        self.clip = float(cfg["clip_actions"])
        self.lower, self.upper = self._position_bounds(kinematics.joint_limits, float(cfg["target_margin"]),
                                                       cfg.get("target_limits"))
        self.legs_scale = self._scales(cfg.get("leg_action_scales"), cfg["legs"]["action_scale"], NUM_LEGS)
        self.arms_scale = self._scales(cfg.get("arm_residual_scales"), cfg["arms"]["residual_scale"], NUM_ARM_JOINTS)
        tau = float(cfg.get("command_filter_tau") or 0.0)
        self.cmd_alpha = 1.0 - np.exp(-self.dt / tau) if tau > 0.0 else None
        self.gait_period = float(cfg["gait_period"])
        self.engage_blend_steps = int(round(float(cfg.get("engage_blend_s") or 0.0) / self.dt))
        # export.yaml's leg_action_lowpass_hz (module doc, leg target low-pass); 1.0: no filter
        lowpass_hz = float(cfg.get("leg_action_lowpass_hz") or 0.0)
        self.legs_lowpass_alpha = 1.0 - np.exp(-2.0 * np.pi * lowpass_hz * self.dt) if lowpass_hz > 0.0 else 1.0
        self.zero_cmd_eps = float(cfg["zero_cmd_eps"])
        self.gait_phase_cos = bool(cfg.get("gait_phase_cos", False))
        self.obs_terms = {agent: self._agent_terms(cfg[f"{agent}_obs_terms"], terms)
                          for agent, terms in (("legs", LEGS_OBS_TERMS), ("arms", ARMS_OBS_TERMS))}
        ranges = cfg["command_ranges"]
        self.cmd_low = np.array([ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_z"][0]])
        self.cmd_high = np.array([ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]])
        self.num_arms = len(cfg["arms"]["ee_frames"])
        self.torque_headroom = cfg.get("torque_headroom")
        if self.torque_headroom is not None:
            self.effort_limits = np.asarray(cfg["effort_limits"], dtype=np.float64)
        self._init_arm_ik(cfg)
        if len(self.default) != NUM_JOINTS or len(self.kp) != NUM_JOINTS or len(self.kd) != NUM_JOINTS:
            raise ValueError("default_joint_pos, kp and kd need one entry per policy joint (27)")
        self.rest_pose = np.asarray(cfg["rest_pose"], dtype=np.float64).reshape(self.num_arms, 7)
        for arm in range(self.num_arms):
            self.rest_pose[arm, 3:] = quat_unique(self.rest_pose[arm, 3:])
        self.cmd_vel = np.zeros(3)
        self.cmd_filtered = np.zeros(3)
        self.reset(None)

    def reset(self, state: RobotState | None, held_target: np.ndarray | None = None) -> None:
        """Start an episode at the measured state (None: the default pose): navigation, empty histories.

        held_target: the 27 joint targets last sent before the episode (rad): the engage blend starts there (module
        doc, engage blend). None: no blend (a restart of a running policy, the band's release in the sim).
        """
        self.mode = Mode.NAV
        self.believed = self.rest_pose.copy()   # (arms, 7) pelvis frame, wxyz: the arm command
        self.steps = 0                          # since reset: the gait clock
        self.cmd_filtered = np.zeros(3)         # the command low-pass starts from zero, as training's does
        # The action terms' state after a reset: the legs see the applied target in action units, the arms zero.
        q_legs = state.q[:NUM_LEGS] if state is not None else self.default[:NUM_LEGS]
        self.legs_last = (q_legs - self.default[:NUM_LEGS]) / self.legs_scale
        # rad, the engage blend's start (None: no blend)
        self.blend_from = (None if held_target is None or not self.engage_blend_steps
                           else np.asarray(held_target, dtype=np.float64)[:NUM_LEGS].copy())
        # rad, the leg target low-pass's state: from the held target at the engage, else the measured legs
        self.legs_lowpass = (np.asarray(held_target, dtype=np.float64)[:NUM_LEGS].copy() if held_target is not None
                             else np.asarray(q_legs, dtype=np.float64).copy())
        self.arms_last = np.zeros(NUM_ARM_JOINTS)
        self.residual = np.zeros(NUM_ARM_JOINTS)   # the arms' action after the integration
        # arm_ik's state, as IKPostureArmAction.reset leaves it: the target from the measured arm angles, at rest
        self.ik_q = (state.q if state is not None else self.default)[ARM_START:].copy()
        self.traj_v = np.zeros(NUM_ARM_JOINTS)      # the trajectory's velocity (rad/s)
        self.integral = np.zeros(NUM_ARM_JOINTS)    # the pose-error integral (rad)
        self.frozen, self.settled_steps = False, 0  # the residual latched (arm_ik.freeze)
        self.posture = self.default[ARM_START:].copy()
        self.seed_q = self.default[ARM_START:].copy()   # (14,) the posture's base: navigation's, the default angles
        self.last_obs: dict[str, np.ndarray] = {}
        for network in (self.legs, self.arms):
            if hasattr(network, "reset"):
                network.reset()

    def set_velocity_command(self, cmd: np.ndarray) -> None:
        """Set the velocity command (vx, vy m/s, wz rad/s, body frame), clipped to the training ranges."""
        self.cmd_vel = np.clip(np.asarray(cmd, dtype=np.float64), self.cmd_low, self.cmd_high)

    def request_arm_goal(self, poses: np.ndarray, arms=None) -> None:
        """Start an arm goal: (arms, 7) wrist targets (x, y, z, qw, qx, qy, qz) in the pelvis frame now.

        arms: the arms (indices) the goal targets, None: all. Each other arm keeps its arm command and posture, and
        its row of poses is not read. It starts at once, from navigation or replacing the goal in arm mode.
        """
        arms = list(range(self.num_arms)) if arms is None else list(arms)
        poses = np.asarray(poses, dtype=np.float64).reshape(self.num_arms, 7)
        goal = self.believed.copy()
        for arm in arms:
            goal[arm, :3] = poses[arm, :3]
            goal[arm, 3:] = quat_unique(poses[arm, 3:] / np.linalg.norm(poses[arm, 3:]))
        self.believed = goal
        self.mode = Mode.ARM
        self._update_seeds(arms)

    def hold_arms(self, q: np.ndarray, arms) -> None:
        """Stop the arms (indices) where they are: each one's command to its measured wrist pose, its posture to its
        measured joint angles (q: rad, policy order), and its trajectory stopped there (ik_q at q, at rest), so its
        target holds the measured pose instead of running on ahead of the lagging arm. The other arm keeps its
        command and trajectory."""
        self.kin.update(q)
        for arm in arms:
            pos, quat = self.kin.wrist_pose(arm)
            self.believed[arm] = np.concatenate([pos, quat])
            cols = self.kin.arm_cols[arm]
            # SAFETY: the target steps back onto the measured angles, which only lowers the PD torque it asks for
            self.seed_q[cols - ARM_START] = q[cols]
            self.ik_q[cols - ARM_START] = q[cols]
            self.traj_v[cols - ARM_START] = 0.0
        self.mode = Mode.ARM

    def step(self, state: RobotState) -> JointCommand:
        """Run one env step at the measured state and return the joint command."""
        self.kin.update(state.q)
        self._filter_command()

        self.last_obs = self.observations(state)
        legs_obs, arms_obs = self.last_obs["legs"], self.last_obs["arms"]

        legs_action = np.clip(self.legs(legs_obs), -self.clip, self.clip)
        arms_action = np.clip(self.arms(arms_obs), -self.clip, self.clip)

        target = self.default.copy()
        target[:NUM_LEGS] = self.legs_targets(legs_action, state)
        target[ARM_START:] = self._posture_arms(state, arms_action)

        # What each agent observes of its action next step (mdp.applied_action).
        self.legs_last = (target[:NUM_LEGS] - self.default[:NUM_LEGS]) / self.legs_scale
        self.arms_last = self.residual.copy()
        self.steps += 1
        return JointCommand(q=target, kp=self.kp.copy(), kd=self.kd.copy())

    def legs_targets(self, legs_action: np.ndarray, state: RobotState) -> np.ndarray:
        """Return the legs' 12 joint targets (rad) for their clipped action at state: scaled, blended at the engage
        (module doc, engage blend; by step()'s count, self.steps), low-passed (leg_action_lowpass_hz), then bounded."""
        target = legs_action * self.legs_scale + self.default[:NUM_LEGS]
        if self.blend_from is not None and self.steps < self.engage_blend_steps:
            # SAFETY: the engage's first targets stay continuous with the held ones (module doc, engage blend)
            target = self.blend_from + (self.steps + 1) / self.engage_blend_steps * (target - self.blend_from)
        if self.legs_lowpass_alpha < 1.0:
            self.legs_lowpass = self.legs_lowpass + self.legs_lowpass_alpha * (target - self.legs_lowpass)
            target = self.legs_lowpass.copy()
        return self._torque_bound(self._bound(target, slice(0, NUM_LEGS)), state, slice(0, NUM_LEGS))

    # -- arm_ik: mdp.IKPostureArmAction ------------------------------------------

    def _init_arm_ik(self, cfg: dict) -> None:
        """Read export.yaml's arm_ik block (Locomanipulation_game's read_deploy_settings writes it)."""
        self.arm_ik = c = cfg["arm_ik"]
        num_actions = cfg["arms"].get("num_actions")
        if num_actions is not None and int(num_actions) != NUM_ARM_JOINTS:
            raise ValueError(f"arms.num_actions {num_actions}, but arm_ik's action is {NUM_ARM_JOINTS} wide")
        if self.num_arms != 2:
            raise ValueError("arm_ik needs two arms")
        arms = slice(ARM_START, NUM_JOINTS)
        self.ik_lower = self.lower[arms] + float(c["limit_margin"])
        self.ik_upper = self.upper[arms] - float(c["limit_margin"])
        self.row_weight = np.array([1.0, 1.0, 1.0, c["rot_weight"], c["rot_weight"], c["rot_weight"]], dtype=np.float64)
        self.settle_steps = max(1, int(round(float(c["freeze"]["settle_s"]) / self.dt))) if c.get("freeze") else 0
        # the goal table's poses and joint angles per arm (export.yaml arm_seed_table; marl_deploy_node makes the path
        # absolute)
        table = np.load(cfg["arm_seed_table"])
        self.seed_table = [(table[f"poses_{arm}"].astype(np.float64), table[f"joints_{arm}"].astype(np.float64))
                           for arm in range(self.num_arms)]
        if c.get("seed_solve"):
            # the solve's range: training's soft limits (SOFT_LIMIT_FACTOR of the hard range) narrowed by the margin
            factor = float(c["seed_solve"].get("soft_limit_factor", SOFT_LIMIT_FACTOR))
            hard = self.kin.joint_limits
            mid, half = hard.mean(axis=1), 0.5 * (hard[:, 1] - hard[:, 0])
            margin = float(c["seed_solve"].get("joint_range_margin", 0.0))
            self.solve_low, self.solve_high = mid - factor * half + margin, mid + factor * half - margin
        # the IMU frame the state's attitude is in, for the gravity offset's pelvis-frame gravity
        self.imu_frame = cfg["imu"]["frame"]
        if c.get("gravity_offset") and self.imu_frame not in self.kin.extra_frames:
            raise ValueError(f"arm_ik.gravity_offset needs the IMU frame {self.imu_frame!r} among the kinematics' frames")

    def _update_seeds(self, arms) -> None:
        """Set seed_q for the arms (indices) whose command was just set (ArmTargetsCommand draws it once per goal): the
        seed table's nearest row, with seed_solve solved for the command; an arm commanded to its rest pose: the
        default angles. The other arm keeps its seed."""
        seed = self.default.copy()
        seed[ARM_START:] = self.seed_q
        reaching = [arm for arm in arms if not np.allclose(self.believed[arm], self.rest_pose[arm], atol=1e-9)]
        for arm in arms:
            cols = self.kin.arm_cols[arm]
            seed[cols] = self._nearest_row(arm) if arm in reaching else self.default[cols]
        if reaching and self.arm_ik.get("seed_solve"):
            c = self.arm_ik["seed_solve"]
            # each arm's solve is its own (its joints, the torso held), so the arms not reaching keep their seed
            solved = self.kin.solve_arms(self.believed, seed, self.solve_low, self.solve_high, iters=int(c["iters"]),
                                         damping=float(c["damping"]), rot_weight=float(c["rot_weight"]))
            for arm in reaching:
                seed[self.kin.arm_cols[arm]] = solved[self.kin.arm_cols[arm]]
        self.seed_q = seed[ARM_START:].copy()

    def _nearest_row(self, arm: int) -> np.ndarray:
        """Return the joint angles of the arm's seed-table row nearest its command: squared position distance plus
        (seed_rot_weight x rotation angle)^2, the angle from 8 (1 - |q . q_row|) (ArmTargetsCommand.nearest_joints)."""
        poses, joints = self.seed_table[arm]
        b = self.believed[arm]
        w2 = float(self.arm_ik["seed_rot_weight"]) ** 2
        d = ((poses[:, :3] - b[:3]) ** 2).sum(axis=1) + w2 * 8.0 * (1.0 - np.abs(poses[:, 3:] @ b[3:]))
        return joints[int(np.argmin(d))]

    def _posture_arms(self, state: RobotState, action: np.ndarray) -> np.ndarray:
        """Return the arms' 14 targets from their action, IKPostureArmAction.process_actions (module doc)."""
        c = self.arm_ik
        if action.shape != (NUM_ARM_JOINTS,):
            raise ValueError(f"arms action of {action.shape}, arm_ik expects ({NUM_ARM_JOINTS},)")
        if not self.frozen:
            ri = c["residual_integrate"]
            leak = np.exp(-self.dt / float(ri["leak_s"]))
            self.residual = np.clip(leak * self.residual + float(ri["gain"]) * action,
                                    -float(ri["clamp"]), float(ri["clamp"]))
        self.posture = np.clip(self.seed_q, self.ik_lower, self.ik_upper)
        arms = slice(ARM_START, NUM_JOINTS)
        self.ik_q = self._bound(self._synchronized_step(), arms)
        if c.get("integral"):
            self._update_integral()
        target = self.ik_q + self._gravity_offset(state, self.ik_q) + self.integral + self.arms_scale * self.residual
        target = self._torque_bound(self._bound(target, arms), state, arms)
        if c.get("freeze"):
            self._update_freeze()
        return target

    def _synchronized_step(self) -> np.ndarray:
        """Return the trajectory's next target: per arm, all joints along the straight line to the posture, the fastest
        at trajectory_max_vel, slowed to stop there, its velocity changed by at most trajectory_max_acc x dt."""
        c, dt = self.arm_ik, self.dt
        vmax, amax = float(c["trajectory_max_vel"]), float(c["trajectory_max_acc"])
        ik = self.ik_q.copy()
        for arm in range(self.num_arms):
            cols = self.kin.arm_cols[arm] - ARM_START
            delta = self.posture[cols] - self.ik_q[cols]
            dist = max(float(np.abs(delta).max()), 1e-9)
            cap = min(vmax, np.sqrt(2.0 * amax * dist))
            v = delta / dt * min(1.0, cap * dt / dist)
            dv = v - self.traj_v[cols]
            v = self.traj_v[cols] + dv * min(1.0, amax * dt / max(float(np.abs(dv).max()), 1e-12))
            self.traj_v[cols] = v
            ik[cols] = self.ik_q[cols] + v * dt
        return ik

    def _arrived(self) -> bool:
        """Return whether the trajectory's target has stopped at the posture."""
        return bool(np.abs(self.ik_q - self.posture).max() < 1e-4 and np.abs(self.traj_v).max() < 1e-3)

    def _wrist_error(self, arm: int) -> np.ndarray:
        """Return the arm's measured wrist (position, axis-angle) error to the arm command, pelvis frame."""
        pos, quat = self.kin.wrist_pose(arm)
        return np.concatenate(compute_pose_error(pos, quat, self.believed[arm, :3], self.believed[arm, 3:]))

    def _update_integral(self) -> None:
        """Leak the pose-error integral; once arrived, add gain x each arm's weighted DLS step on its wrist's error."""
        c, ik = self.arm_ik["integral"], self.arm_ik
        self.integral *= np.exp(-self.dt / float(c["leak_s"]))
        if self._arrived():
            damp = float(ik["ik_damping"]) ** 2 * np.eye(6)
            for arm in range(self.num_arms):
                jac = self.kin.jacobian(arm) * self.row_weight[:, None]
                err = self._wrist_error(arm) * self.row_weight
                cols = self.kin.arm_cols[arm] - ARM_START
                self.integral[cols] += float(c["gain"]) * (jac.T @ np.linalg.solve(jac @ jac.T + damp, err))
        self.integral = np.clip(self.integral, -float(c["clamp"]), float(c["clamp"]))

    def _update_freeze(self) -> None:
        """Latch the residual once arrived with both wrists on the command for settle_s; release when the posture
        moves or the arm goal ends."""
        f = self.arm_ik["freeze"]
        arrived, arm_mode = self._arrived(), self.mode == Mode.ARM
        self.frozen = self.frozen and arrived and arm_mode
        on_goal = arm_mode
        for arm in range(self.num_arms):
            err = self._wrist_error(arm)
            on_goal = on_goal and np.linalg.norm(err[:3]) < float(f["pos_tol"]) and np.linalg.norm(err[3:]) < float(f["rot_tol"])
        self.settled_steps = self.settled_steps + 1 if (arrived and on_goal and not self.frozen) else 0
        self.frozen = self.frozen or self.settled_steps >= self.settle_steps

    def corrected_state(self, state: RobotState) -> RobotState:
        """Return the measured state with the IMU calibration applied."""
        quat, gyro = correct_imu(state.quat, state.gyro, self.imu_calibration)
        return replace(state, quat=quat, gyro=gyro)

    def gravity_b(self, state: RobotState) -> np.ndarray:
        """Return gravity (m/s^2) in the pelvis frame at the measured state: the IMU's projected gravity (calibrated)
        carried through imu_frame's pose."""
        _, q_pelvis_imu = self.kin.frame_pose(self.imu_frame)
        return GRAVITY * quat_apply(q_pelvis_imu, projected_gravity(self.corrected_state(state).quat))

    def _gravity_offset(self, state: RobotState, arm_q: np.ndarray) -> np.ndarray | float:
        """Return minus each arm joint's gravity torque at arm_q over its kp (arm_ik.gravity_offset), else 0."""
        if not self.arm_ik.get("gravity_offset"):
            return 0.0
        q = self.default.copy()        # training's ArmKinematics holds the torso at its default angle
        q[ARM_START:] = arm_q
        tau = self.kin.gravity_torques(q, self.gravity_b(state))[ARM_START:]
        return -tau / self.kp[ARM_START:]

    # -- observations and bounds -----------------------------------------------

    def _agent_terms(self, trained, terms: list[str]) -> list[str]:
        """Return the terms an actor observes: trained (an export's list), checked to be terms, in order."""
        full = [t for t in terms if t != "gait_phase_cos" or self.gait_phase_cos]
        trained = list(trained)
        unknown = [t for t in trained if t not in full]
        order = [full.index(t) for t in trained if t in full]
        if unknown or order != sorted(set(order)):
            raise ValueError(f"obs terms {trained} are not observations this controller builds in its order {full}")
        return trained

    def _position_bounds(self, hard: np.ndarray, margin: float, target_limits: dict | None):
        """Return (lower, upper) per joint (rad): the hard limits widened by margin, narrowed to target_limits."""
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
        """Clip the joints' targets to the position bounds."""
        return np.minimum(np.maximum(target, self.lower[joints]), self.upper[joints])

    def _torque_bound(self, target: np.ndarray, state: RobotState, joints: slice) -> np.ndarray:
        """Clip the joints' targets so the PD torque they ask for stays within torque_headroom x effort_limits."""
        if self.torque_headroom is None:
            return target
        q, dq = state.q[joints], state.dq[joints]
        kp, kd = self.kp[joints], self.kd[joints]
        tau = self.torque_headroom * self.effort_limits[joints]
        return np.minimum(np.maximum(target, q + (kd * dq - tau) / kp), q + (kd * dq + tau) / kp)

    def velocity_command(self) -> np.ndarray:
        """Return the operator's velocity command, live in navigation and through an arm goal."""
        return self.cmd_vel.copy()

    def observed_command(self) -> np.ndarray:
        """Return the command the policies see and the gait clock reads: velocity_command low-passed (command_filter_tau)."""
        return self.cmd_filtered.copy() if self.cmd_alpha is not None else self.velocity_command()

    def _filter_command(self) -> None:
        """Step the command low-pass once toward velocity_command."""
        if self.cmd_alpha is not None:
            self.cmd_filtered += self.cmd_alpha * (self.velocity_command() - self.cmd_filtered)

    @staticmethod
    def _scales(per_joint, scale, n: int) -> np.ndarray:
        """Return (n,) action scales: per_joint when given, else scale for every joint."""
        out = np.full(n, float(scale)) if per_joint is None else np.asarray(per_joint, dtype=np.float64)
        if out.shape != (n,):
            raise ValueError(f"{len(out)} action scales for {n} joints")
        return out

    def _gait_phase(self, cmd: np.ndarray) -> np.ndarray:
        """Return the sin of each leg's gait phase, then with gait_phase_cos the cos; phase 0 while standing."""
        if np.linalg.norm(cmd) < self.zero_cmd_eps:
            phase = np.zeros(2)
        else:
            p = (self.steps * self.dt % self.gait_period) / self.gait_period
            phase = np.array([p, (p + 0.5) % 1.0])
        angle = 2.0 * np.pi * phase
        return np.concatenate([np.sin(angle), np.cos(angle)]) if self.gait_phase_cos else np.sin(angle)

    def observations(self, state: RobotState) -> dict[str, np.ndarray]:
        """Return each agent's observation vector at the measured state (the IMU calibrated): its terms (obs_terms)
        in the env's order."""
        state = self.corrected_state(state)
        cmd = self.observed_command()
        phase = self._gait_phase(cmd)
        terms = {
            "base_ang_vel": state.gyro,
            "projected_gravity": projected_gravity(state.quat),
            "velocity_commands": cmd,
            "ee_targets": np.concatenate([np.concatenate([p[:3], wxyz_to_xyzw(p[3:])]) for p in self.believed]),
            "joint_pos": state.q - self.default,
            "joint_vel": state.dq,
            "gait_phase": phase[:2],
            "gait_phase_cos": phase[2:],
        }
        legs = {**terms, "actions": self.legs_last}
        arms = {**terms, "actions": self.arms_last, "wrist_poses": self._wrist_poses(),
                "wrist_errors": self._wrist_errors()}
        return {"legs": np.concatenate([legs[t] for t in self.obs_terms["legs"]]),
                "arms": np.concatenate([arms[t] for t in self.obs_terms["arms"]])}

    def _wrist_poses(self) -> np.ndarray:
        """Return each wrist's pelvis-frame pose as position and xyzw quaternion, concatenated."""
        out = []
        for arm in range(self.num_arms):
            pos, quat = self.kin.wrist_pose(arm)
            out.append(np.concatenate([pos, wxyz_to_xyzw(quat)]))
        return np.concatenate(out)

    def _wrist_errors(self) -> np.ndarray:
        """Return each wrist's (position, axis-angle) error to the arm command, concatenated."""
        out = []
        for arm in range(self.num_arms):
            pos, quat = self.kin.wrist_pose(arm)
            pos_err, rot_err = compute_pose_error(pos, quat, self.believed[arm, :3], self.believed[arm, 3:])
            out.append(np.concatenate([pos_err, rot_err]))
        return np.concatenate(out)
