"""Test the ROS-free core: rotations, kinematics, observation layout, actions, SLAM odometry and modes.

Run on a host with pinocchio + torch (e.g. the env_isaaclab conda env):
    python -m pytest test/test_controller.py
The Isaac Lab parity tests also need ISAACLAB_MATH, the path to
IsaacLab/source/isaaclab/isaaclab/utils/math.py; they are skipped without it.
"""

import importlib.util
import os

import numpy as np
import pytest
import yaml

from lmg_deploy import rotations as R
from lmg_deploy.controller import ARMS_OBS_TERMS, LEGS_OBS_TERMS, MarlController, Mode, RobotState
from lmg_deploy.kinematics import ArmKinematics

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "data", "explicit500.yaml")   # legs 95 / arms 123 inputs
URDF = os.environ.get(
    "H12_URDF", os.path.join(HERE, "..", "..", "..", "..", "CL_Assets", "ros_assets", "h1_2_magpie.urdf"))
RNG = np.random.default_rng(0)


def _cfg():
    """Return the test deploy config."""
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _kin(cfg):
    """Return the H1-2 kinematics for cfg's joints, wrists and SLAM frame."""
    names = cfg["joint_names"]
    return ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                         extra_frames=[cfg["slam_frame"]])


def _random_quat():
    """Return a random unit quaternion."""
    q = RNG.normal(size=4)
    return q / np.linalg.norm(q)


def _state(cfg, q=None):
    """Return a still, upright robot state at q (default: the config's default pose)."""
    q = np.array(cfg["default_joint_pos"]) if q is None else q
    return RobotState(q=q, dq=np.zeros(27), quat=np.array([1.0, 0, 0, 0]),
                      gyro=np.zeros(3), acc=np.array([0.0, 0.0, 9.81]))


class ZeroNetwork:
    """Stand in for an actor that always outputs zeros."""

    def __init__(self, num_outputs: int):
        """Output num_outputs zeros."""
        self._out = np.zeros(num_outputs)

    def reset(self) -> None:
        """Do nothing: the stand-in has no state."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return zeros."""
        return self._out.copy()


class ConstNetwork:
    """Stand in for an actor with a constant output, recording its inputs."""

    def __init__(self, out):
        """Output `out` on every call."""
        self.out = np.asarray(out, dtype=np.float64)
        self.inputs = []

    def __call__(self, x):
        """Record x and return the constant output."""
        self.inputs.append(x.copy())
        return self.out.copy()


# -- rotations -----------------------------------------------------------------


def test_matrix_roundtrip():
    """quat_from_matrix inverts the rotation matrix of a quaternion, and quat_apply agrees with it."""
    for _ in range(50):
        q = R.quat_unique(_random_quat())
        w, x, y, z = q
        m = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        np.testing.assert_allclose(R.quat_from_matrix(m), q, atol=1e-9)
        v = RNG.normal(size=3)
        np.testing.assert_allclose(R.quat_apply(q, v), m @ v, atol=1e-9)


def test_rotvec_roundtrip():
    """axis_angle_from_quat inverts quat_from_rotvec, and a zero rotation vector is the identity."""
    for _ in range(50):
        v = RNG.normal(size=3) * 0.5
        np.testing.assert_allclose(R.axis_angle_from_quat(R.quat_from_rotvec(v)), v, atol=1e-9)
    np.testing.assert_allclose(R.quat_from_rotvec(np.zeros(3)), [1, 0, 0, 0])


def _isaaclab_math():
    """Import Isaac Lab's math module from ISAACLAB_MATH, or skip the test."""
    path = os.environ.get("ISAACLAB_MATH")
    if not path:
        pytest.skip("ISAACLAB_MATH not set")
    spec = importlib.util.spec_from_file_location("isaaclab_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matches_isaaclab():
    """The rotation helpers match Isaac Lab's functions of the same name."""
    import torch

    m = _isaaclab_math()
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64).unsqueeze(0)  # noqa: E731
    for _ in range(50):
        qa, qb = _random_quat(), _random_quat()
        pa, pb = RNG.normal(size=3), RNG.normal(size=3)
        pos, quat = m.subtract_frame_transforms(t(pa), t(qa), t(pb), t(qb))
        mine = R.subtract_frame_transforms(pa, qa, pb, qb)
        np.testing.assert_allclose(mine[0], pos[0].numpy(), atol=1e-9)
        np.testing.assert_allclose(mine[1], quat[0].numpy(), atol=1e-9)
        perr, rerr = m.compute_pose_error(t(pa), t(qa), t(pb), t(qb), rot_error_type="axis_angle")
        mine = R.compute_pose_error(pa, qa, pb, qb)
        np.testing.assert_allclose(mine[0], perr[0].numpy(), atol=1e-9)
        np.testing.assert_allclose(mine[1], rerr[0].numpy(), atol=1e-9)
        g = m.quat_apply_inverse(t(qa), t([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(R.projected_gravity(qa), g[0].numpy(), atol=1e-9)
        v = RNG.normal(size=3) * 0.3
        qv = m.quat_from_angle_axis(torch.tensor([np.linalg.norm(v)]), t(v))
        np.testing.assert_allclose(R.quat_from_rotvec(v), qv[0].numpy(), atol=1e-9)


# -- kinematics -------------------------------------------------------------------


def test_jacobian_matches_finite_differences():
    """The wrist Jacobian's linear rows are the COM point's (as PhysX reports them), its angular rows the link's."""
    cfg = _cfg()
    kin = _kin(cfg)
    q = np.array(cfg["default_joint_pos"]) + RNG.uniform(-0.3, 0.3, size=27)
    kin.update(q)

    def com(arm):
        """Return the wrist link's COM in the pelvis frame."""
        pos, quat = kin.wrist_pose(arm)
        return pos + R.quat_apply(quat, kin._jac_points[arm])

    for arm in range(2):
        assert np.linalg.norm(kin._jac_points[arm]) > 0.02    # the URDF inertial origin was read
        jac = kin._jacobians[arm]
        pos0, quat0 = kin.wrist_pose(arm)
        com0 = com(arm)
        eps = 1e-6
        for k, col in enumerate(kin.arm_cols[arm]):
            dq = q.copy()
            dq[col] += eps
            kin.update(dq)
            pos1, quat1 = kin.wrist_pose(arm)
            np.testing.assert_allclose((com(arm) - com0) / eps, jac[:3, k], atol=1e-4)
            dtheta = R.axis_angle_from_quat(R.quat_mul(quat1, R.quat_conjugate(quat0))) / eps
            np.testing.assert_allclose(dtheta, jac[3:, k], atol=1e-4)
        kin.update(q)


def test_ik_steps_converge():
    """Repeated IK steps bring both wrists onto reachable goals."""
    cfg = _cfg()
    kin = _kin(cfg)
    default = np.array(cfg["default_joint_pos"])
    goal_q = default.copy()
    goal_q[13:27] = RNG.uniform(-0.4, 0.4, size=14)
    goal_q[16] = goal_q[23] = 0.8   # elbows bent, away from the straight-arm singularity
    kin.update(goal_q)
    targets = [kin.wrist_pose(arm) for arm in range(2)]
    q = default.copy()
    for _ in range(60):
        kin.update(q)
        for arm in range(2):
            q[kin.arm_cols[arm]] += kin.ik_step(arm, *targets[arm], damping=0.05, max_step=0.1)
    kin.update(q)
    for arm in range(2):
        pos_err, rot_err = R.compute_pose_error(*kin.wrist_pose(arm), *targets[arm])
        assert np.linalg.norm(pos_err) < 5e-3
        assert np.linalg.norm(rot_err) < 2e-2


def test_ik_step_capped():
    """No joint's IK step exceeds max_step."""
    cfg = _cfg()
    kin = _kin(cfg)
    kin.update(np.array(cfg["default_joint_pos"]))
    step = kin.ik_step(0, np.array([0.5, 0.5, 0.5]), np.array([0.0, 1.0, 0.0, 0.0]), 0.05, 0.1)
    assert np.all(np.abs(step) <= 0.1 + 1e-12)


# -- controller --------------------------------------------------------------------


def _controller(legs=None, arms=None, **overrides):
    """Return a controller on the test config with the overrides, and the config; zero actors by default."""
    cfg = _cfg()
    cfg.update(overrides)
    return MarlController(cfg, _kin(cfg), legs or ZeroNetwork(12), arms or ZeroNetwork(14)), cfg


def test_observation_sizes_and_layout():
    """The legs see 95 inputs and the arms 123, laid out as the training env's."""
    legs, arms = ConstNetwork(np.zeros(12)), ConstNetwork(np.zeros(14))
    ctrl, cfg = _controller(legs=legs, arms=arms)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.set_velocity_command(np.array([0.5, 0.0, 0.0]))
    ctrl.step(state)
    lo, ao = legs.inputs[-1], arms.inputs[-1]
    assert lo.shape == (95,) and ao.shape == (123,)
    np.testing.assert_allclose(lo[:83], ao[:83])                # the shared ProprioCfg terms
    np.testing.assert_allclose(lo[0:3], [0, 0, 9.81])           # accelerometer
    np.testing.assert_allclose(lo[6:9], [0, 0, -1])             # projected gravity
    np.testing.assert_allclose(lo[9:12], [0.5, 0, 0])           # velocity command
    assert lo[12] == 0.0                                        # arm_goal
    rest = ctrl.rest_pose
    np.testing.assert_allclose(lo[13:16], rest[0, :3])          # left target position
    np.testing.assert_allclose(lo[16:20], R.wxyz_to_xyzw(rest[0, 3:]))  # xyzw
    np.testing.assert_allclose(lo[27:54], 0.0)                  # q - default at default
    np.testing.assert_allclose(lo[81:83], [0.0, 0.0], atol=1e-12)  # t = 0: sin(0), sin(pi)
    np.testing.assert_allclose(ao[97:111].reshape(2, 7)[:, :3], rest[:, :3], atol=1e-9)  # wrist poses = rest
    np.testing.assert_allclose(ao[111:123], 0.0, atol=1e-9)     # no error at rest


def test_gait_phase_zero_when_standing():
    """With no velocity command both legs' gait phase is 0."""
    legs = ConstNetwork(np.zeros(12))
    ctrl, cfg = _controller(legs=legs)
    ctrl.reset(_state(cfg))
    ctrl.step(_state(cfg))
    np.testing.assert_allclose(legs.inputs[-1][81:83], 0.0)


def test_gait_phase_cos():
    """gait_phase_cos adds each leg's phase cos after its sin (legs 97 / arms 125), phase 0 while standing.

    As legs_rewards.leg_phase: sin 0, cos 1 while standing.
    """
    legs, arms = ConstNetwork(np.zeros(12)), ConstNetwork(np.zeros(14))
    ctrl, cfg = _controller(legs=legs, arms=arms, gait_phase_cos=True)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.step(state)                                            # standing
    lo, ao = legs.inputs[-1], arms.inputs[-1]
    assert lo.shape == (97,) and ao.shape == (125,)
    np.testing.assert_allclose(lo[81:85], [0.0, 0.0, 1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(lo[:85], ao[:85])                # the shared ProprioCfg terms
    ctrl.set_velocity_command(np.array([0.5, 0.0, 0.0]))
    for _ in range(10):                                         # the 11th step sees t = 0.2 s: a quarter period
        ctrl.step(state)
    phase = 2.0 * np.pi * np.array([0.25, 0.75])
    np.testing.assert_allclose(legs.inputs[-1][81:85], np.concatenate([np.sin(phase), np.cos(phase)]), atol=1e-9)


def test_legs_action_mapping_and_last_action():
    """Leg actions are clipped, scaled, bounded by position and torque, and observed as applied."""
    action = np.full(12, 20.0)                  # beyond the clip
    legs = ConstNetwork(action)
    ctrl, cfg = _controller(legs=legs)
    state = _state(cfg)
    ctrl.reset(state)
    np.testing.assert_allclose(ctrl.legs_last, 0.0)
    cmd = ctrl.step(state)
    default = np.array(cfg["default_joint_pos"])
    # The position bound, then the torque bound: at rest kp (target - q) <= headroom x effort.
    torque_upper = default[:12] + cfg["torque_headroom"] * np.array(cfg["effort_limits"][:12]) / np.array(cfg["kp"][:12])
    expected = np.minimum(np.minimum(default[:12] + 10.0 * 0.25, ctrl.upper[:12]), torque_upper)
    np.testing.assert_allclose(cmd.q[:12], expected)
    np.testing.assert_allclose(ctrl.legs_last, (expected - default[:12]) / 0.25)
    assert cmd.q[12] == 0.0


def test_position_bounds_inside_hard_limits_and_target_limits():
    """Targets stay target_margin inside the URDF limits, narrowed to target_limits where it names a joint.

    target_limits is export.yaml's GOLEM position e-stop range; the margin is negative.
    """
    ctrl, cfg = _controller(target_limits={"left_hip_yaw_joint": [-0.1, 0.05], "left_elbow_joint": [-0.2, 0.3]})
    hard = _kin(cfg).joint_limits
    np.testing.assert_allclose(ctrl.upper[1], hard[1, 1] - 0.001)
    np.testing.assert_allclose(ctrl.lower[1], hard[1, 0] + 0.001)
    np.testing.assert_allclose([ctrl.lower[0], ctrl.upper[0]], [-0.1, 0.05])
    elbow = cfg["joint_names"].index("left_elbow_joint")
    np.testing.assert_allclose([ctrl.lower[elbow], ctrl.upper[elbow]], [-0.2, 0.3])
    legs = ConstNetwork(np.full(12, 1.0))       # hip yaw 0.25 rad, past the 0.05 limit
    ctrl, cfg = _controller(legs=legs, target_limits={"left_hip_yaw_joint": [-0.1, 0.05]})
    state = _state(cfg)
    ctrl.reset(state)
    assert ctrl.step(state).q[0] == pytest.approx(0.05)


def test_example_config_builds_a_controller():
    """config/example.yaml is a config the controller accepts, with gait_phase_cos's 97 / 125 inputs."""
    with open(os.path.join(HERE, "..", "config", "example.yaml")) as f:
        cfg = yaml.safe_load(f)
    legs, arms = ConstNetwork(np.zeros(12)), ConstNetwork(np.zeros(14))
    ctrl = MarlController(cfg, _kin(cfg), legs, arms)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.step(state)
    assert legs.inputs[-1].shape == (cfg["legs"]["num_obs"],) and arms.inputs[-1].shape == (cfg["arms"]["num_obs"],)


def test_target_limits_must_name_policy_joints():
    """A target_limits entry for an unknown joint is refused."""
    with pytest.raises(ValueError):
        _controller(target_limits={"not_a_joint": [-1.0, 1.0]})


def test_arm_goal_modes_and_settle():
    """An arm goal settles at zero velocity before arm mode, and ending it settles back to navigation."""
    ctrl, cfg = _controller(settle_to_arm_s=0.75, settle_to_nav_s=1.5)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.set_velocity_command(np.array([0.5, 0.0, 0.0]))
    goal = ctrl.rest_pose.copy()
    goal[:, 2] -= 0.1
    ctrl.request_arm_goal(goal)
    assert ctrl.mode == Mode.SETTLE_TO_ARM
    np.testing.assert_allclose(ctrl.velocity_command(), 0.0)
    for _ in range(int(round(0.75 / 0.02))):
        ctrl.step(state)
    assert ctrl.mode == Mode.ARM
    np.testing.assert_allclose(ctrl.believed, goal)
    ctrl.end_arm_goal()
    assert ctrl.mode == Mode.SETTLE_TO_NAV
    for _ in range(int(round(1.5 / 0.02))):
        ctrl.step(state)
    assert ctrl.mode == Mode.NAV
    np.testing.assert_allclose(ctrl.velocity_command(), [0.5, 0, 0])


def test_arms_track_goal_with_pure_ik():
    """With zero residual, stepping the joints to the IK targets walks the wrists onto the goal."""
    ctrl, cfg = _controller(settle_to_arm_s=0.0)
    q = np.array(cfg["default_joint_pos"])
    ctrl.reset(_state(cfg, q))
    goal = ctrl.rest_pose.copy()
    goal[0, :3] += [0.05, 0.05, -0.1]
    goal[1, :3] += [0.05, -0.05, -0.1]
    ctrl.request_arm_goal(goal)
    for _ in range(40):
        cmd = ctrl.step(_state(cfg, q))
        q[13:] = cmd.q[13:]                     # perfect tracking
    assert np.all(ctrl.wrist_errors() < 5e-3)


def _slam_body_pose(ctrl, state, pelvis_pos, pelvis_quat):
    """Return the body pose FAST-LIO would report for a pelvis at this world pose, at the state's joints."""
    ctrl.kin.update(state.q)
    p_pb, q_pb = ctrl.kin.frame_pose(ctrl.slam_frame)
    return pelvis_pos + R.quat_apply(pelvis_quat, p_pb), R.quat_mul(pelvis_quat, q_pb)


def test_slam_odometry_moves_goal_against_pelvis_motion():
    """When the pelvis walks forward and turns, the goal moves back and turns the other way in its frame."""
    ctrl, cfg = _controller()
    state = _state(cfg)
    ctrl.reset(state)
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    yaw = 0.2
    for pelvis_pos, pelvis_quat in ((np.zeros(3), np.array([1.0, 0, 0, 0])),
                                    (np.array([0.1, 0.0, 0.0]), R.quat_from_rotvec(np.array([0.0, 0.0, yaw])))):
        ctrl.set_slam_pose(*_slam_body_pose(ctrl, state, pelvis_pos, pelvis_quat))
        ctrl.step(state)
    world = [(np.zeros(3) + p[:3], p[3:]) for p in goal]         # the first pelvis pose is the world frame
    for arm in range(2):
        pos, quat = R.subtract_frame_transforms(np.array([0.1, 0.0, 0.0]),
                                                R.quat_from_rotvec(np.array([0.0, 0.0, yaw])), *world[arm])
        np.testing.assert_allclose(ctrl.believed[arm, :3], pos, atol=1e-9)
        np.testing.assert_allclose(R.quat_unique(ctrl.believed[arm, 3:]), R.quat_unique(quat), atol=1e-9)


def test_slam_odometry_follows_the_torso_joint():
    """A torso turn moves the lidar but not the pelvis, so the body pose it reports leaves the goal in place."""
    ctrl, cfg = _controller()
    state = _state(cfg)
    ctrl.reset(state)
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    ctrl.set_slam_pose(*_slam_body_pose(ctrl, state, np.zeros(3), np.array([1.0, 0, 0, 0])))
    ctrl.step(state)
    turned = _state(cfg)
    turned.q[12] = 0.4                                            # torso_joint, rad
    ctrl.set_slam_pose(*_slam_body_pose(ctrl, turned, np.zeros(3), np.array([1.0, 0, 0, 0])))
    ctrl.step(turned)
    np.testing.assert_allclose(ctrl.believed, goal, atol=1e-9)


def test_slam_odometry_moves_goal_only_on_a_new_pose():
    """One SLAM pose moves the goal once, however many policy steps follow it."""
    ctrl, cfg = _controller()
    state = _state(cfg)
    ctrl.reset(state)
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    ctrl.set_slam_pose(*_slam_body_pose(ctrl, state, np.zeros(3), np.array([1.0, 0, 0, 0])))
    ctrl.step(state)
    ctrl.set_slam_pose(*_slam_body_pose(ctrl, state, np.array([0.05, 0.0, 0.0]), np.array([1.0, 0, 0, 0])))
    for _ in range(5):                                            # one SLAM pose, five policy steps
        ctrl.step(state)
    np.testing.assert_allclose(ctrl.believed[:, 0], goal[:, 0] - 0.05, atol=1e-9)
    np.testing.assert_allclose(ctrl.believed[:, 1:3], goal[:, 1:3], atol=1e-9)


# -- action terms ------------------------------------------------------------------------


def test_residual_low_pass_and_observed_action():
    """The arms' residual is low-passed, and the arms observe the filtered residual as their last action."""
    action = np.full(14, 1.0)
    arms = ConstNetwork(action)
    ctrl, cfg = _controller(arms=arms)
    state = _state(cfg)
    ctrl.reset(state)
    alpha = 1.0 - np.exp(-2.0 * np.pi * 3.0 * 0.02)
    ctrl.step(state)
    np.testing.assert_allclose(ctrl.residual, alpha * action)
    np.testing.assert_allclose(ctrl.arms_last, alpha * action)        # the arms observe the filtered residual
    ctrl.step(state)
    np.testing.assert_allclose(ctrl.residual, (alpha + alpha * (1 - alpha)) * action)
    np.testing.assert_allclose(arms.inputs[-1][83:97], alpha * action)


def test_targets_inside_the_safety_clip():
    """Saturated actions keep every leg and arm target inside the URDF limits by the margin."""
    ctrl, cfg = _controller(legs=ConstNetwork(np.full(12, 10.0)), arms=ConstNetwork(np.full(14, 10.0)))
    state = _state(cfg)
    ctrl.reset(state)
    for _ in range(20):
        cmd = ctrl.step(state)
    limits = ctrl.kin.joint_limits
    legs_arms = np.r_[0:12, 13:27]
    assert np.all(cmd.q[legs_arms] <= limits[legs_arms, 1] - 0.001 + 1e-12)
    assert np.all(cmd.q[legs_arms] >= limits[legs_arms, 0] + 0.001 - 1e-12)


def test_torque_headroom_bounds_the_pd_demand():
    """Every target's PD torque at the measured state is within torque_headroom x effort_limits."""
    ctrl, cfg = _controller(legs=ConstNetwork(np.full(12, 10.0)), arms=ConstNetwork(np.full(14, -10.0)))
    state = _state(cfg)
    state.dq = RNG.normal(scale=2.0, size=27)
    ctrl.reset(state)
    cmd = ctrl.step(state)
    joints = np.r_[0:12, 13:27]
    demand = ctrl.kp[joints] * (cmd.q[joints] - state.q[joints]) - ctrl.kd[joints] * state.dq[joints]
    assert np.all(np.abs(demand) <= 0.85 * ctrl.effort_limits[joints] + 1e-9)


# -- recurrent actors ----------------------------------------------------------------------

def test_recurrent_network_carries_and_resets_its_state(tmp_path):
    """load_network wraps an LSTM export, which carries its state between calls until reset()."""
    torch = pytest.importorskip("torch")
    from lmg_deploy.networks import RecurrentTorchScriptNetwork, load_network

    class Counter(torch.nn.Module):
        """Output the number of steps since the last reset."""

        hidden_size: int
        num_layers: int

        def __init__(self):
            """Set the recurrent attributes load_network looks for."""
            super().__init__()
            self.hidden_size, self.num_layers = 3, 1

        def forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
            """Count one step in h and output it."""
            h = h + 1.0
            return obs[:, :2] * 0.0 + h[0, :, :1], h, c

    path = str(tmp_path / "counter.pt")
    torch.jit.script(Counter()).save(path)
    net = load_network(path, 5, 2)
    assert isinstance(net, RecurrentTorchScriptNetwork)
    x = np.zeros(5)
    assert [net(x)[0] for _ in range(3)] == [1.0, 2.0, 3.0]
    net.reset()
    assert net(x)[0] == 1.0


def test_controller_reset_resets_recurrent_actors():
    """Resetting the controller resets both actors."""
    class Resettable(ZeroNetwork):
        """Count resets."""

        resets = 0

        def reset(self):
            """Count one reset."""
            self.resets += 1

    legs, arms = Resettable(12), Resettable(14)
    ctrl, cfg = _controller(legs=legs, arms=arms)
    before = legs.resets, arms.resets
    ctrl.reset(_state(cfg))
    assert (legs.resets, arms.resets) == (before[0] + 1, before[1] + 1)


# -- manual mode switch and arm goals (~/set_mode, ~/set_arm_goals) ------------------------

def test_starts_walking_and_switches_by_hand():
    """The controller starts walking; set_mode switches to arm and back, and refuses another mode."""
    ctrl, cfg = _controller(settle_to_arm_s=0.0, settle_to_nav_s=0.0)
    state = _state(cfg)
    ctrl.reset(state)
    assert ctrl.mode == Mode.NAV
    ctrl.set_velocity_command(np.array([0.4, 0.0, 0.2]))
    np.testing.assert_allclose(ctrl.velocity_command(), [0.4, 0.0, 0.2])
    ctrl.set_mode("arm")                               # no goals set: the arms hold the rest pose
    assert ctrl.mode == Mode.ARM
    np.testing.assert_allclose(ctrl.velocity_command(), 0.0)
    np.testing.assert_allclose(ctrl.believed, ctrl.rest_pose)
    ctrl.set_mode("walk")
    assert ctrl.mode == Mode.NAV
    np.testing.assert_allclose(ctrl.velocity_command(), [0.4, 0.0, 0.2])
    with pytest.raises(ValueError):
        ctrl.set_mode("run")


def test_arm_goals_wait_for_arm_mode_and_apply_at_once_in_it():
    """Arm goals wait for arm mode, apply at once in it, and survive a reset."""
    ctrl, cfg = _controller(settle_to_arm_s=0.0, settle_to_nav_s=0.0)
    ctrl.reset(_state(cfg))
    goal = ctrl.rest_pose.copy()
    goal[:, 2] -= 0.3                                  # both wrists 30 cm lower: a crouch
    ctrl.set_arm_goals(goal)
    assert ctrl.mode == Mode.NAV                       # setting goals doesn't switch the mode
    np.testing.assert_allclose(ctrl.believed, ctrl.rest_pose)
    ctrl.set_mode("arm")
    np.testing.assert_allclose(ctrl.believed, goal)
    other = goal.copy()
    other[0, 1] += 0.1
    ctrl.set_arm_goals(other)                          # in arm mode: at once
    np.testing.assert_allclose(ctrl.believed, other)
    ctrl.set_mode("walk")
    np.testing.assert_allclose(ctrl.believed, ctrl.rest_pose)
    ctrl.reset(_state(cfg))                            # the goals stay set across a reset (the band release)
    ctrl.set_mode("arm")
    np.testing.assert_allclose(ctrl.believed, other)


def test_set_mode_arm_goes_through_the_settle():
    """set_mode arm goes through the settle, and a repeated call does not restart it."""
    ctrl, cfg = _controller(settle_to_arm_s=0.1, settle_to_nav_s=0.0)
    state = _state(cfg)
    ctrl.reset(state)
    goal = ctrl.rest_pose.copy()
    goal[:, 0] += 0.05
    ctrl.set_arm_goals(goal)
    ctrl.set_mode("arm")
    assert ctrl.mode == Mode.SETTLE_TO_ARM
    ctrl.set_mode("arm")                               # already on its way: no restart of the settle
    for _ in range(5):
        ctrl.step(state)
    assert ctrl.mode == Mode.ARM
    np.testing.assert_allclose(ctrl.believed, goal, atol=1e-9)


def test_per_joint_action_scales():
    """Per-joint scales map each action to its target and back to the observed action; a wrong count raises."""
    scales = np.linspace(0.1, 0.3, 12)
    residual = np.linspace(0.05, 0.4, 14)
    legs = ConstNetwork(np.ones(12))
    ctrl, cfg = _controller(legs=legs, arms=ConstNetwork(np.ones(14)), leg_action_scales=scales.tolist(),
                            arm_residual_scales=residual.tolist(), torque_headroom=None, target_limits=None,
                            target_margin=1.0)
    state = _state(cfg)
    ctrl.reset(state)
    cmd = ctrl.step(state)
    np.testing.assert_allclose(cmd.q[:12], np.array(cfg["default_joint_pos"][:12]) + scales)
    np.testing.assert_allclose(ctrl.legs_last, np.ones(12))              # observed back in action units
    with pytest.raises(ValueError):
        _controller(leg_action_scales=[0.25] * 11)


def test_command_filter_low_passes_the_observed_command():
    """With command_filter_tau the policies see the command low-passed, zero during an arm goal and after a reset."""
    legs = ConstNetwork(np.zeros(12))
    ctrl, cfg = _controller(legs=legs, command_filter_tau=0.5)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.set_velocity_command(np.array([0.5, 0.0, 0.0]))
    alpha = 1.0 - np.exp(-ctrl.dt / 0.5)
    ctrl.step(state)
    np.testing.assert_allclose(legs.inputs[-1][9:12], [0.5 * alpha, 0, 0])
    ctrl.step(state)
    np.testing.assert_allclose(legs.inputs[-1][9:12], [0.5 * (1 - (1 - alpha) ** 2), 0, 0])
    ctrl.set_mode("arm")
    while ctrl.mode != Mode.ARM:
        ctrl.step(state)
    ctrl.step(state)
    np.testing.assert_allclose(legs.inputs[-1][9:12], 0.0)               # zero during an arm goal
    ctrl.reset(state)
    np.testing.assert_allclose(ctrl.observed_command(), 0.0)             # and from zero after a reset


def test_obs_terms_must_match_the_layout():
    """An export's observation term lists must be the controller's own; matching lists and none both build."""
    legs = [t for t in LEGS_OBS_TERMS if t != "gait_phase_cos"]   # explicit500 has no gait_phase_cos
    arms = [t for t in ARMS_OBS_TERMS if t != "gait_phase_cos"]
    _controller(legs_obs_terms=legs, arms_obs_terms=arms)
    with pytest.raises(ValueError):
        _controller(legs_obs_terms=legs[:-1] + ["height_scan", "actions"], arms_obs_terms=arms)
    with pytest.raises(ValueError):
        _controller(legs_obs_terms=legs, arms_obs_terms=LEGS_OBS_TERMS)


def test_arm_goals_refused_for_a_policy_that_never_trained_one():
    """With arm_goals false, set_mode arm and request_arm_goal raise and the controller keeps navigating."""
    ctrl, cfg = _controller(arm_goals=False)
    ctrl.reset(_state(cfg))
    with pytest.raises(ValueError):
        ctrl.set_mode("arm")
    with pytest.raises(ValueError):
        ctrl.request_arm_goal(ctrl.rest_pose)
    assert ctrl.mode == Mode.NAV
    ctrl.set_mode("walk")                       # walking stays allowed
