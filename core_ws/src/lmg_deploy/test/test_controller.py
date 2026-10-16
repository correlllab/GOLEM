"""Unit tests for the ROS-free core: rotations, kinematics, observation layout, odometry.

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
from lmg_deploy.controller import MarlController, Mode, RobotState
from lmg_deploy.kinematics import ArmKinematics
from lmg_deploy.networks import ZeroNetwork

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "..", "policies", "marl_ik", "marl_ik.yaml")
URDF = os.environ.get(
    "H12_URDF", os.path.join(HERE, "..", "..", "..", "..", "CL_Assets", "ros_assets", "h1_2_magpie.urdf"))
RNG = np.random.default_rng(0)


def _cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _kin(cfg):
    names = cfg["joint_names"]
    return ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]])


def _random_quat():
    q = RNG.normal(size=4)
    return q / np.linalg.norm(q)


def _state(cfg, q=None):
    q = np.array(cfg["default_joint_pos"]) if q is None else q
    return RobotState(q=q, dq=np.zeros(27), tau=np.zeros(27), quat=np.array([1.0, 0, 0, 0]),
                      gyro=np.zeros(3), acc=np.array([0.0, 0.0, 9.81]))


class ConstNetwork:
    def __init__(self, out):
        self.out = np.asarray(out, dtype=np.float64)
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.copy())
        return self.out.copy()


# -- rotations -----------------------------------------------------------------


def test_matrix_roundtrip():
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
    for _ in range(50):
        v = RNG.normal(size=3) * 0.5
        np.testing.assert_allclose(R.axis_angle_from_quat(R.quat_from_rotvec(v)), v, atol=1e-9)
    np.testing.assert_allclose(R.quat_from_rotvec(np.zeros(3)), [1, 0, 0, 0])


def _isaaclab_math():
    path = os.environ.get("ISAACLAB_MATH")
    if not path:
        pytest.skip("ISAACLAB_MATH not set")
    spec = importlib.util.spec_from_file_location("isaaclab_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matches_isaaclab():
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
    """Linear rows: the wrist link's COM point (as PhysX reports them); angular rows: the link."""
    cfg = _cfg()
    kin = _kin(cfg)
    q = np.array(cfg["default_joint_pos"]) + RNG.uniform(-0.3, 0.3, size=27)
    kin.update(q)

    def com(arm):
        pos, quat = kin.wrist_pose(arm)
        return pos + R.quat_apply(quat, kin._jac_points[arm])

    for arm in range(2):
        assert np.linalg.norm(kin._jac_points[arm]) > 0.02    # read from the URDF inertial origin
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
    cfg = _cfg()
    kin = _kin(cfg)
    kin.update(np.array(cfg["default_joint_pos"]))
    step = kin.ik_step(0, np.array([0.5, 0.5, 0.5]), np.array([0.0, 1.0, 0.0, 0.0]), 0.05, 0.1)
    assert np.all(np.abs(step) <= 0.1 + 1e-12)


# -- controller --------------------------------------------------------------------


def _controller(legs=None, arms=None, estimator=None, **overrides):
    cfg = _cfg()
    cfg.update(overrides)
    return MarlController(cfg, _kin(cfg), legs or ZeroNetwork(12), arms or ZeroNetwork(14),
                          estimator or ZeroNetwork(6)), cfg


def test_observation_sizes_and_layout():
    legs, arms = ConstNetwork(np.zeros(12)), ConstNetwork(np.zeros(14))
    ctrl, cfg = _controller(legs=legs, arms=arms)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.set_velocity_command(np.array([0.5, 0.0, 0.0]))
    ctrl.step(state)
    lo, ao = legs.inputs[-1], arms.inputs[-1]
    assert lo.shape == (95,) and ao.shape == (123,)             # height_drop_obs: false
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
    legs = ConstNetwork(np.zeros(12))
    ctrl, cfg = _controller(legs=legs)
    ctrl.reset(_state(cfg))
    ctrl.step(_state(cfg))
    np.testing.assert_allclose(legs.inputs[-1][81:83], 0.0)


def test_legs_action_mapping_and_last_action():
    action = np.full(12, 20.0)                  # beyond the clip
    legs = ConstNetwork(action)
    ctrl, cfg = _controller(legs=legs)
    state = _state(cfg)
    ctrl.reset(state)
    np.testing.assert_allclose(ctrl.legs_last, 0.0)
    cmd = ctrl.step(state)
    default = np.array(cfg["default_joint_pos"])
    # the position bound, then the torque bound: at rest kp (target - q) <= headroom x effort
    torque_upper = default[:12] + cfg["torque_headroom"] * np.array(cfg["effort_limits"][:12]) / np.array(cfg["kp"][:12])
    expected = np.minimum(np.minimum(default[:12] + 10.0 * 0.25, ctrl.upper[:12]), torque_upper)
    np.testing.assert_allclose(cmd.q[:12], expected)
    np.testing.assert_allclose(ctrl.legs_last, (expected - default[:12]) / 0.25)
    assert cmd.q[12] == 0.0


def test_position_bounds_inside_hard_limits_and_target_limits():
    """Targets stay target_margin inside the URDF limits (negative margin), narrowed to export.yaml's
    target_limits (GOLEM's position e-stop range) where it names a joint."""
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


def test_target_limits_must_name_policy_joints():
    with pytest.raises(ValueError):
        _controller(target_limits={"not_a_joint": [-1.0, 1.0]})


def test_arm_goal_modes_and_settle():
    ctrl, cfg = _controller()
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
    """Zero residual: stepping the joints to the targets the IK gives walks the wrists onto the goal."""
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


def test_odometry_moves_goal_against_pelvis_motion():
    vx = 0.5                                     # m/s forward, estimated
    ctrl, cfg = _controller(estimator=ConstNetwork([vx, 0, 0, 0, 0, 0]), settle_to_arm_s=0.0)
    state = _state(cfg)
    ctrl.reset(state)
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    n = 10
    for _ in range(n + 1):                       # the first step has no previous action: no estimate
        ctrl.step(state)
    np.testing.assert_allclose(ctrl.believed[:, 0], goal[:, 0] - vx * 0.02 * n, atol=1e-9)
    np.testing.assert_allclose(ctrl.believed[:, 1:3], goal[:, 1:3], atol=1e-9)


def test_estimator_input_layout():
    est = ConstNetwork(np.zeros(6))
    ctrl, cfg = _controller(estimator=est)
    states = []
    for k in range(6):
        s = _state(cfg)
        s.acc = np.array([k, 0.0, 0.0])
        s.tau = np.full(27, float(k))
        states.append(s)
    ctrl.reset(states[0])
    for s in states:
        ctrl.step(s)
    x = est.inputs[-1]                           # after the 6th frame: window frames 2..5
    assert x.shape == (326,)
    np.testing.assert_allclose(x[0:12:3], [2, 3, 4, 5])          # acc x, oldest first, term-major
    np.testing.assert_allclose(x[252:300].reshape(4, 12)[:, 0], [2, 3, 4, 5])  # leg torques
    first = est.inputs[0]                        # the 2nd step: window = frame 0 repeated, then frame 1
    np.testing.assert_allclose(first[0:12:3], [0, 0, 0, 1])


# -- the Golem policy's deploy features ------------------------------------------------

GOLEM_CFG_PATH = os.path.join(HERE, "..", "policies", "marl_golem", "marl_golem.yaml")


def _golem_controller(legs=None, arms=None, **overrides):
    with open(GOLEM_CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    names = cfg["joint_names"]
    odom = cfg["legs_odometry"]
    kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                        extra_frames=odom["foot_frames"] + [odom["imu_frame"]])
    return MarlController(cfg, kin, legs or ZeroNetwork(12), arms or ZeroNetwork(14), ZeroNetwork(6)), cfg


def test_residual_low_pass_and_observed_action():
    action = np.full(14, 1.0)
    arms = ConstNetwork(action)
    ctrl, cfg = _golem_controller(arms=arms)
    state = _state(cfg)
    ctrl.reset(state)
    alpha = 1.0 - np.exp(-2.0 * np.pi * 3.0 * 0.02)
    ctrl.step(state)
    np.testing.assert_allclose(ctrl.residual, alpha * action)
    np.testing.assert_allclose(ctrl.arms_last, alpha * action)        # the arms observe the filtered residual
    ctrl.step(state)
    np.testing.assert_allclose(ctrl.residual, (alpha + alpha * (1 - alpha)) * action)
    np.testing.assert_allclose(arms.inputs[-1][84:98], alpha * action)


def test_golem_observes_height_drop():
    legs, arms = ConstNetwork(np.zeros(12)), ConstNetwork(np.zeros(14))
    ctrl, cfg = _golem_controller(legs=legs, arms=arms)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.step(state)
    lo, ao = legs.inputs[-1], arms.inputs[-1]
    assert lo.shape == (96,) and ao.shape == (124,)
    assert lo[27] == 0.0                                        # height_drop
    np.testing.assert_allclose(lo[28:55], 0.0)                  # q - default at default


def test_golem_targets_inside_the_safety_clip():
    ctrl, cfg = _golem_controller(legs=ConstNetwork(np.full(12, 10.0)), arms=ConstNetwork(np.full(14, 10.0)))
    state = _state(cfg)
    ctrl.reset(state)
    for _ in range(20):
        cmd = ctrl.step(state)
    limits = ctrl.kin.joint_limits
    legs_arms = np.r_[0:12, 13:27]
    assert np.all(cmd.q[legs_arms] <= limits[legs_arms, 1] - 0.001 + 1e-12)
    assert np.all(cmd.q[legs_arms] >= limits[legs_arms, 0] + 0.001 - 1e-12)


def test_leg_odometry_moves_goal_against_pelvis_motion():
    """Feet fixed in the world while the pelvis walks forward: they slide back in the pelvis frame."""
    ctrl, cfg = _golem_controller(settle_to_arm_s=0.0)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.kin.update(state.q)
    feet0 = {f: ctrl.kin.frame_pose(f) for f in ctrl.foot_frames}
    shift = {"x": 0.0}
    real_update = ctrl.kin.update

    def update(q):
        real_update(q)
        for f, (pos, quat) in feet0.items():
            ctrl.kin._extra_poses[f] = (pos - np.array([shift["x"], 0.0, 0.0]), quat)

    ctrl.kin.update = update
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    v, n = 0.5, 10
    ctrl.step(state)                             # the first step only records the feet
    for _ in range(n):
        shift["x"] += v * 0.02
        ctrl.step(state)
    np.testing.assert_allclose(ctrl.believed[:, 0], goal[:, 0] - v * 0.02 * n, atol=1e-9)
    np.testing.assert_allclose(ctrl.believed[:, 1:3], goal[:, 1:3], atol=1e-9)


def test_leg_odometry_ignores_a_lifted_foot():
    ctrl, cfg = _golem_controller(settle_to_arm_s=0.0)
    state = _state(cfg)
    ctrl.reset(state)
    ctrl.kin.update(state.q)
    left, right = ctrl.foot_frames
    feet0 = {f: ctrl.kin.frame_pose(f) for f in ctrl.foot_frames}
    t = {"k": 0}
    real_update = ctrl.kin.update

    def update(q):
        real_update(q)
        pos, quat = feet0[left]
        ctrl.kin._extra_poses[left] = (pos, quat)                                  # planted
        pos, quat = feet0[right]
        ctrl.kin._extra_poses[right] = (pos + np.array([0.02 * t["k"], 0.0, 0.1]), quat)  # swinging, lifted

    ctrl.kin.update = update
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    for k in range(6):
        t["k"] = k
        ctrl.step(state)
    np.testing.assert_allclose(ctrl.believed[:, :3], goal[:, :3], atol=1e-9)    # the planted foot didn't move


def test_torque_headroom_bounds_the_pd_demand():
    ctrl, cfg = _golem_controller(legs=ConstNetwork(np.full(12, 10.0)), arms=ConstNetwork(np.full(14, -10.0)))
    state = _state(cfg)
    state.dq = RNG.normal(scale=2.0, size=27)
    ctrl.reset(state)
    cmd = ctrl.step(state)
    joints = np.r_[0:12, 13:27]
    demand = ctrl.kp[joints] * (cmd.q[joints] - state.q[joints]) - ctrl.kd[joints] * state.dq[joints]
    assert np.all(np.abs(demand) <= 0.85 * ctrl.effort_limits[joints] + 1e-9)


# -- Explicit500: recurrent actors, feet and gravity odometry -----------------------------

def test_align_tilt_matches_gravity_and_keeps_yaw():
    for _ in range(20):
        delta, up_prev = _random_quat(), RNG.normal(size=3)
        up_prev /= np.linalg.norm(up_prev)
        up = RNG.normal(size=3)
        up /= np.linalg.norm(up)
        aligned = R.align_tilt(delta, up_prev, up)
        np.testing.assert_allclose(R.quat_apply(aligned, up), up_prev, atol=1e-9)    # carries up onto up_prev
    yaw = R.quat_from_rotvec(np.array([0.0, 0.0, 0.3]))
    z = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(R.align_tilt(yaw, z, z), yaw, atol=1e-12)             # a level turn stays


def _feet_controller(estimator, **odometry):
    cfg = _cfg()
    cfg.update(settle_to_arm_s=0.0, odometry={"mode": "learned", "history_frames": 4, **odometry},
               legs_odometry={"foot_frames": ["left_ankle_roll_link", "right_ankle_roll_link"], "imu_frame": "torso_link"})
    names = cfg["joint_names"]
    kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                        extra_frames=["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"])
    return MarlController(cfg, kin, ZeroNetwork(12), ZeroNetwork(14), estimator), cfg


def test_vertical_from_feet_replaces_the_estimated_rise():
    vz = 0.5                                     # m/s up, estimated; the feet say the pelvis stays at its height
    for vertical, expected_dz in ((False, -vz * 0.02 * 10), (True, 0.0)):
        ctrl, cfg = _feet_controller(ConstNetwork([0, 0, vz, 0, 0, 0]), vertical_from_feet=vertical)
        assert ctrl.history == 4 and ctrl.vertical_from_feet == vertical
        state = _state(cfg)
        ctrl.reset(state)
        goal = ctrl.rest_pose.copy()
        ctrl.request_arm_goal(goal)
        for _ in range(11):
            ctrl.step(state)
        np.testing.assert_allclose(ctrl.believed[:, 2], goal[:, 2] + expected_dz, atol=1e-9)


def test_tilt_from_gravity_removes_an_estimated_roll():
    ctrl, cfg = _feet_controller(ConstNetwork([0, 0, 0, 0.5, 0, 0]), tilt_from_gravity=True)  # rad/s roll, estimated
    state = _state(cfg)
    ctrl.reset(state)
    goal = ctrl.rest_pose.copy()
    ctrl.request_arm_goal(goal)
    for _ in range(11):
        ctrl.step(state)
    np.testing.assert_allclose(ctrl.believed, goal, atol=1e-9)   # gravity says the pelvis never tilted


def test_recurrent_network_carries_and_resets_its_state(tmp_path):
    torch = pytest.importorskip("torch")
    from lmg_deploy.networks import RecurrentTorchScriptNetwork, load_network

    class Counter(torch.nn.Module):
        hidden_size: int
        num_layers: int

        def __init__(self):
            super().__init__()
            self.hidden_size, self.num_layers = 3, 1

        def forward(self, obs: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
            h = h + 1.0                                  # counts the steps since the last reset
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
    class Resettable(ZeroNetwork):
        resets = 0

        def reset(self):
            self.resets += 1

    legs, arms = Resettable(12), Resettable(14)
    ctrl, cfg = _controller(legs=legs, arms=arms)
    before = legs.resets, arms.resets
    ctrl.reset(_state(cfg))
    assert (legs.resets, arms.resets) == (before[0] + 1, before[1] + 1)


# -- manual mode switch and arm goals (~/set_mode, ~/set_arm_goals) ------------------------

def test_starts_walking_and_switches_by_hand():
    ctrl, cfg = _controller(settle_to_arm_s=0.0, settle_to_nav_s=0.0)
    state = _state(cfg)
    ctrl.reset(state)
    assert ctrl.mode == Mode.NAV
    ctrl.set_velocity_command(np.array([0.4, 0.0, 0.2]))
    np.testing.assert_allclose(ctrl.velocity_command(), [0.4, 0.0, 0.2])
    ctrl.set_mode("arm")                               # no goals set: the arms hold the rest pose, standing still
    assert ctrl.mode == Mode.ARM
    np.testing.assert_allclose(ctrl.velocity_command(), 0.0)
    np.testing.assert_allclose(ctrl.believed, ctrl.rest_pose)
    ctrl.set_mode("walk")
    assert ctrl.mode == Mode.NAV
    np.testing.assert_allclose(ctrl.velocity_command(), [0.4, 0.0, 0.2])
    with pytest.raises(ValueError):
        ctrl.set_mode("run")


def test_arm_goals_wait_for_arm_mode_and_apply_at_once_in_it():
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
