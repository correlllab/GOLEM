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

from locomotion_game_deploy import rotations as R
from locomotion_game_deploy.controller import MarlController, Mode, RobotState
from locomotion_game_deploy.kinematics import ArmKinematics
from locomotion_game_deploy.networks import ZeroNetwork

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
    assert lo.shape == (96,) and ao.shape == (124,)
    np.testing.assert_allclose(lo[:84], ao[:84])                # the shared ProprioCfg terms
    np.testing.assert_allclose(lo[0:3], [0, 0, 9.81])           # accelerometer
    np.testing.assert_allclose(lo[6:9], [0, 0, -1])             # projected gravity
    np.testing.assert_allclose(lo[9:12], [0.5, 0, 0])           # velocity command
    assert lo[12] == 0.0                                        # arm_goal
    rest = ctrl.rest_pose
    np.testing.assert_allclose(lo[13:16], rest[0, :3])          # left target position
    np.testing.assert_allclose(lo[16:20], R.wxyz_to_xyzw(rest[0, 3:]))  # xyzw
    assert lo[27] == 0.0                                        # height_drop
    np.testing.assert_allclose(lo[28:55], 0.0)                  # q - default at default
    np.testing.assert_allclose(lo[82:84], [0.0, 0.0], atol=1e-12)  # t = 0: sin(0), sin(pi)
    np.testing.assert_allclose(ao[98:112].reshape(2, 7)[:, :3], rest[:, :3], atol=1e-9)  # wrist poses = rest
    np.testing.assert_allclose(ao[112:124], 0.0, atol=1e-9)     # no error at rest


def test_gait_phase_zero_when_standing():
    legs = ConstNetwork(np.zeros(12))
    ctrl, cfg = _controller(legs=legs)
    ctrl.reset(_state(cfg))
    ctrl.step(_state(cfg))
    np.testing.assert_allclose(legs.inputs[-1][82:84], 0.0)


def test_legs_action_mapping_and_last_action():
    action = np.full(12, 20.0)                  # beyond the clip
    legs = ConstNetwork(action)
    ctrl, cfg = _controller(legs=legs)
    state = _state(cfg)
    ctrl.reset(state)
    np.testing.assert_allclose(ctrl.legs_last, 0.0)
    cmd = ctrl.step(state)
    default = np.array(cfg["default_joint_pos"])
    expected = np.minimum(default[:12] + 10.0 * 0.25, ctrl.upper[:12])
    np.testing.assert_allclose(cmd.q[:12], expected)
    np.testing.assert_allclose(ctrl.legs_last, (expected - default[:12]) / 0.25)
    assert cmd.q[12] == 0.0


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
