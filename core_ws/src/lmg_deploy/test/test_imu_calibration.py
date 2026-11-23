"""Test the torso-IMU calibration: the correction, the export contract in the controller, and the estimate.

    python -m pytest test/test_imu_calibration.py
"""

import logging
import os

import numpy as np
import pytest
import yaml

from lmg_deploy import imu_calibration as C
from lmg_deploy import rotations as R
from lmg_deploy.controller import MarlController, RobotState
from lmg_deploy.kinematics import ArmKinematics

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")
CFG_PATH = os.path.join(HERE, "data", "explicit500.yaml")
URDF = os.environ.get(
    "H12_URDF", os.path.join(HERE, "..", "..", "..", "..", "CL_Assets", "ros_assets", "h1_2_magpie.urdf"))
RNG = np.random.default_rng(1)
CAL = {"roll_deg": -0.3, "pitch_deg": -2.4, "yaw_deg": 0.7, "rotate_gyro": False}
LOWSTATE_IMU = {"source": "lowstate", "frame": "imu_link", "offset": [0.0, 0.0, 0.0], "gyro_lowpass_hz": 8.0,
                "acc": "dropped"}
CALIBRATED_IMU = {**LOWSTATE_IMU, "orientation": "calibrated",
                  "residual_deg": {"roll": [-1.0, 1.0], "pitch": [-1.0, 1.0], "yaw": [-1.0, 1.0]}}
RAW_IMU = {**LOWSTATE_IMU, "quat_bias_deg": {"roll": [-0.6, 0.4], "pitch": [-3.6, -2.0], "yaw": [-1.0, 1.0]}}


def _random_attitude(scale=0.3):
    """Return a random attitude near upright."""
    return R.quat_from_rotvec(RNG.normal(0.0, scale, 3))


def _same_rotation(a, b, atol=1e-12):
    """Assert two unit quaternions are the same rotation."""
    np.testing.assert_allclose(R.quat_unique(a), R.quat_unique(b), atol=atol)


# -- the correction -------------------------------------------------------------------


def test_bias_quat_is_isaacs_euler_xyz():
    """R_bias = Rz(yaw) Ry(pitch) Rx(roll), Isaac's quat_from_euler_xyz (training's bias draw)."""
    r, p, y = np.radians([CAL["roll_deg"], CAL["pitch_deg"], CAL["yaw_deg"]])
    half = lambda a, i: np.array([np.cos(a / 2), *(np.sin(a / 2) * np.eye(3)[i])])  # noqa: E731
    _same_rotation(C.bias_quat(CAL), R.quat_mul(R.quat_mul(half(y, 2), half(p, 1)), half(r, 0)))


def test_negative_pitch_bias_reads_nose_up_when_level():
    """The robot's calibration (pitch about -2.4) is a level torso read nose up: gravity toward -x."""
    g = R.projected_gravity(C.bias_quat({"roll_deg": 0.0, "pitch_deg": -2.4}))
    np.testing.assert_allclose(C.tilt_deg(g), [0.0, -2.4], atol=1e-9)
    assert g[0] < 0.0


def test_correction_recovers_the_true_attitude():
    """A true attitude, biased as the IMU reports it (true * R_bias), comes back from correct_imu; the gyro is left
    alone by default (training biases the quaternion only)."""
    for _ in range(50):
        true = _random_attitude()
        gyro = RNG.normal(size=3)
        quat, out_gyro = C.correct_imu(R.quat_mul(true, C.bias_quat(CAL)), gyro, CAL)
        _same_rotation(quat, true)
        np.testing.assert_allclose(R.projected_gravity(quat), R.projected_gravity(true), atol=1e-12)
        np.testing.assert_array_equal(out_gyro, gyro)


def test_no_calibration_returns_the_inputs():
    """correct_imu(..., None) is the identity."""
    q, w = _random_attitude(), RNG.normal(size=3)
    out_q, out_w = C.correct_imu(q, w, None)
    np.testing.assert_array_equal(out_q, q)
    np.testing.assert_array_equal(out_w, w)


def test_rotate_gyro_expresses_a_tilted_imus_rate_in_the_level_frame():
    """A rotating body seen by an IMU mounted tilted by R_bias: rotate_gyro turns the IMU's rate (the tilted frame's
    body rate, from finite differences of its attitude) into the body's own rate."""
    cal = {**CAL, "roll_deg": 4.0, "pitch_deg": -7.0, "yaw_deg": 10.0, "rotate_gyro": True}
    r = C.bias_quat(cal)
    w_body = np.array([0.4, -1.1, 0.7])     # rad/s in the level (torso) frame
    q0, dt = _random_attitude(), 1e-6
    q1 = R.quat_mul(q0, R.quat_from_rotvec(w_body * dt))
    imu0, imu1 = R.quat_mul(q0, r), R.quat_mul(q1, r)    # the tilted IMU's attitude
    w_imu = R.axis_angle_from_quat(R.quat_mul(R.quat_conjugate(imu0), imu1)) / dt   # what its gyro measures
    assert np.linalg.norm(w_imu - w_body) > 0.1           # the tilt matters at this size
    quat, gyro = C.correct_imu(imu0, w_imu, cal)
    np.testing.assert_allclose(gyro, w_body, atol=1e-6)
    _same_rotation(quat, q0)
    # the specific force rotates the same way, and not at all without rotate_gyro
    f_body = np.array([0.3, -0.2, 9.8])
    np.testing.assert_allclose(C.level_vector(R.quat_apply_inverse(r, f_body), cal), f_body, atol=1e-12)
    np.testing.assert_array_equal(C.level_vector(f_body, CAL), f_body)


def test_validate_calibration():
    """Defaults fill in; a missing, unknown, non-finite or huge value is refused."""
    assert C.validate_calibration({"roll_deg": 1, "pitch_deg": -2}) == {
        "roll_deg": 1.0, "pitch_deg": -2.0, "yaw_deg": 0.0, "rotate_gyro": False}
    assert C.validate_calibration(None) is None
    for bad in ({"pitch_deg": -2.0}, {"roll_deg": 0.0, "pitch_deg": float("nan")}, {"roll_deg": 0.0, "pitch_deg": 30.0},
                {"roll_deg": 0.0, "pitch_deg": 0.0, "trim": 1.0}, {"roll_deg": 0.0, "pitch_deg": 0.0, "rotate_gyro": 1}):
        with pytest.raises(ValueError):
            C.validate_calibration(bad)


def test_load_calibration(tmp_path):
    """load_calibration reads the robot's entry; null or absent is None. The shipped file loads for every robot."""
    path = tmp_path / "cal.yaml"
    path.write_text(yaml.safe_dump({"h12": {"roll_deg": 0.1, "pitch_deg": -2.3}, "g1": None}))
    assert C.load_calibration(str(path), "h12")["pitch_deg"] == -2.3
    assert C.load_calibration(str(path), "g1") is None
    assert C.load_calibration(str(path), "other") is None
    shipped = os.path.join(PKG, "config", "imu_calibration.yaml")
    with open(shipped) as f:
        robots = yaml.safe_load(f)
    assert set(robots) >= {"h12", "h12_sim", "g1"}
    for robot in robots:
        C.load_calibration(shipped, robot)


def test_the_sim_entry_removes_no_offset():
    """The shipped h12_sim entry is all zero: a calibrated policy reads the sim's exact torso IMU unchanged."""
    shipped = os.path.join(PKG, "config", "imu_calibration.yaml")
    sim = C.load_calibration(shipped, "h12_sim")
    assert (sim["roll_deg"], sim["pitch_deg"], sim["yaw_deg"], sim["rotate_gyro"]) == (0.0, 0.0, 0.0, False)
    assert C.load_calibration(shipped, "h12") != sim


# -- the export contract in the controller ---------------------------------------------


def _cfg(imu=None):
    """Return the test deploy config with an imu block (None: none, as Explicit500)."""
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    if imu is not None:
        cfg["imu"] = imu
    return cfg


def _ctrl(imu=None, imu_calibration=None, gravity_offset=False):
    """Return a controller on the test config; with gravity_offset, arm_ik's gravity offset on."""
    cfg = _cfg(imu)
    names = cfg["joint_names"]
    if gravity_offset:
        cfg["arm_ik"] = {"rot_weight": 0.3, "null_gain": 0.05, "posture_scale": 0.5, "posture_gain": 1.0,
                         "limit_margin": 0.05, "ik_damping": 0.05, "ik_passes": 3, "posture_from_command": False,
                         "trajectory": True, "seed_rot_weight": 0.3, "posture_actions": False,
                         "trajectory_max_vel": 2.0, "trajectory_max_acc": 10.0, "gravity_offset": True}
        cfg["torque_headroom"] = None
    kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                        extra_frames=[cfg["slam_frame"], "imu_link"])
    return MarlController(cfg, kin, _Echo(12), _Echo(14), imu_calibration=imu_calibration)


class _Echo:
    """Stand in for an actor: a fixed function of its input, so different inputs give different actions."""

    def __init__(self, n):
        self.w = np.random.default_rng(n).normal(0.0, 0.01, (n, 200))
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x.copy())
        return self.w[:, :len(x)] @ x


def _states(cfg, n=5):
    """Return n states with a biased, tilted attitude, a gyro rate and arm angles off the default."""
    out = []
    for _ in range(n):
        q = np.array(cfg["default_joint_pos"], dtype=np.float64) + RNG.normal(0.0, 0.05, 27)
        out.append(RobotState(q=q, dq=RNG.normal(0.0, 0.1, 27), quat=R.quat_mul(_random_attitude(0.1), C.bias_quat(CAL)),
                              gyro=RNG.normal(0.0, 0.3, 3), acc=np.array([0.1, -0.2, 9.8])))
    return out


def test_calibrated_policy_without_a_calibration_is_refused():
    """imu.orientation calibrated and no calibration: the constructor raises, naming the calibration file."""
    with pytest.raises(ValueError, match="imu_calibration.yaml"):
        _ctrl(CALIBRATED_IMU, None)


def test_unknown_orientation_and_calibrated_livox_are_refused():
    """An orientation other than raw / calibrated, or calibrated on a source other than the torso IMU, is refused."""
    with pytest.raises(ValueError, match="orientation"):
        _ctrl({**LOWSTATE_IMU, "orientation": "levelled"}, CAL)
    with pytest.raises(ValueError, match="lowstate"):
        _ctrl({"frame": "livox_imu_link", "orientation": "calibrated"}, CAL)


def test_calibrated_policy_sees_the_corrected_imu():
    """With imu.orientation calibrated the observations' projected gravity is the true attitude's, while the gyro is
    passed as reported (rotate_gyro false), and the calibration is logged."""
    ctrl = _ctrl(CALIBRATED_IMU, CAL)
    assert ctrl.imu_calibration["pitch_deg"] == CAL["pitch_deg"]
    assert "applying" in ctrl.imu_calibration_note
    true = _random_attitude(0.1)
    gyro = np.array([0.1, -0.2, 0.3])
    state = RobotState(q=ctrl.default.copy(), dq=np.zeros(27), quat=R.quat_mul(true, C.bias_quat(CAL)), gyro=gyro,
                       acc=np.zeros(3))
    ctrl.reset(state)
    ctrl.step(state)
    legs = ctrl.last_obs["legs"]                    # Explicit500's layout: acc 0:3, gyro 3:6, gravity 6:9
    np.testing.assert_allclose(legs[6:9], R.projected_gravity(true), atol=1e-12)
    np.testing.assert_allclose(legs[3:6], gyro)
    np.testing.assert_allclose(ctrl.observations(state)["legs"][6:9], R.projected_gravity(true), atol=1e-12)


def test_calibrated_gravity_offset_uses_the_corrected_attitude():
    """gravity_b (the arms' gravity offset) reads the corrected attitude: a calibrated controller on the biased IMU
    gives the gravity, and the joint targets, an uncalibrated one gives on the true attitude."""
    cal_ctrl = _ctrl(CALIBRATED_IMU, CAL, gravity_offset=True)
    ref_ctrl = _ctrl({**LOWSTATE_IMU, "orientation": "raw"}, None, gravity_offset=True)
    for state in _states(_cfg(), 3):
        true_state = RobotState(q=state.q, dq=state.dq, quat=R.quat_mul(state.quat, R.quat_conjugate(C.bias_quat(CAL))),
                                gyro=state.gyro, acc=state.acc)
        for ctrl, s in ((cal_ctrl, state), (ref_ctrl, true_state)):
            ctrl.reset(s)
        a, b = cal_ctrl.step(state), ref_ctrl.step(true_state)
        np.testing.assert_allclose(cal_ctrl.gravity_b(state), ref_ctrl.gravity_b(true_state), atol=1e-9)
        np.testing.assert_allclose(a.q, b.q, atol=1e-9)
        np.testing.assert_allclose(cal_ctrl.last_obs["legs"], ref_ctrl.last_obs["legs"], atol=1e-9)


def test_raw_and_old_exports_ignore_a_calibration(caplog):
    """An old export (no imu block, or quat_bias_deg and no orientation) or an explicit orientation raw reads the IMU
    as reported: with the robot's calibration given its observations and targets are bit for bit those without one,
    and the node's log says the calibration is ignored."""
    for imu in (None, RAW_IMU, {**LOWSTATE_IMU, "orientation": "raw"}):
        with caplog.at_level(logging.INFO, logger="lmg_deploy.controller"):
            caplog.clear()
            given = _ctrl(imu, CAL)
        assert given.imu_calibration is None
        assert any("ignored" in r.message and r.levelno == logging.WARNING for r in caplog.records)
        plain = _ctrl(imu, None)
        for state in _states(_cfg(imu)):
            for ctrl in (given, plain):
                ctrl.reset(state)
            a, b = given.step(state), plain.step(state)
            np.testing.assert_array_equal(a.q, b.q)
            np.testing.assert_array_equal(given.last_obs["legs"], plain.last_obs["legs"])
            np.testing.assert_array_equal(given.last_obs["arms"], plain.last_obs["arms"])
            # the biased quaternion reaches the policy unchanged
            np.testing.assert_array_equal(given.last_obs["legs"][6:9], R.projected_gravity(state.quat))


def test_positional_arguments_still_work():
    """MarlController(cfg, kin, legs, arms) as before (deploy_parity.py's call): no calibration."""
    cfg = _cfg()
    names = cfg["joint_names"]
    kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                        extra_frames=[cfg["slam_frame"]])
    ctrl = MarlController(cfg, kin, _Echo(12), _Echo(14))
    assert ctrl.imu_calibration is None


# -- the estimate ---------------------------------------------------------------------------


def _kin():
    """Return the H1-2 kinematics with imu_link and the soles among its frames."""
    cfg = _cfg()
    names = cfg["joint_names"]
    return ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                         extra_frames=[C.IMU_FRAME, *C.FEET]), cfg


def _standing_q(cfg, left_extra_pitch=0.0):
    """Return a symmetric stance with both soles parallel (each leg's pitch chain summing to the same), a random
    crouch and hip roll; left_extra_pitch tilts the left sole about its pitch axis."""
    q = np.array(cfg["default_joint_pos"], dtype=np.float64)
    hip, knee = -0.35 + RNG.normal(0.0, 0.05), 0.7 + RNG.normal(0.0, 0.05)
    lean = RNG.normal(0.0, 0.05)        # the torso pitched over the feet
    roll = RNG.normal(0.0, 0.02)
    for leg, sign in ((0, 1.0), (6, -1.0)):
        q[leg + 1], q[leg + 3], q[leg + 4] = hip + lean, knee, -(hip + lean) - knee
        q[leg + 2], q[leg + 5] = roll, -roll
    q[4] += left_extra_pitch
    q[12] = RNG.normal(0.0, 0.05)       # the torso joint (yaw)
    return q


def _reported(kin, q, cal, noise_deg=0.0):
    """Return the quaternion the IMU reports at stance q with the right sole level (random yaw): true * R_bias."""
    kin.update(q)
    _, q_imu = kin.frame_pose(C.IMU_FRAME)
    _, q_foot = kin.frame_pose(C.FEET[1])
    q_w_foot = R.quat_from_rotvec(np.array([0.0, 0.0, RNG.uniform(-np.pi, np.pi)]))
    true = R.quat_mul(q_w_foot, R.quat_mul(R.quat_conjugate(q_foot), q_imu))
    noise = R.quat_from_rotvec(np.radians(np.r_[RNG.normal(0.0, noise_deg, 2), 0.0]))
    return R.quat_mul(R.quat_mul(true, C.bias_quat(cal)), noise)


def test_estimator_recovers_the_calibration_from_synthetic_standing():
    """Synthetic standing samples (FK soles -> imu_link, the soles level, the IMU biased, noise): calibrate recovers
    roll and pitch, both feet agree, and the kinematic tilt corrected by the estimate matches the quaternion's."""
    kin, cfg = _kin()
    cal = {"roll_deg": 0.35, "pitch_deg": -2.4, "yaw_deg": 0.0}
    g_l, g_r, quats = [], [], []
    for _ in range(200):
        q = _standing_q(cfg)
        quats.append(_reported(kin, q, cal, noise_deg=0.05))
        g = C.fk_imu_gravity(kin)
        g_l.append(g[0])
        g_r.append(g[1])
    result = C.calibrate(np.array(g_l), np.array(g_r), np.array(quats))
    assert abs(result["roll_deg"] - cal["roll_deg"]) < 0.02
    assert abs(result["pitch_deg"] - cal["pitch_deg"]) < 0.02
    assert max(result["feet_disagreement"]) < 0.02
    assert result["samples"] == 200 and result["yaw_deg"] == 0.0


def test_estimator_ignores_the_yaw_part():
    """A yaw bias doesn't move the roll / pitch estimate (yaw is unobservable from gravity)."""
    kin, cfg = _kin()
    g, quats = [], []
    for _ in range(50):
        q = _standing_q(cfg)
        quats.append(_reported(kin, q, {"roll_deg": -0.5, "pitch_deg": 1.5, "yaw_deg": 0.0}))
        g.append(C.fk_imu_gravity(kin)[1])
    est = C.estimate_bias(np.array(g), np.array(quats))
    np.testing.assert_allclose(est, [-0.5, 1.5], atol=1e-6)


def test_estimator_flags_a_foot_that_is_not_flat():
    """With the left sole tilted 3 deg (pitch) relative to the right, the feet's estimates disagree by about that."""
    kin, cfg = _kin()
    cal = {"roll_deg": 0.0, "pitch_deg": -2.0}
    g_l, g_r, quats = [], [], []
    for _ in range(50):
        q = _standing_q(cfg, left_extra_pitch=np.radians(3.0))
        quats.append(_reported(kin, q, cal))   # the right sole is the level one
        g = C.fk_imu_gravity(kin)
        g_l.append(g[0])
        g_r.append(g[1])
    result = C.calibrate(np.array(g_l), np.array(g_r), np.array(quats))
    assert abs(result["right"][1] - cal["pitch_deg"]) < 0.02
    assert abs(result["feet_disagreement"][1] - 3.0) < 0.1


def test_still_windows_pick_long_still_loaded_runs():
    """still_and_loaded needs still legs, a quiet gyro and both knees loaded; still_windows keeps runs >= min_s, the
    edges trimmed."""
    t = np.arange(0.0, 10.0, 0.01)
    n = len(t)
    dq, gyro, tau = np.zeros((n, 27)), np.zeros((n, 3)), np.zeros((n, 27))
    tau[:, list(C.KNEES)] = -20.0
    tau[t < 2.0, 3] = 6.0             # hanging: the left knee not loaded
    dq[(t > 5.2) & (t < 5.5), 2] = 0.5  # a leg moving
    gyro[t > 8.0] = [0.0, 0.3, 0.0]   # the torso swinging
    tau[(t > 3.0) & (t < 3.5), 9] = -2.0   # a short unload: leaves 2.0..3.0, under min_s 1.5
    mask = C.still_and_loaded(dq, gyro, tau, C.StillThresholds())
    assert not mask[t < 2.0].any() and mask[(t > 3.6) & (t < 4.9)].all() and not mask[t > 8.0].any()
    keep = C.still_windows(t, mask, min_s=1.5, trim_s=0.25)
    assert not keep[t < 3.5].any()                              # 2.0..3.0 is too short
    assert keep[(t > 3.8) & (t < 4.7)].all() and not keep[(t > 3.5) & (t < 3.7)].any()   # trimmed
    assert keep[(t > 5.8) & (t < 7.7)].all() and not keep[t > 7.8].any()


def test_standing_tilt_check_reports_once_and_resets_when_interrupted():
    """StandingTiltCheck reports after hold_s of consecutive standing, once; a break restarts the count; the report
    flags a difference over tol_deg."""
    check = C.StandingTiltCheck(dt=0.1, hold_s=1.0, tol_deg=1.0)
    level = np.array([0.0, 0.0, -1.0])
    tilted = R.projected_gravity(C.bias_quat({"roll_deg": 0.0, "pitch_deg": 1.5}))
    for _ in range(9):
        assert check.update(True, tilted, level) is None
    assert check.update(False, tilted, level) is None          # interrupted: the count restarts
    for _ in range(9):
        assert check.update(True, tilted, level) is None
    report = check.update(True, tilted, level)
    assert report["mismatch"] and abs(report["diff"][1] - 1.5) < 1e-9
    assert check.update(True, tilted, level) is None           # once
    ok = C.StandingTiltCheck(dt=0.1, hold_s=0.2, tol_deg=1.0)
    ok.update(True, level, level)
    assert ok.update(True, level, level)["mismatch"] is False
