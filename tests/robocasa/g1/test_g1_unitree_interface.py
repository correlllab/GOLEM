"""Tests for g1.unitree_interface.SimInterface on the standalone leprechaun_g1 MJCF.

robosuite is not needed: the model is the source MJCF (names unprefixed) with
the pelvis framequat g1_robosuite adds, and the handlers are called directly
with crafted messages. SimInterface still opens Unitree DDS channels, so run
this only where DDS cannot reach the robot: tests/robocasa/g1/run_in_container.sh
runs it in the robocasa image with --network none and ROS_DOMAIN_ID=97, with
h1_robocasa on PYTHONPATH.
"""
import math
import os
import sys

import mujoco
import numpy as np

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_ as HandCmd_default
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.utils.crc import CRC

import g1
from g1.sim_names import BODY_JOINTS, HAND_JOINTS, NameResolver
from g1.stand_in import StandIn
from g1 import unitree_interface as ui

# two levels up from h1_robocasa/g1/: the repo root on the host, /home/code in the container
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(g1.__file__)), "..", ".."))
XML = os.path.join(REPO, "third_party", "leprechaun_g1", "assets", "g1_description",
                   "g1_29dof_with_hand_rev_1_0.xml")
UPPER = slice(ui.ARM_SDK_SLOTS.start, ui.ARM_SDK_SLOTS.stop)


def _model():
    xml = open(XML).read()
    xml = xml.replace('meshdir="meshes"', f'meshdir="{os.path.join(os.path.dirname(XML), "meshes")}"')
    xml = xml.replace("<sensor>", '<sensor>\n    <framequat name="imu-pelvis-orientation" '
                                  'objtype="site" objname="imu_in_pelvis"/>')
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = 0.002
    return m


MODEL = _model()
RESOLVER = NameResolver(MODEL, robot_prefix="")
STAND_IN = StandIn(RESOLVER.body.names, MODEL.opt.timestep)
DATA = mujoco.MjData(MODEL)
IFACE = ui.SimInterface(MODEL, DATA, RESOLVER, STAND_IN)
CRC_ = CRC()


def _reset(q=None):
    """Fresh state and latches on the shared interface (DDS is opened once)"""
    mujoco.mj_resetData(MODEL, DATA)
    if q is not None:
        DATA.qpos[RESOLVER.body.qpos] = q
    mujoco.mj_forward(MODEL, DATA)
    IFACE.sim_time = float(DATA.time)
    IFACE._lowcmd = IFACE._arm_sdk = None
    IFACE._hand_cmd = {side: None for side in RESOLVER.hands}
    IFACE.released = IFACE._lowcmd_timed_out = IFACE._arm_sdk_timed_out = False
    IFACE.arm_sdk_w, IFACE._arm_sdk_w_time, IFACE._arm_sdk_resuming = 0.0, float(DATA.time), False


def _lowcmd(q=0.3, kp=40.0, kd=2.0, tau=0.5, mode=1, weight=None, mode_machine=5, mode_pr=0):
    msg = LowCmd_default()
    msg.mode_machine, msg.mode_pr = mode_machine, mode_pr
    for i in range(len(BODY_JOINTS)):
        c = msg.motor_cmd[i]
        c.mode, c.q, c.dq, c.tau, c.kp, c.kd = mode, q, 0.0, tau, kp, kd
    if weight is not None:
        msg.motor_cmd[ui.ARM_SDK_WEIGHT_SLOT].q = weight
    msg.crc = CRC_.Crc(msg)
    return msg


def _state():
    b = RESOLVER.body
    return DATA.qpos[b.qpos].copy(), DATA.qvel[b.qvel].copy()


def test_names_resolve_in_slot_order():
    b = RESOLVER.body
    for i, name in enumerate(BODY_JOINTS):
        jid = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert b.qpos[i] == MODEL.jnt_qposadr[jid] and MODEL.actuator_trnid[b.ctrl[i], 0] == jid
    # the MJCF orders the right hand's index before its middle finger; DDS slot 3 is middle_0
    for side, hand in RESOLVER.hands.items():
        names = [mujoco.mj_id2name(MODEL, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in hand.ctrl]
        assert names == list(HAND_JOINTS[side]), names
    first = RESOLVER.hands["right"].ctrl[0]
    assert list(RESOLVER.hands["right"].ctrl - first) == [0, 1, 2, 5, 6, 3, 4]
    assert b.tau_max[BODY_JOINTS.index("left_knee_joint")] == 139.0
    assert b.tau_max[BODY_JOINTS.index("left_wrist_yaw_joint")] == 5.0


def test_stand_in_drives_before_any_command():
    _reset()
    IFACE.write_ctrl()
    q, dq = _state()
    b = RESOLVER.body
    expected = np.clip(STAND_IN.torques(q, dq), b.tau_min, b.tau_max)
    assert np.array_equal(DATA.ctrl[b.ctrl], expected)
    for hand in RESOLVER.hands.values():  # never commanded: limp
        assert np.all(DATA.ctrl[hand.ctrl] == 0.0)


def test_lowcmd_frames_are_validated():
    _reset()
    bad_crc = _lowcmd()
    bad_crc.crc ^= 1
    for msg in (bad_crc, _lowcmd(mode_machine=0), _lowcmd(mode_pr=1), _lowcmd(q=math.nan)):
        IFACE._on_lowcmd(msg)
        assert not IFACE.released and IFACE._lowcmd is None
    assert ui.lowcmd_error(_lowcmd(), CRC_) is None
    IFACE._on_lowcmd(_lowcmd())
    assert IFACE.released


def test_fast_crc_is_the_sdks():
    """The sim's CRC (CRC_ is the same per-process object, sped up by the
    interface) equals unitree_sdk2py's own on random frames, and its
    pure-Python reference on raw bytes"""
    assert getattr(CRC_, "_g1_sim_fast", False)
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(20):
        cmd = _lowcmd(q=float(rng.normal()), kp=float(rng.uniform(0, 100)), tau=float(rng.normal()))
        cmd.reserve = [int(v) for v in rng.integers(0, 2**32, 4)]
        state = LowState_default()
        state.tick = int(rng.integers(0, 2**32))
        state.imu_state.quaternion = [float(v) for v in rng.normal(size=4)]
        for motor in state.motor_state:
            motor.q, motor.dq, motor.tau_est = (float(v) for v in rng.normal(size=3))
            motor.temperature = [int(v) for v in rng.integers(-100, 100, 2)]
        frames += [cmd, state]
    fast = [CRC_.Crc(msg) for msg in frames]
    raw = bytes(rng.integers(0, 256, 2092, dtype=np.uint8))
    fast_raw = CRC_._crc_ctypes(CRC_._CRC__Trans(raw))
    fast_words, fast_crc32 = CRC_.__dict__.pop("_CRC__Trans"), CRC_.__dict__.pop("_crc_ctypes")
    try:   # the SDK's own methods, with the sim's taken off the object
        assert [CRC_.Crc(msg) for msg in frames] == fast
        assert CRC_._crc_py(CRC_._CRC__Trans(raw)) == fast_raw
    finally:
        CRC_._CRC__Trans, CRC_._crc_ctypes = fast_words, fast_crc32


def test_timeout_knobs_must_be_finite_and_positive():
    for name in ("GOLEM_ARM_SDK_TIMEOUT", "GOLEM_CMD_TIMEOUT"):
        assert ui.env_seconds(name, 0.5, {}) == 0.5
        assert ui.env_seconds(name, 0.5, {name: " "}) == 0.5   # empty: the default, as compose passes it
        assert ui.env_seconds(name, 0.5, {name: "0.25"}) == 0.25
        for bad in ("nan", "inf", "-inf", "-0.5", "0", "fast"):
            try:
                ui.env_seconds(name, 0.5, {name: bad})
            except ValueError as e:
                assert name in str(e) and repr(bad) in str(e), e
            else:
                raise AssertionError(f"{name}={bad!r} accepted")


def test_lowcmd_drives_every_motor_after_release():
    _reset()
    IFACE._on_lowcmd(_lowcmd(q=0.3, kp=40.0, kd=2.0, tau=0.5))
    IFACE.write_ctrl()
    q, dq = _state()
    b = RESOLVER.body
    expected = np.clip(40.0 * (0.3 - q) + 2.0 * (0.0 - dq) + 0.5, b.tau_min, b.tau_max)
    assert np.allclose(DATA.ctrl[b.ctrl], expected)
    # motor mode 0 is off
    IFACE._on_lowcmd(_lowcmd(mode=0))
    IFACE.write_ctrl()
    assert np.all(DATA.ctrl[b.ctrl] == 0.0)
    # rt/arm_sdk is ignored for good
    IFACE._on_arm_sdk(_lowcmd(weight=1.0))
    assert IFACE._arm_sdk is None


def test_lowcmd_timeout_turns_the_motors_off():
    _reset()
    IFACE._on_lowcmd(_lowcmd())
    IFACE.write_ctrl()
    assert np.any(DATA.ctrl[RESOLVER.body.ctrl] != 0.0)
    DATA.time = IFACE.lowcmd_timeout + 0.01
    IFACE.write_ctrl()
    assert np.all(DATA.ctrl[RESOLVER.body.ctrl] == 0.0)
    IFACE.sim_time = float(DATA.time)
    IFACE._on_lowcmd(_lowcmd())  # back
    IFACE.write_ctrl()
    assert np.any(DATA.ctrl[RESOLVER.body.ctrl] != 0.0)


def test_arm_sdk_blends_waist_and_arms_by_weight():
    _reset()
    IFACE.write_ctrl()
    hold = DATA.ctrl[RESOLVER.body.ctrl].copy()
    IFACE._on_arm_sdk(_lowcmd(q=0.3, kp=60.0, kd=1.5, tau=0.0, mode=0, weight=0.25, mode_machine=0))
    assert not IFACE.released  # rt/arm_sdk needs no mode_machine and never releases the stand-in
    IFACE.write_ctrl()
    q, dq = _state()
    b = RESOLVER.body
    tau_hold = STAND_IN.torques(q, dq)
    sdk = 60.0 * (0.3 - q) + 1.5 * (0.0 - dq)  # mode 0: rt/arm_sdk has no per-motor switch
    expected = tau_hold.copy()
    expected[UPPER] = 0.25 * sdk[UPPER] + 0.75 * tau_hold[UPPER]
    assert np.allclose(DATA.ctrl[b.ctrl], np.clip(expected, b.tau_min, b.tau_max))
    assert np.array_equal(DATA.ctrl[b.ctrl][:12], hold[:12])  # the legs stay HOMIE's
    # the weight is clipped to [0, 1]
    IFACE._on_arm_sdk(_lowcmd(weight=3.0))
    assert IFACE._arm_sdk.weight == 1.0
    # a bad CRC is dropped
    bad = _lowcmd(weight=0.5)
    bad.crc ^= 1
    IFACE._on_arm_sdk(bad)
    assert IFACE._arm_sdk.weight == 1.0


def _weights(sdk, start, stop):
    """arm_sdk_weight once per physics step from sim time start to stop"""
    dt = MODEL.opt.timestep
    return [IFACE.arm_sdk_weight(sdk, start + i * dt) for i in range(round((stop - start) / dt) + 1)]


def test_arm_sdk_weight_ramps_to_zero_when_silent():
    _reset()
    IFACE._on_arm_sdk(_lowcmd(weight=0.8))
    sdk = IFACE._arm_sdk
    timeout, ramp = IFACE.arm_sdk_timeout, ui.ARM_SDK_RELEASE_S
    assert set(_weights(sdk, 0.0, timeout - 0.01)) == {0.8}
    # then down at 1/ARM_SDK_RELEASE_S per second, no step
    w = _weights(sdk, timeout - 0.01, timeout + 0.4 * ramp)
    assert math.isclose(w[-1], 0.4, abs_tol=0.01)
    assert max(abs(np.diff(w))) <= MODEL.opt.timestep / ramp + 1e-9
    assert _weights(sdk, timeout + 0.4 * ramp, timeout + ramp)[-1] == 0.0
    DATA.time = timeout + ramp + 0.002
    IFACE.write_ctrl()
    q, dq = _state()
    b = RESOLVER.body
    assert np.array_equal(DATA.ctrl[b.ctrl], np.clip(STAND_IN.torques(q, dq), b.tau_min, b.tau_max))


def _silent_then_back(weight_before, silence, weight_after):
    """rt/arm_sdk at weight_before from t=0, silent from then for timeout +
    silence, then one frame at weight_after; returns the sim time it came"""
    _reset()
    IFACE._on_arm_sdk(_lowcmd(weight=weight_before))
    back = IFACE.arm_sdk_timeout + silence
    _weights(IFACE._arm_sdk, 0.0, back)
    IFACE.sim_time = back
    IFACE._on_arm_sdk(_lowcmd(weight=weight_after))
    return back


def test_arm_sdk_weight_ramps_back_up_when_frames_return():
    ramp = ui.ARM_SDK_RELEASE_S
    back = _silent_then_back(1.0, 0.3 * ramp, 1.0)   # fell to 0.7 while silent
    assert math.isclose(IFACE.arm_sdk_w, 0.7, abs_tol=0.01)
    w = [IFACE.arm_sdk_w] + _weights(IFACE._arm_sdk, back, back + 0.5 * ramp)
    steps = np.diff(w)   # back up at 1/ARM_SDK_RELEASE_S per second, no step
    assert np.all(steps >= 0.0) and steps.max() <= MODEL.opt.timestep / ramp + 1e-9
    assert w[-1] == 1.0 and math.isclose(w.index(1.0) * MODEL.opt.timestep, 0.3 * ramp, abs_tol=0.005)


def test_arm_sdk_resume_follows_a_lower_weight_at_once():
    back = _silent_then_back(1.0, 0.5 * ui.ARM_SDK_RELEASE_S, 1.0)
    w = _weights(IFACE._arm_sdk, back, back + 0.1)   # on the way back up, at ~0.6
    assert 0.55 < w[-1] < 0.65
    IFACE.sim_time = back + 0.1
    IFACE._on_arm_sdk(_lowcmd(weight=0.2))   # e.g. the safety layer releasing
    assert IFACE.arm_sdk_weight(IFACE._arm_sdk, back + 0.102) == 0.2


def test_physics_reset_restamps_the_latched_commands():
    _reset()
    IFACE.sim_time = 40.0
    IFACE._on_arm_sdk(_lowcmd(weight=1.0))
    IFACE._on_lowcmd(_lowcmd())
    DATA.time = 0.002   # MuJoCo's auto-reset puts the time back to 0
    IFACE.physics_reset()
    assert IFACE._arm_sdk.time == IFACE._lowcmd.time == IFACE.sim_time == 0.002
    DATA.time = 0.002 + IFACE.lowcmd_timeout + 0.01   # so the timeouts fire again
    IFACE.write_ctrl()
    assert np.all(DATA.ctrl[RESOLVER.body.ctrl] == 0.0)


def test_hand_commands_by_joint_name_and_mode_byte():
    _reset()
    assert ui.hand_motor_on(0x01 << 4 | 3)
    assert not ui.hand_motor_on(0x01 << 4 | 3 | 1 << 7)  # timeout bit: released
    assert not ui.hand_motor_on(3)                         # status 0
    msg = HandCmd_default()
    for i, c in enumerate(msg.motor_cmd):
        c.mode = i | 0x01 << 4
        c.q, c.kp, c.kd = 0.5, 1.5, 0.1
    msg.motor_cmd[6].mode |= 1 << 7
    IFACE._on_hand_cmd("right", msg)
    IFACE.write_ctrl()
    hand = RESOLVER.hands["right"]
    q = DATA.qpos[hand.qpos]
    dq = DATA.qvel[hand.qvel]
    expected = np.clip(1.5 * (0.5 - q) + 0.1 * (0.0 - dq), hand.tau_min, hand.tau_max)
    expected[6] = 0.0
    assert np.allclose(DATA.ctrl[hand.ctrl], expected)
    assert np.all(DATA.ctrl[RESOLVER.hands["left"].ctrl] == 0.0)


def test_lowstate_and_hand_state():
    q0 = np.linspace(-0.2, 0.2, len(BODY_JOINTS))
    _reset(q0)
    writes = {"lowstate": 0, "hand": 0}
    IFACE._low_state_pub.Write = lambda msg, *a: writes.__setitem__("lowstate", writes["lowstate"] + 1)
    for pub in IFACE._hand_state_pub.values():
        pub.Write = lambda msg, *a: writes.__setitem__("hand", writes["hand"] + 1)
    IFACE._steps = 0
    for _ in range(5):
        IFACE.write_ctrl()
        mujoco.mj_step(MODEL, DATA)
        IFACE.after_step()
    msg = IFACE.low_state
    assert writes == {"lowstate": 5, "hand": 2}  # 500 Hz and 100 Hz of sim time, both hands
    assert msg.tick == round(DATA.time * 1000) == 10
    assert (msg.mode_machine, msg.mode_pr) == (5, 0)
    assert CRC_.Crc(msg) == msg.crc
    b = RESOLVER.body
    assert np.allclose([m.q for m in msg.motor_state[:29]], DATA.qpos[b.qpos], atol=1e-6)
    assert np.allclose([m.tau_est for m in msg.motor_state[:29]], DATA.qfrc_actuator[b.qvel], atol=1e-4)
    quat = DATA.sensordata[RESOLVER.sensor_adr["quat"][0]:][:4]
    assert np.allclose(msg.imu_state.quaternion, quat, atol=1e-6) and quat[0] > 0.99  # w first
    acc = DATA.sensordata[RESOLVER.sensor_adr["acc"][0]:][:3]
    assert np.allclose(msg.imu_state.accelerometer, acc, atol=1e-5)
    assert np.allclose(msg.imu_state.rpy, ui.rpy_from_quat(*quat), atol=1e-6)
    hand = RESOLVER.hands["left"]
    state = IFACE._hand_state["left"]
    assert np.allclose([m.q for m in state.motor_state[:7]], DATA.qpos[hand.qpos], atol=1e-6)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    IFACE.shutdown()
    sys.exit(1 if failed else 0)
