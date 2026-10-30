"""Tests for g1.unitree_interface.SimInterface on the standalone G1 MJCF in CL_Assets/g1.

robosuite is not needed: the model is the source MJCF (names unprefixed) with
the pelvis framequat g1_robosuite adds, and the handlers are called directly
with crafted messages. SimInterface still opens Unitree DDS channels, so run
this only where DDS cannot reach the robot: tests/robocasa/g1/run_in_container.sh
runs it in the robocasa image with --network none and ROS_DOMAIN_ID=97, with
golem_robocasa on PYTHONPATH.
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
from g1 import unitree_interface as ui

# two levels up from golem_robocasa/g1/: the repo root on the host, /home/code in the container
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(g1.__file__)), "..", ".."))
XML = os.path.join(REPO, "CL_Assets", "g1", "g1_description", "g1_29dof_with_hand_rev_1_0.xml")


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
DATA = mujoco.MjData(MODEL)
IFACE = ui.SimInterface(MODEL, DATA, RESOLVER)
CRC_ = CRC()


def _reset(q=None):
    """Fresh state and latches on the shared interface (DDS is opened once)"""
    mujoco.mj_resetData(MODEL, DATA)
    if q is not None:
        DATA.qpos[RESOLVER.body.qpos] = q
    mujoco.mj_forward(MODEL, DATA)
    IFACE.sim_time = float(DATA.time)
    IFACE._lowcmd = None
    IFACE._hand_cmd = {side: None for side in RESOLVER.hands}
    IFACE.commanded = IFACE._lowcmd_timed_out = False


def _lowcmd(q=0.3, kp=40.0, kd=2.0, tau=0.5, mode=1, mode_machine=5, mode_pr=0):
    msg = LowCmd_default()
    msg.mode_machine, msg.mode_pr = mode_machine, mode_pr
    for i in range(len(BODY_JOINTS)):
        c = msg.motor_cmd[i]
        c.mode, c.q, c.dq, c.tau, c.kp, c.kd = mode, q, 0.0, tau, kp, kd
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


def test_motors_off_until_first_valid_lowcmd():
    """Nothing drives the robot before rt/lowcmd: every motor, body and hand,
    is off (zero torque), however long the sim runs"""
    _reset(np.linspace(-0.3, 0.3, len(BODY_JOINTS)))
    DATA.qvel[RESOLVER.body.qvel] = 0.5
    DATA.ctrl[:] = 1.0   # whatever a previous step left
    for t in (0.0, 1.0, 100.0):
        DATA.time = t
        IFACE.write_ctrl()
        assert not IFACE.commanded
        assert np.all(DATA.ctrl[RESOLVER.body.ctrl] == 0.0)
        for hand in RESOLVER.hands.values():
            assert np.all(DATA.ctrl[hand.ctrl] == 0.0)


def test_invalid_lowcmd_ignored():
    """A frame with a bad CRC, another robot's mode_machine, mode_pr 1 or a
    value that is not finite drives nothing"""
    _reset()
    bad_crc = _lowcmd()
    bad_crc.crc ^= 1
    for msg in (bad_crc, _lowcmd(mode_machine=6), _lowcmd(mode_pr=1), _lowcmd(q=math.nan)):
        IFACE._on_lowcmd(msg)
        IFACE.write_ctrl()
        assert not IFACE.commanded and IFACE._lowcmd is None
        assert np.all(DATA.ctrl[RESOLVER.body.ctrl] == 0.0)
    assert ui.lowcmd_error(_lowcmd(), CRC_) is None
    IFACE._on_lowcmd(_lowcmd())
    assert IFACE.commanded


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


def test_timeout_knob_must_be_finite_and_positive():
    name = "GOLEM_CMD_TIMEOUT"
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


def test_first_valid_lowcmd_drives_motors():
    _reset()
    IFACE._on_lowcmd(_lowcmd(q=0.3, kp=40.0, kd=2.0, tau=0.5))
    assert IFACE.commanded
    IFACE.write_ctrl()
    q, dq = _state()
    b = RESOLVER.body
    expected = np.clip(40.0 * (0.3 - q) + 2.0 * (0.0 - dq) + 0.5, b.tau_min, b.tau_max)
    assert np.allclose(DATA.ctrl[b.ctrl], expected)
    assert np.any(DATA.ctrl[b.ctrl] != 0.0)
    # motor mode 0 is off
    IFACE._on_lowcmd(_lowcmd(mode=0))
    IFACE.write_ctrl()
    assert np.all(DATA.ctrl[b.ctrl] == 0.0)


def test_band_is_none_until_the_sim_builds_it():
    """sim_loop hands the interface its elastic band once built (the tests/g1
    recorders read it); the interface itself never touches it"""
    assert IFACE.band is None


def test_no_arm_sdk_subscription():
    """The sim consumes rt/lowcmd and the hands' commands, and no rt/arm_sdk"""
    assert set(IFACE._subscribers) == {"rt/lowcmd", "rt/dex3/left/cmd", "rt/dex3/right/cmd"}
    assert not [name for name in dir(ui) if "arm_sdk" in name.lower()]
    assert not [name for name in dir(IFACE) if "arm_sdk" in name.lower()]


def test_lowcmd_timeout_turns_motors_off():
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


def test_physics_reset_restamps_the_latched_commands():
    _reset()
    IFACE.sim_time = 40.0
    IFACE._on_lowcmd(_lowcmd())
    DATA.time = 0.002   # MuJoCo's auto-reset puts the time back to 0
    IFACE.physics_reset()
    assert IFACE._lowcmd.time == IFACE.sim_time == 0.002
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
