'''End-to-end safety layer modes over DDS inside one process (no robot, no network)'''

import copy
import time
import threading
from pathlib import Path

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_safety_layer.core.config import load_config
from g1_safety_layer.core.safety_layer import SafetyLayer
from g1_safety_layer.core.joint_limits import ARM_SDK_WEIGHT_SLOT

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
RAMP_S = 0.3
_factory = []


def _init_dds():
    # the Unitree channel factory is process-wide; initialize it once
    if not _factory:
        ChannelFactoryInitialize(TEST_DOMAIN)
        _factory.append(True)


def _stream_state(stop: threading.Event) -> None:
    publisher = ChannelPublisher('rt/lowstate', LowState_)
    publisher.Init()
    state = LowStateDefault()
    state.mode_machine = 5
    # at the pose the tests command, which the engage gate requires
    state.motor_state[3].q = 0.5  # left knee
    state.motor_state[15].q = 0.2  # left shoulder pitch
    try:
        while not stop.is_set():
            publisher.Write(state)
            time.sleep(0.005)
    finally:
        publisher.Close()


def _command(q: float) -> LowCmd_:
    cmd = LowCmdDefault()
    for i in (12, 13, 14):  # the waist held at 0, as the controller does
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].kp = 60.0
        cmd.motor_cmd[i].kd = 1.5
    cmd.motor_cmd[15].mode = 1  # left shoulder pitch
    cmd.motor_cmd[15].q = q
    cmd.motor_cmd[15].kp = 60.0
    cmd.motor_cmd[15].kd = 1.5
    return cmd


def test_engage_then_hand_back_to_unitree_on_bad_command():
    _init_dds()
    config = load_config(CONFIG_DIR / 'arm_sdk.yaml')
    config['arm_sdk'].update({'engage_seconds': RAMP_S, 'release_seconds': RAMP_S})
    config['logging']['enabled'] = False
    # test_hardening_dds.py tests the rt/lowstate watchdog; relaxed here, so a
    # stall of this process's own state stream thread cannot fire it
    config['control']['state_stale_estop_seconds'] = 1.0

    received = []  # (weight, left shoulder pitch q) as published on rt/arm_sdk
    sub = ChannelSubscriber('rt/arm_sdk', LowCmd_)
    sub.Init(lambda m: received.append(
        (m.motor_cmd[ARM_SDK_WEIGHT_SLOT].q, m.motor_cmd[15].q)), 50)
    cmd_pub = ChannelPublisher('rt/safety/lowcmd_upper_in', LowCmd_)
    cmd_pub.Init()

    layer = SafetyLayer(config)
    threading.Thread(target=layer.start, daemon=True).start()
    stop = threading.Event()
    streamer = threading.Thread(target=_stream_state, args=(stop,), daemon=True)
    streamer.start()
    try:
        time.sleep(0.5)
        assert received == [], 'must stay silent until the first command'

        # good commands: the weight ramps in and our target passes through
        deadline = time.time() + 3 * RAMP_S
        while time.time() < deadline:
            cmd_pub.Write(_command(0.2))
            time.sleep(0.01)
        engage = list(received)
        weights = [w for w, _ in engage]
        assert weights[0] < 0.5 and weights[-1] == 1.0
        assert all(b >= a for a, b in zip(weights, weights[1:])), 'engage must be monotonic'
        assert all(abs(q - 0.2) < 1e-6 for _, q in engage)

        # a NaN command latches the estop: hold the last good target, ramp to 0
        bad = copy.deepcopy(_command(0.2))
        bad.motor_cmd[16].q = float('nan')
        cmd_pub.Write(bad)
        time.sleep(3 * RAMP_S)
        release = received[len(engage):]
        weights = [w for w, _ in release]
        assert weights[-1] == 0.0
        assert all(b <= a for a, b in zip(weights, weights[1:])), 'release must be monotonic'
        assert all(abs(q - 0.2) < 1e-6 for _, q in release), 'must hold the last good target'

        # further commands are ignored once estopped
        n = len(received)
        cmd_pub.Write(_command(0.5))
        time.sleep(0.1)
        assert all(w == 0.0 and abs(q - 0.2) < 1e-6 for w, q in received[n:])
    finally:
        # close every channel: one left to the garbage collector frees its
        # listener but keeps its DDS entity, and the next channel on the same
        # topic (the next test's) then segfaults
        stop.set()
        streamer.join()
        layer.stop()
        sub.Close()
        cmd_pub.Close()


def test_split_mode_merges_channels_and_stamps_the_g1_header():
    _init_dds()
    config = load_config(CONFIG_DIR / 'split.yaml')
    config['logging']['enabled'] = False
    # test_hardening_dds.py tests the rt/lowstate watchdog; relaxed here, so a
    # stall of this process's own state stream thread cannot fire it
    config['control']['state_stale_estop_seconds'] = 1.0

    received = []
    sub = ChannelSubscriber('rt/lowcmd', LowCmd_)
    sub.Init(lambda m: received.append(
        (m.mode_pr, m.mode_machine, m.motor_cmd[3].q, m.motor_cmd[15].q)), 50)
    lower_pub = ChannelPublisher('rt/safety/lowcmd_lower_in', LowCmd_)
    lower_pub.Init()
    upper_pub = ChannelPublisher('rt/safety/lowcmd_upper_in', LowCmd_)
    upper_pub.Init()

    layer = SafetyLayer(config)
    threading.Thread(target=layer.start, daemon=True).start()
    stop = threading.Event()
    streamer = threading.Thread(target=_stream_state, args=(stop,), daemon=True)
    streamer.start()
    try:
        lower = LowCmdDefault()
        lower.motor_cmd[3].q = 0.5  # left knee, from the lower channel
        lower.motor_cmd[15].q = 9.9  # must be ignored: arms come from upper
        upper = _command(0.2)
        upper.motor_cmd[3].q = 9.9  # must be ignored: legs come from lower
        upper.mode_machine = 6  # the H1 value; the layer must echo rt/lowstate's
        deadline = time.time() + 0.5
        while time.time() < deadline:
            lower_pub.Write(lower)
            upper_pub.Write(upper)
            time.sleep(0.01)
        mode_pr, mode_machine, knee, shoulder = received[-1]
        assert (mode_pr, mode_machine) == (0, 5)
        assert abs(knee - 0.5) < 1e-6 and abs(shoulder - 0.2) < 1e-6
    finally:
        stop.set()
        streamer.join()
        layer.stop()
        sub.Close()
        lower_pub.Close()
        upper_pub.Close()
