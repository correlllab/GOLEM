'''split_mode and full_body_mode over DDS inside one process (no robot, no network)'''

import json
import time
import itertools
import threading
from pathlib import Path
from typing import Callable
from contextlib import contextmanager

import pytest
import numpy as np

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_description.joints import BODY
from g1_safety_layer.core.config import load_config
from g1_safety_layer.core.joint_limits import MOTOR_COUNT
from g1_safety_layer.core.safety_layer import UPPER_STALE_ESTOP_SECONDS, SafetyLayer

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
SPLIT_INPUTS = ('low_cmd_lower_in', 'low_cmd_upper_in')
# (config, input topic keys) of each rt/lowcmd mode
LOWCMD_MODES = [('split.yaml', SPLIT_INPUTS), ('full.yaml', ('low_cmd_in',))]
KNEE = 3  # left knee, a leg slot
SHOULDER = 15  # left shoulder pitch, an arm slot
# slack for thread scheduling when timing a watchdog, which fires a few ms after
# its timeout; below the 0.2 s a watchdog firing at twice its 0.2 s timeout, or
# 0.25 s late, would add
LATE_S = 0.1
_factory = []
_runs = itertools.count()


def _init_dds():
    # the Unitree channel factory is process-wide; initialize it once
    if not _factory:
        ChannelFactoryInitialize(TEST_DOMAIN)
        _factory.append(True)


def _publisher(topic: str) -> ChannelPublisher:
    _init_dds()
    publisher = ChannelPublisher(topic, LowCmd_)
    publisher.Init()
    return publisher


def _state() -> LowState_:
    '''rt/lowstate of a G1 at rest at q = 0, reporting mode_machine 5'''
    state = LowStateDefault()
    state.mode_machine = 5
    return state


def _stream_state(topic: str, stop: threading.Event, state: LowState_) -> None:
    publisher = ChannelPublisher(topic, LowState_)
    publisher.Init()
    try:
        while not stop.is_set():
            publisher.Write(state)
            time.sleep(0.005)
    finally:
        publisher.Close()


def _command(slot: int, q: float) -> LowCmd_:
    cmd = LowCmdDefault()
    cmd.motor_cmd[slot].mode = 1
    cmd.motor_cmd[slot].q = q
    cmd.motor_cmd[slot].kp = 60.0
    cmd.motor_cmd[slot].kd = 1.5
    return cmd


def _publish_for(seconds: float, *pairs) -> float:
    '''Write each (publisher, command) pair at 100 Hz; return the last write time'''
    deadline = time.time() + seconds
    last = time.time()
    while time.time() < deadline:
        # taken before the write, so the layer receives it no earlier
        last = time.time()
        for publisher, cmd in pairs:
            publisher.Write(cmd)
        time.sleep(0.01)
    return last


def _is_zeroed(msg: LowCmd_) -> bool:
    '''Every motor off, the rt/lowcmd estop command (estop_response zero)'''
    return all(
        m.mode == 0 and m.q == m.dq == m.tau == m.kp == m.kd == 0.0
        for m in msg.motor_cmd[:MOTOR_COUNT]
    )


def _is_damped(msg: LowCmd_, kd: np.ndarray) -> bool:
    '''Every motor on with kp = 0 and only kd, the estop command of estop_response damp'''
    return all(
        m.mode == 1 and m.q == m.dq == m.tau == m.kp == 0.0 and abs(m.kd - kd[i]) < 1e-6
        for i, m in enumerate(msg.motor_cmd[:MOTOR_COUNT])
    )


def _first_estop(
    published: list, is_estop: Callable[[LowCmd_], bool] = _is_zeroed
) -> int:
    '''Index of the first estop command; the estop latches, so all later ones are too'''
    idx = next((i for i, (_, msg) in enumerate(published) if is_estop(msg)), None)
    assert idx is not None, 'must estop'
    assert all(is_estop(msg) for _, msg in published[idx:]), 'the estop must latch'
    return idx


def _assert_on_time(stale: float, timeout: float) -> None:
    '''Check the watchdog fired once its timeout had passed, and at most LATE_S later'''
    assert timeout - 0.02 < stale < timeout + LATE_S, f'fired after {stale:.3f} s'


def _events(capsys) -> list[dict]:
    '''JSON events the safety layer printed'''
    lines = capsys.readouterr().out.splitlines()
    return [json.loads(line) for line in lines if line.startswith('{')]


def _estop_reasons(events: list[dict], event: str = 'estop_command_written') -> list[str]:
    return [e['reason'] for e in events if e['event'] == event]


@contextmanager
def _running_layer(
    config_name: str, *inputs: str, state: LowState_ | None = None, **sections
):
    '''
    Run a SafetyLayer fed rt/lowstate (state, which the test may change while it
    streams); yield its config, the (receive time, LowCmd_) it writes on
    rt/lowcmd, and a publisher per input topic key. Each keyword argument
    updates that section of the loaded config
    '''
    _init_dds()
    config = load_config(CONFIG_DIR / config_name)
    config['logging']['enabled'] = False
    # these tests command poses up to 0.5 rad from the streamed state, which the
    # configured engage gate (0.15 rad) refuses; test_hardening_dds.py tests it
    config['engage']['max_position_error'] = 1.0
    # and they time the input watchdogs: the 0.1 s rt/lowstate watchdog would
    # also fire on a stall of this process's own state stream thread (up to
    # 0.09 s in a 120 s probe), so it is relaxed here; test_hardening_dds.py
    # tests it
    config['control']['state_stale_estop_seconds'] = 1.0
    for section, values in sections.items():
        config[section].update(values)
    # topics of its own, and every channel closed afterwards, so no channel of
    # another test can match this run's
    prefix = f'test_lowcmd_modes/{next(_runs)}/'
    topics = {key: prefix + name for key, name in config['topics'].items()}
    config['topics'] = topics
    if state is None:
        state = _state()

    received = []
    sub = ChannelSubscriber(topics['low_cmd_out'], LowCmd_)
    sub.Init(lambda m: received.append((time.time(), m)), 50)
    publishers = [_publisher(topics[key]) for key in inputs]

    layer = SafetyLayer(config)
    threading.Thread(target=layer.start, daemon=True).start()
    stop = threading.Event()
    streamer = threading.Thread(
        target=_stream_state, args=(topics['low_state'], stop, state), daemon=True
    )
    streamer.start()
    try:
        # the layer's publisher loop starts after its 0.2 s writer setup
        time.sleep(0.5)
        yield config, received, publishers
    finally:
        stop.set()
        streamer.join()
        layer.stop()
        sub.Close()
        for publisher in publishers:
            publisher.Close()


@pytest.mark.parametrize('first', ['lower', 'upper'])
def test_split_mode_is_silent_until_both_inputs_have_sent(first, capsys):
    lower = _command(KNEE, 0.5)
    lower.motor_cmd[SHOULDER].q = 0.9  # must be ignored: arms come from upper
    upper = _command(SHOULDER, 0.2)
    upper.motor_cmd[KNEE].q = 0.9  # must be ignored: legs come from lower
    with _running_layer('split.yaml', *SPLIT_INPUTS) as (_, received, pubs):
        lower_pub, upper_pub = pubs
        assert received == [], 'must not publish the zeroed boot command'

        alone = (lower_pub, lower) if first == 'lower' else (upper_pub, upper)
        _publish_for(0.3, alone)
        assert received == [], f'must stay silent with only the {first} input'

        _publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
        published = list(received)

    assert published, 'must publish once both inputs have sent'
    for _, msg in published:
        assert (msg.mode_pr, msg.mode_machine) == (0, 5)
        assert abs(msg.motor_cmd[KNEE].q - 0.5) < 1e-6
        assert abs(msg.motor_cmd[SHOULDER].q - 0.2) < 1e-6
    engage = [e for e in _events(capsys) if e['event'] == 'lowcmd_engage']
    assert engage == [{'event': 'lowcmd_engage', 'mode': 'split_mode'}]


def test_split_mode_estops_when_the_lower_input_goes_stale(capsys):
    lower = _command(KNEE, 0.5)
    upper = _command(SHOULDER, 0.2)
    with _running_layer('split.yaml', *SPLIT_INPUTS) as (config, received, pubs):
        lower_pub, upper_pub = pubs
        timeout = config['control']['lower_stale_estop_seconds']
        last_lower = _publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
        # the upper input keeps streaming; only the lower one went quiet
        _publish_for(timeout + LATE_S, (upper_pub, upper))
        # the lower input coming back does not clear the latched estop
        _publish_for(0.1, (lower_pub, lower), (upper_pub, upper))
        published = list(received)

    idx = _first_estop(published)
    assert idx > 0, 'must have published the merged command first'
    assert all(abs(msg.motor_cmd[KNEE].q - 0.5) < 1e-6 for _, msg in published[:idx])
    _assert_on_time(published[idx][0] - last_lower, timeout)
    reasons = _estop_reasons(_events(capsys))
    assert len(reasons) == 1 and reasons[0].startswith('Lower-body command stale')


def test_split_mode_upper_watchdog_still_estops(capsys):
    lower = _command(KNEE, 0.5)
    upper = _command(SHOULDER, 0.2)
    with _running_layer('split.yaml', *SPLIT_INPUTS) as (_, received, pubs):
        lower_pub, upper_pub = pubs
        last_upper = _publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
        # the lower input keeps streaming; only the upper one went quiet
        _publish_for(UPPER_STALE_ESTOP_SECONDS + LATE_S, (lower_pub, lower))
        published = list(received)

    idx = _first_estop(published)
    assert idx > 0, 'must have published the merged command first'
    _assert_on_time(published[idx][0] - last_upper, UPPER_STALE_ESTOP_SECONDS)
    reasons = _estop_reasons(_events(capsys))
    assert len(reasons) == 1 and reasons[0].startswith('Upper-body command stale')


def test_split_mode_estop_before_both_inputs_stays_silent(capsys):
    lower = _command(KNEE, 0.5)
    upper = _command(SHOULDER, 0.2)
    bad = _command(KNEE, float('nan'))
    with _running_layer('split.yaml', *SPLIT_INPUTS) as (_, received, pubs):
        lower_pub, upper_pub = pubs
        _publish_for(0.1, (lower_pub, lower))
        lower_pub.Write(bad)
        # each input has its own reader thread: let the NaN latch the estop
        # before the upper input starts
        _publish_for(0.1, (lower_pub, lower))
        _publish_for(0.3, (lower_pub, lower), (upper_pub, upper))
        published = list(received)

    assert published == [], 'never took control, so the estop must not write either'
    events = _events(capsys)
    assert not [e for e in events if e['event'] == 'lowcmd_engage']
    # decided but never written: its own event, not estop_command_written
    assert _estop_reasons(events) == []
    reasons = _estop_reasons(events, 'estop_latched_silent')
    assert len(reasons) == 1
    assert reasons[0].startswith('Command validation failed on split_mode_lower')


def test_full_body_mode_waits_for_its_first_command_and_estops_when_stale(capsys):
    cmd = _command(KNEE, 0.5)
    cmd.motor_cmd[KNEE].kp = 300.0  # above kp_max (200): clipped and reported once
    cmd.motor_cmd[KNEE].kd = 20.0  # above kd_max (10)
    with _running_layer('full.yaml', 'low_cmd_in') as (config, received, (pub,)):
        timeout = config['control']['lower_stale_estop_seconds']
        assert received == [], 'must not publish the zeroed boot command'
        last = _publish_for(0.3, (pub, cmd))
        time.sleep(timeout + LATE_S)
        published = list(received)

    idx = _first_estop(published)
    assert idx > 0, 'must have published the command first'
    for _, msg in published[:idx]:
        assert (msg.mode_pr, msg.mode_machine) == (0, 5)
        knee = msg.motor_cmd[KNEE]
        assert abs(knee.q - 0.5) < 1e-6 and (knee.kp, knee.kd) == (200.0, 10.0)
    _assert_on_time(published[idx][0] - last, timeout)

    events = _events(capsys)
    engage = [e for e in events if e['event'] == 'lowcmd_engage']
    assert engage == [{'event': 'lowcmd_engage', 'mode': 'full_body_mode'}]
    # about 30 commands sent, one event per clipped joint and gain
    clipped = [e for e in events if e['event'] == 'gain_clipped']
    assert [(e['joint'], e['gain'], e['commanded'], e['max']) for e in clipped] == [
        ('left_knee_joint', 'kp', 300.0, 200.0),
        ('left_knee_joint', 'kd', 20.0, 10.0),
    ]
    reasons = _estop_reasons(events)
    assert len(reasons) == 1 and reasons[0].startswith('Full-body command stale')


def test_lower_watchdog_uses_the_configured_timeout(capsys):
    # not full.yaml's 0.2 s, so a watchdog that ignores the config fails
    timeout = 0.5
    control = {'lower_stale_estop_seconds': timeout}
    cmd = _command(KNEE, 0.5)
    with _running_layer('full.yaml', 'low_cmd_in', control=control) as (_, received, (pub,)):
        last = _publish_for(0.3, (pub, cmd))
        time.sleep(timeout + LATE_S)
        published = list(received)

    idx = _first_estop(published)
    assert idx > 0, 'must have published the command first'
    _assert_on_time(published[idx][0] - last, timeout)
    reasons = _estop_reasons(_events(capsys))
    assert len(reasons) == 1 and reasons[0].startswith('Full-body command stale')
    assert reasons[0].endswith(f'(> {timeout}s)')


@pytest.mark.parametrize('config_name, inputs', LOWCMD_MODES, ids=['split', 'full'])
def test_damp_estop_keeps_every_motor_on_with_only_kd(config_name, inputs, capsys):
    kd = np.asarray([0.1 * (i + 1) for i in range(MOTOR_COUNT)])  # per joint, under kd_max
    damp = {'mode': 'damp', 'kd': kd}
    bad = _command(KNEE, float('nan'))
    with _running_layer(config_name, *inputs, estop_response=damp) as (_, received, pubs):
        # the knee command to the leg input (the lower one in split_mode), the
        # shoulder command to the upper one
        pairs = list(zip(pubs, (_command(KNEE, 0.5), _command(SHOULDER, 0.2))))
        _publish_for(0.3, *pairs)
        pubs[0].Write(bad)  # a NaN on the leg input latches the estop
        _publish_for(0.2, *pairs)
        published = list(received)

    idx = _first_estop(published, lambda msg: _is_damped(msg, kd))
    assert idx > 0, 'must have published the commands first'
    assert all(abs(msg.motor_cmd[KNEE].q - 0.5) < 1e-6 for _, msg in published[:idx])
    for _, msg in published[idx:]:
        assert (msg.mode_pr, msg.mode_machine) == (0, 5)
    reasons = _estop_reasons(_events(capsys))
    assert len(reasons) == 1 and reasons[0].startswith('Command validation failed')


@pytest.mark.parametrize('config_name, inputs', LOWCMD_MODES, ids=['split', 'full'])
def test_state_limits_are_checked_only_once_engaged(config_name, inputs, capsys):
    # a limp knee resting past its URDF limit (where split.yaml and full.yaml
    # estop) while the layer waits for its inputs
    past_limit = BODY[KNEE].lower - 0.05
    state = _state()
    state.motor_state[KNEE].q = past_limit
    with _running_layer(config_name, *inputs, state=state) as (_, received, pubs):
        # it streamed through the setup; bring the knee back and let the
        # samples already queued drain before engaging
        state.motor_state[KNEE].q = 0.0
        time.sleep(0.1)
        pairs = list(zip(pubs, (_command(KNEE, 0.5), _command(SHOULDER, 0.2))))
        _publish_for(0.3, *pairs)
        # engaged now, so the same state latches the estop
        state.motor_state[KNEE].q = past_limit
        _publish_for(0.2, *pairs)
        published = list(received)

    idx = _first_estop(published)
    assert idx > 0, 'a state past its limit before engaging must not latch the estop'
    assert all(abs(msg.motor_cmd[KNEE].q - 0.5) < 1e-6 for _, msg in published[:idx])
    events = _events(capsys)
    estops = [e for e in events if e['event'].startswith('estop')]
    assert [e['event'] for e in estops] == ['estop_command_written']
    assert estops[0]['reason'].startswith('motor 3 (left_knee_joint) q out of estop range')
