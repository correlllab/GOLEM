'''
Engage gate, state watchdog, second writer, waist floor, exit and clocks over
DDS inside one process (no robot, no network)
'''

import time
import itertools
import threading
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowStateDefault
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

from g1_safety_layer.core import safety_layer as safety_layer_module
from g1_safety_layer.core.clock import watchdog_time
from g1_safety_layer.core.config import load_config
from g1_safety_layer.core.joint_limits import ARM_SDK_WEIGHT_SLOT, MOTOR_COUNT
from g1_safety_layer.core.safety_layer import EXIT_HOLD_SECONDS, SafetyLayer

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
TEST_DOMAIN = 97  # never 0 (the real robot) or 1 (the sim)
RAMP_S = 0.3
KNEE = 3  # left knee, a leg slot
SHOULDER = 15  # left shoulder pitch, an arm slot
WAIST = (12, 13, 14)
# the pose rt/lowstate reports, which the tests command unless they test the gate
KNEE_Q = 0.5
SHOULDER_Q = 0.2
# slack for thread scheduling when timing a watchdog (see test_lowcmd_modes_dds.py)
LATE_S = 0.1
ARM_SDK = ('low_cmd_upper_in',)
SPLIT = ('low_cmd_lower_in', 'low_cmd_upper_in')
FULL = ('low_cmd_in',)
FAST_RAMPS = {'engage_seconds': RAMP_S, 'release_seconds': RAMP_S}
# the rt/lowstate watchdog in the tests of something else: the shipped 0.1 s
# would also fire on a stall of this process's own state stream thread (up to
# 0.09 s in a 120 s probe), which the robot's rt/lowstate does not share
HARNESS_STATE_STALE_S = 1.0
_factory = []
_runs = itertools.count()


def _init_dds():
    # the Unitree channel factory is process-wide; initialize it once
    if not _factory:
        ChannelFactoryInitialize(TEST_DOMAIN)
        _factory.append(True)


def _state(mode_machine: int = 5) -> LowState_:
    state = LowStateDefault()
    state.mode_machine = mode_machine
    state.motor_state[KNEE].q = KNEE_Q
    state.motor_state[SHOULDER].q = SHOULDER_Q
    return state


def _command(knee: float | None = None, shoulder: float | None = None,
             waist_kp: float = 60.0) -> LowCmd_:
    '''Like the controller's: the waist held at 0 with waist_kp, and the given joints'''
    cmd = LowCmdDefault()
    for i in WAIST:
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].kp = waist_kp
        cmd.motor_cmd[i].kd = 1.5
    for slot, q in ((KNEE, knee), (SHOULDER, shoulder)):
        if q is not None:
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
        last = time.time()
        for publisher, cmd in pairs:
            publisher.Write(cmd)
        time.sleep(0.01)
    return last


def _weight(msg: LowCmd_) -> float:
    return msg.motor_cmd[ARM_SDK_WEIGHT_SLOT].q


def _is_zeroed(msg: LowCmd_) -> bool:
    '''Every motor off, the rt/lowcmd estop command (estop_response zero)'''
    return all(
        m.mode == 0 and m.q == m.dq == m.tau == m.kp == m.kd == 0.0
        for m in msg.motor_cmd[:MOTOR_COUNT]
    )


def _monotonic(values, rising: bool) -> bool:
    pairs = zip(values, values[1:])
    return all(b >= a for a, b in pairs) if rising else all(b <= a for a, b in pairs)


def _assert_on_time(stale: float, timeout: float) -> None:
    '''Check the watchdog fired once its timeout had passed, and at most LATE_S later'''
    assert timeout - 0.02 < stale < timeout + LATE_S, f'fired after {stale:.3f} s'


def _named(events: list, name: str) -> list:
    '''(time, event) of each event called name'''
    return [(t, e) for t, e in events if e['event'] == name]


class _StateStream:
    '''Write rt/lowstate every 5 ms from a thread, except while paused'''
    def __init__(self, topic: str, state: LowState_, paused: bool) -> None:
        self.state = state
        self.last_write: float | None = None
        self._paused = threading.Event()
        if paused:
            self._paused.set()
        self._stop = threading.Event()
        self._publisher = ChannelPublisher(topic, LowState_)
        self._publisher.Init()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._paused.set()
        # let a write in progress finish
        time.sleep(0.01)

    def resume(self) -> None:
        self._paused.clear()

    def close(self) -> None:
        self._stop.set()
        self._thread.join()
        self._publisher.Close()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._paused.is_set():
                self.last_write = time.time()
                self._publisher.Write(self.state)
            time.sleep(0.005)


@contextmanager
def _running_layer(config_name: str, stream: bool = True, state: LowState_ | None = None,
                   state_watchdog: bool = False, **sections):
    '''
    Run a SafetyLayer on topics of its own, fed rt/lowstate (state, which the
    test may change, and pause, while it streams); yield its config, the
    (receive time, LowCmd_) it writes, the (time, event) it logs, the layer, the
    state stream, a publisher(topic key) factory and close_publisher(). The
    rt/lowstate watchdog keeps its configured timeout with state_watchdog, else
    HARNESS_STATE_STALE_S. Each keyword argument updates that section of the
    loaded config
    '''
    _init_dds()
    config = load_config(CONFIG_DIR / config_name)
    config['logging']['enabled'] = False
    if not state_watchdog:
        config['control']['state_stale_estop_seconds'] = HARNESS_STATE_STALE_S
    for section, values in sections.items():
        config[section].update(values)
    prefix = f'test_hardening/{next(_runs)}/'
    topics = {key: prefix + name for key, name in config['topics'].items()}
    config['topics'] = topics

    received = []
    events = []
    sub = ChannelSubscriber(topics['low_cmd_out'], LowCmd_)
    sub.Init(lambda m: received.append((time.time(), m)), 50)
    publishers = []

    def publisher(key: str) -> ChannelPublisher:
        pub = ChannelPublisher(topics[key], LowCmd_)
        pub.Init()
        publishers.append(pub)
        return pub

    def close_publisher(pub: ChannelPublisher) -> None:
        # the SDK's Close() fails on a second call
        publishers.remove(pub)
        pub.Close()

    layer = SafetyLayer(config)
    log_event = layer._log_event

    def capture(data: dict) -> None:
        events.append((time.time(), dict(data)))
        log_event(data)

    layer._log_event = capture
    threading.Thread(target=layer.start, daemon=True).start()
    states = _StateStream(topics['low_state'], state or _state(), paused=not stream)
    try:
        # the layer's publisher loop starts after its 0.2 s writer setup
        time.sleep(0.5)
        yield SimpleNamespace(config=config, received=received, events=events,
                              layer=layer, stream=states, publisher=publisher,
                              close_publisher=close_publisher)
    finally:
        # close every channel, so that none can match a later test's
        states.close()
        layer.stop()
        sub.Close()
        for pub in publishers:
            pub.Close()


def _engaged_arm_sdk(run) -> tuple[ChannelPublisher, LowCmd_]:
    '''Engage arm_sdk_mode (weight 1) on the pose rt/lowstate reports'''
    pub = run.publisher('low_cmd_upper_in')
    good = _command(shoulder=SHOULDER_Q)
    _publish_for(3 * RAMP_S, (pub, good))
    assert run.received and _weight(run.received[-1][1]) == 1.0, 'must have engaged'
    return pub, good


def _lowcmd_inputs(run, config_name: str) -> list[ChannelPublisher]:
    '''A publisher per input of split_mode (lower, upper) or full_body_mode'''
    keys = SPLIT if config_name.startswith('split') else FULL
    return [run.publisher(key) for key in keys]


def _lowcmd_pairs(pubs: list, knee: float = KNEE_Q) -> list:
    '''(publisher, command) per input: the knee to the leg input, the shoulder to the upper'''
    if len(pubs) == 2:
        return [(pubs[0], _command(knee=knee)), (pubs[1], _command(shoulder=SHOULDER_Q))]
    return [(pubs[0], _command(knee=knee, shoulder=SHOULDER_Q))]


def test_arm_sdk_engages_only_with_fresh_state_its_mode_machine_and_a_close_command(monkeypatch):
    # every refusal logged, not once a second, to see each phase's reason
    monkeypatch.setattr(safety_layer_module, 'ENGAGE_REFUSAL_LOG_SECONDS', 0.0)
    with _running_layer('arm_sdk.yaml', stream=False, arm_sdk=FAST_RAMPS) as run:
        pub = run.publisher('low_cmd_upper_in')
        good = _command(shoulder=SHOULDER_Q)

        def refusals(seconds: float, cmd: LowCmd_) -> set:
            n = len(run.events)
            _publish_for(seconds, (pub, cmd))
            return {e['reason'] for _, e in run.events[n:] if e['event'] == 'engage_refused'}

        assert refusals(0.3, good) == {'no rt/lowstate received yet'}
        run.stream.state.mode_machine = 6
        run.stream.resume()
        time.sleep(0.1)
        assert refusals(0.3, good) == {'rt/lowstate mode_machine is 6, expected 5'}
        run.stream.state.mode_machine = 5
        time.sleep(0.1)
        far = _command(shoulder=SHOULDER_Q + 0.3)
        assert refusals(0.3, far) == {
            'motor 15 (left_shoulder_pitch_joint) command q 0.500 is 0.300 rad from '
            'rt/lowstate q 0.200 (> 0.15 rad)'
        }
        run.stream.pause()
        time.sleep(0.2)
        stale = refusals(0.3, good)
        assert stale and all(r.startswith('rt/lowstate is ') and r.endswith(' s old (> 0.1 s)')
                             for r in stale)
        assert run.received == [], 'must stay silent until the gate passes'

        run.stream.resume()
        time.sleep(0.1)
        _publish_for(3 * RAMP_S, (pub, good))
        published = list(run.received)
        events = list(run.events)

    weights = [_weight(msg) for _, msg in published]
    assert weights[0] < 0.5 and weights[-1] == 1.0 and _monotonic(weights, rising=True)
    assert all(abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6 for _, msg in published)
    assert len(_named(events, 'arm_sdk_engage')) == 1


def test_engage_refusals_are_logged_at_most_once_a_second():
    with _running_layer('arm_sdk.yaml', state=_state(mode_machine=6)) as run:
        pub = run.publisher('low_cmd_upper_in')
        # about 250 refused commands
        _publish_for(2.5, (pub, _command(shoulder=SHOULDER_Q)))
        published = list(run.received)
        refused = [t for t, _ in _named(run.events, 'engage_refused')]
    assert published == []
    assert len(refused) == 3, refused  # at about 0, 1 and 2 s
    assert all(b - a > 0.99 for a, b in zip(refused, refused[1:]))


@pytest.mark.parametrize('config_name', ['split.yaml', 'full.yaml'])
def test_lowcmd_modes_engage_only_with_every_slot_close_to_rt_lowstate(config_name):
    with _running_layer(config_name) as run:
        pubs = _lowcmd_inputs(run, config_name)
        _publish_for(0.3, *_lowcmd_pairs(pubs, knee=KNEE_Q + 0.3))
        assert run.received == [], 'a leg 0.3 rad from rt/lowstate must not engage'
        reasons = [e['reason'] for _, e in _named(run.events, 'engage_refused')]
        _publish_for(0.3, *_lowcmd_pairs(pubs))
        published = list(run.received)
        events = list(run.events)

    assert reasons and reasons[0].startswith('motor 3 (left_knee_joint) command q 0.800')
    assert published, 'must publish once the command matches rt/lowstate'
    for _, msg in published:
        assert abs(msg.motor_cmd[KNEE].q - KNEE_Q) < 1e-6
        assert abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6
    assert len(_named(events, 'lowcmd_engage')) == 1


def test_arm_sdk_estops_on_stale_state_and_releases_once_it_flows_again():
    with _running_layer('arm_sdk.yaml', state_watchdog=True, arm_sdk=FAST_RAMPS) as run:
        timeout = run.config['control']['state_stale_estop_seconds']
        pub, good = _engaged_arm_sdk(run)
        run.stream.pause()
        last_state = run.stream.last_write
        n = len(run.received)
        # the controller keeps commanding
        _publish_for(0.6, (pub, good))
        held = run.received[n:]
        n = len(run.received)
        run.stream.resume()
        _publish_for(3 * RAMP_S, (pub, good))
        release = run.received[n:]
        events = list(run.events)

    estops = _named(events, 'estop_command_written')
    assert len(estops) == 1 and estops[0][1]['reason'].startswith('rt/lowstate stale for')
    _assert_on_time(estops[0][0] - last_state, timeout)
    # stale: the last good target held at the weight reached, not released blind
    assert held and all(_weight(msg) == 1.0 for _, msg in held)
    weights = [_weight(msg) for _, msg in release]
    assert weights[-1] == 0.0 and _monotonic(weights, rising=False)
    for _, msg in held + release:
        assert abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6
    release_events = [e['event'] for _, e in events if e['event'].startswith('arm_sdk_release')]
    assert release_events == ['arm_sdk_release_held', 'arm_sdk_release_ramping']


def test_arm_sdk_release_holds_mid_ramp_while_state_is_stale_then_resumes():
    release = 1.0
    ramps = {'engage_seconds': RAMP_S, 'release_seconds': release}
    with _running_layer('arm_sdk.yaml', state_watchdog=True, arm_sdk=ramps) as run:
        stale_after = run.config['control']['state_stale_estop_seconds']
        pub, good = _engaged_arm_sdk(run)
        n = len(run.received)
        # a soft waist latches the estop with rt/lowstate flowing: the ramp starts
        pub.Write(_command(shoulder=SHOULDER_Q, waist_kp=30.0))
        _publish_for(0.4, (pub, good))
        run.stream.pause()
        paused = time.time()
        _publish_for(0.5, (pub, good))
        run.stream.resume()
        resumed = time.time()
        _publish_for(release, (pub, good))
        tail = run.received[n:]
        events = list(run.events)

    weights = [_weight(msg) for _, msg in tail]
    assert weights[0] == 1.0 and weights[-1] == 0.0 and _monotonic(weights, rising=False)
    # no jump down either: at 500 Hz the 1 s ramp drops 0.002 a message, and
    # resuming from the ramp's start time would drop about 0.4 at once
    assert max(a - b for a, b in zip(weights, weights[1:])) < 0.1
    # the weight reached is held while rt/lowstate is stale
    held = [_weight(msg) for t, msg in tail if paused + stale_after + LATE_S < t < resumed]
    assert held and len(set(held)) == 1 and 0.2 < held[0] < 0.8, held[:3]
    assert all(abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6 for _, msg in tail)
    release_events = [(t, e['event']) for t, e in events
                      if e['event'].startswith('arm_sdk_release')]
    assert [name for _, name in release_events] == [
        'arm_sdk_release_ramping', 'arm_sdk_release_held', 'arm_sdk_release_ramping'
    ]
    # the ramp itself still takes release_seconds, the hold aside
    hold = release_events[2][0] - release_events[1][0]
    ramp_start = next(t for t, msg in tail if _weight(msg) < 1.0)
    ramp_end = next(t for t, msg in tail if _weight(msg) == 0.0)
    assert release - 0.1 < ramp_end - ramp_start - hold < release + 0.1


@pytest.mark.parametrize('config_name', ['split.yaml', 'full.yaml'])
def test_lowcmd_modes_estop_on_stale_state(config_name):
    with _running_layer(config_name, state_watchdog=True) as run:
        timeout = run.config['control']['state_stale_estop_seconds']
        pairs = _lowcmd_pairs(_lowcmd_inputs(run, config_name))
        _publish_for(0.3, *pairs)
        run.stream.pause()
        last_state = run.stream.last_write
        _publish_for(timeout + LATE_S + 0.1, *pairs)
        published = list(run.received)
        events = list(run.events)

    idx = next((i for i, (_, msg) in enumerate(published) if _is_zeroed(msg)), None)
    assert idx is not None and idx > 0, 'must publish the commands, then estop'
    assert all(_is_zeroed(msg) for _, msg in published[idx:]), 'the estop must latch'
    _assert_on_time(published[idx][0] - last_state, timeout)
    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    assert len(reasons) == 1 and reasons[0].startswith('rt/lowstate stale for')


def test_a_second_writer_latches_the_estop_before_its_command_is_used():
    with _running_layer('arm_sdk.yaml', arm_sdk=FAST_RAMPS) as run:
        first, good = _engaged_arm_sdk(run)
        second = run.publisher('low_cmd_upper_in')
        _publish_for(0.2, (first, good), (second, _command(shoulder=SHOULDER_Q + 0.05)))
        _publish_for(3 * RAMP_S, (first, good))
        published = list(run.received)
        events = list(run.events)

    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    topic = run.config['topics']['low_cmd_upper_in']
    assert reasons == [f'Second writer on {topic} (arm_sdk_mode): two publishers']
    # the second writer's target never reached rt/arm_sdk; the release held the first's
    assert all(abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6 for _, msg in published)
    assert _weight(published[-1][1]) == 0.0


def test_a_restarted_writer_also_latches_the_estop():
    # a controller restarted while the safety layer runs is a new DDS writer
    with _running_layer('arm_sdk.yaml', arm_sdk=FAST_RAMPS) as run:
        first, good = _engaged_arm_sdk(run)
        run.close_publisher(first)
        restarted = run.publisher('low_cmd_upper_in')
        _publish_for(3 * RAMP_S, (restarted, good))
        published = list(run.received)
        events = list(run.events)

    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    topic = run.config['topics']['low_cmd_upper_in']
    assert reasons == [f'Second writer on {topic} (arm_sdk_mode): two publishers']
    assert _weight(published[-1][1]) == 0.0


def test_a_second_writer_on_the_lower_input_latches_the_estop_before_engaging():
    with _running_layer('split.yaml') as run:
        lower, upper = _lowcmd_pairs(_lowcmd_inputs(run, 'split.yaml'))
        second = run.publisher('low_cmd_lower_in')
        # the lower input alone first: split_mode cannot engage without the upper
        _publish_for(0.2, lower, (second, lower[1]))
        _publish_for(0.3, lower, upper)
        published = list(run.received)
        events = list(run.events)

    assert published == [], 'latched before engaging: nothing may be written'
    topic = run.config['topics']['low_cmd_lower_in']
    reasons = [e['reason'] for _, e in _named(events, 'estop_latched_silent')]
    assert reasons == [f'Second writer on {topic} (split_mode_lower): two publishers']


def test_a_command_with_a_soft_waist_latches_the_estop_before_engaging():
    with _running_layer('arm_sdk.yaml') as run:
        pub = run.publisher('low_cmd_upper_in')
        # a stray publisher's default command: kp 0 on the waist
        _publish_for(0.2, (pub, _command(shoulder=SHOULDER_Q, waist_kp=0.0)))
        _publish_for(0.3, (pub, _command(shoulder=SHOULDER_Q)))
        published = list(run.received)
        events = list(run.events)

    assert published == []
    reasons = [e['reason'] for _, e in _named(events, 'estop_latched_silent')]
    assert reasons == [
        'Command validation failed on arm_sdk_mode: motor 12 (waist_yaw_joint) kp 0.0 is below '
        'arm_sdk.waist_kp_min 40.0'
    ]


def test_a_command_with_a_soft_waist_hands_back_to_unitree_once_engaged():
    with _running_layer('arm_sdk.yaml', arm_sdk=FAST_RAMPS) as run:
        pub, good = _engaged_arm_sdk(run)
        pub.Write(_command(shoulder=SHOULDER_Q + 0.05, waist_kp=30.0))
        _publish_for(3 * RAMP_S, (pub, good))
        published = list(run.received)
        events = list(run.events)

    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    assert len(reasons) == 1 and reasons[0].endswith('kp 30.0 is below arm_sdk.waist_kp_min 40.0')
    assert all(abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6 for _, msg in published)
    assert all(msg.motor_cmd[12].kp == 60.0 for _, msg in published)
    assert _weight(published[-1][1]) == 0.0


def test_arm_sdk_writes_zeros_on_the_legs_and_reports_clipped_gains():
    cmd = _command(shoulder=SHOULDER_Q)
    for i in range(12):
        leg = cmd.motor_cmd[i]
        leg.mode, leg.q, leg.dq, leg.tau, leg.kp, leg.kd = 1, 0.3, 0.1, 2.0, 50.0, 1.0
    cmd.motor_cmd[SHOULDER].kp = 300.0  # above kp_max (200)
    with _running_layer('arm_sdk.yaml', arm_sdk=FAST_RAMPS) as run:
        pub = run.publisher('low_cmd_upper_in')
        _publish_for(3 * RAMP_S, (pub, cmd))
        published = list(run.received)
        events = list(run.events)

    assert published
    for _, msg in published:
        for leg in msg.motor_cmd[:12]:
            fields = (leg.mode, leg.q, leg.dq, leg.tau, leg.kp, leg.kd)
            assert fields == (0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert msg.motor_cmd[SHOULDER].kp == 200.0
    clipped = [e for _, e in _named(events, 'gain_clipped')]
    assert clipped == [{
        'event': 'gain_clipped', 'source': 'arm_sdk_mode', 'joint': 'left_shoulder_pitch_joint',
        'gain': 'kp', 'commanded': 300.0, 'max': 200.0,
    }]


def _shutdown_while_commanding(run, pairs: list) -> tuple[bool, float, float]:
    '''Call shutdown() while the inputs keep commanding; return (result, start, end)'''
    result = []
    start = time.time()
    worker = threading.Thread(target=lambda: result.append(run.layer.shutdown('test exit')))
    worker.start()
    while worker.is_alive():
        for pub, cmd in pairs:
            pub.Write(cmd)
        time.sleep(0.01)
    return result[0], start, time.time()


@pytest.mark.parametrize('state_flowing', [True, False], ids=['state', 'stale_state'])
def test_shutdown_ramps_the_weight_to_zero_then_stops(state_flowing):
    # stale_state needs the shipped watchdog to latch within its 0.3 s pause
    with _running_layer('arm_sdk.yaml', state_watchdog=not state_flowing,
                        arm_sdk=FAST_RAMPS) as run:
        pub, good = _engaged_arm_sdk(run)
        if not state_flowing:
            # the stale-state estop holds the release; the exit must ramp anyway
            run.stream.pause()
            _publish_for(0.3, (pub, good))
        n = len(run.received)
        finished, start, end = _shutdown_while_commanding(run, [(pub, good)])
        time.sleep(0.2)
        tail = run.received[n:]
        events = list(run.events)

    assert finished
    weights = [_weight(msg) for _, msg in tail]
    assert weights[0] > 0.95 and weights[-1] == 0.0 and _monotonic(weights, rising=False)
    assert all(abs(msg.motor_cmd[SHOULDER].q - SHOULDER_Q) < 1e-6 for _, msg in tail)
    # weight 0 written for EXIT_HOLD_SECONDS, then nothing once shutdown returned
    zero_from = next(t for t, msg in tail if _weight(msg) == 0.0)
    assert tail[-1][0] - zero_from > EXIT_HOLD_SECONDS - 0.02
    assert tail[-1][0] < end + 0.05
    assert RAMP_S < end - start < RAMP_S + EXIT_HOLD_SECONDS + 0.5
    exits = [e for _, e in _named(events, 'exit_requested')]
    assert exits == [{'event': 'exit_requested', 'reason': 'test exit', 'engaged': True}]
    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    if state_flowing:
        assert reasons == ['test exit']
    else:
        assert len(reasons) == 1 and reasons[0].startswith('rt/lowstate stale for')


@pytest.mark.parametrize('config_name', ['split.yaml', 'full.yaml'])
def test_shutdown_writes_the_estop_command_in_lowcmd_modes(config_name):
    with _running_layer(config_name) as run:
        pairs = _lowcmd_pairs(_lowcmd_inputs(run, config_name))
        _publish_for(0.3, *pairs)
        n = len(run.received)
        finished, _, end = _shutdown_while_commanding(run, pairs)
        time.sleep(0.2)
        tail = run.received[n:]
        events = list(run.events)

    assert finished
    idx = next(i for i, (_, msg) in enumerate(tail) if _is_zeroed(msg))
    assert all(_is_zeroed(msg) for _, msg in tail[idx:])
    assert tail[-1][0] - tail[idx][0] > EXIT_HOLD_SECONDS - 0.02
    assert tail[-1][0] < end + 0.05
    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    assert reasons == ['test exit']


def test_shutdown_before_engaging_writes_nothing():
    with _running_layer('arm_sdk.yaml') as run:
        start = time.time()
        assert run.layer.shutdown('test exit')
        elapsed = time.time() - start
        time.sleep(0.1)
        published = list(run.received)
        events = list(run.events)
    assert elapsed < 0.5
    assert published == []
    assert [e for _, e in _named(events, 'exit_requested')] == [
        {'event': 'exit_requested', 'reason': 'test exit', 'engaged': False}
    ]
    assert not [e for _, e in events if e['event'].startswith('estop')]


class _SteppedTime:
    '''
    The time module with time() stepped like NTP or date would: the step lands
    inside a publisher loop iteration, between two of that thread's time()
    calls, where a loop timed on time() would sleep for the size of the step
    '''
    def __init__(self) -> None:
        self._offset = 0.0
        self._step: float | None = None

    def step(self, seconds: float) -> None:
        self._step = seconds

    def time(self) -> float:
        now = time.time() + self._offset
        if self._step is not None and threading.current_thread().name == 'low_cmd_publisher':
            # this call still sees the old time, every later one the stepped time
            self._offset += self._step
            self._step = None
        return now

    def __getattr__(self, name: str):
        return getattr(time, name)


def test_wall_clock_steps_neither_fire_a_watchdog_nor_stall_the_loop(monkeypatch):
    stepped = _SteppedTime()
    monkeypatch.setattr(safety_layer_module, 'time', stepped)
    with _running_layer('arm_sdk.yaml', arm_sdk=FAST_RAMPS) as run:
        pub, good = _engaged_arm_sdk(run)
        n = len(run.received)
        _publish_for(0.5, (pub, good))
        rate = len(run.received) - n
        for step in (-3600.0, 7200.0):  # an hour back, then an hour ahead
            stepped.step(step)
            n = len(run.received)
            _publish_for(0.5, (pub, good))
            assert len(run.received) - n > 0.5 * rate, f'the loop stalled after a {step} s step'
        events = list(run.events)
    assert not [e for _, e in events if e['event'].startswith('estop')]


def test_a_suspend_counts_as_a_gap(monkeypatch):
    jump = []
    monkeypatch.setattr(safety_layer_module, 'watchdog_time', lambda: watchdog_time() + sum(jump))
    with _running_layer('arm_sdk.yaml', state_watchdog=True, arm_sdk=FAST_RAMPS) as run:
        pub, good = _engaged_arm_sdk(run)
        # nothing arrives while the laptop sleeps, and CLOCK_BOOTTIME then
        # jumps by the time slept
        run.stream.pause()
        jump.append(5.0)
        time.sleep(0.05)
        run.stream.resume()
        _publish_for(0.2, (pub, good))
        events = list(run.events)

    reasons = [e['reason'] for _, e in _named(events, 'estop_command_written')]
    assert len(reasons) == 1
    assert reasons[0].startswith(('Upper-body command stale for 5.', 'rt/lowstate stale for 5.'))
