'''Safety layer for G1 low-level commands, relayed to rt/arm_sdk or rt/lowcmd'''

import copy
import time
import json
import threading
from typing import Any

import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_safety_layer.core import console
from g1_safety_layer.core.clock import watchdog_time
from g1_safety_layer.core.chunk_logger import ChunkLogger
from g1_safety_layer.core.arm_sdk import engage_weight, release_weight, with_weight, zero_legs
from g1_safety_layer.core.joint_limits import (
    ARM_SDK_SLOTS,
    JOINT_NAMES,
    MODE_PR,
    MOTOR_COUNT,
    SPLIT_UPPER_START,
)
from g1_safety_layer.core.safety_checks import (
    CmdValidationError,
    EStopTriggered,
    check_estop_limits,
    check_waist_kp,
    engage_refusal,
    make_estop_cmd,
    clip_low_cmd,
    clipped_gains,
)


LOG_SAMPLES_PER_CHUNK = 1000
LOG_WRITE_HZ = 5.0
LOG_MAX_QUEUE_SIZE = 10000
MODES = ('arm_sdk_mode', 'split_mode', 'full_body_mode')
# Once the upper-body input has sent a command, treat longer-than-this gaps as a
# fault and escalate to estop. Designed to detect frame_task_server crashes
# mid-run; does not fire before the first upper command, but split_mode needs
# both inputs to engage, so it publishes nothing without the upper one.
UPPER_STALE_ESTOP_SECONDS = 2.0
# motor slots each input drives; a kp/kd clipped to kp_max/kd_max on them is
# logged once per joint and gain
GAIN_REPORT_SLOTS = {
    'arm_sdk_mode': ARM_SDK_SLOTS,
    'full_body_mode': range(MOTOR_COUNT),
    'split_mode_lower': range(SPLIT_UPPER_START),
    'split_mode_upper': range(SPLIT_UPPER_START, MOTOR_COUNT),
}
# the config topic each input reads
SOURCE_TOPICS = {
    'arm_sdk_mode': 'low_cmd_upper_in',
    'full_body_mode': 'low_cmd_in',
    'split_mode_lower': 'low_cmd_lower_in',
    'split_mode_upper': 'low_cmd_upper_in',
}
# while waiting to engage, why it does not is logged at most this often
ENGAGE_REFUSAL_LOG_SECONDS = 1.0
# on exit, once the hand-back command (weight 0 on rt/arm_sdk, the estop command
# on rt/lowcmd) is written, keep writing it this long so that one lost packet
# cannot leave the robot on an older command
EXIT_HOLD_SECONDS = 0.1


class SafetyLayer:
    '''Relay low_cmd with validation, clipping, estop, and async logging'''
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._lock = threading.Lock()
        self._estop = False
        self._estop_reason = ''
        # created while holding _lock and emitted only after the first estop
        # command has been written (or, before engaging, found nothing to
        # write), so the report carries the write's timing
        self._pending_estop_report: dict[str, Any] | None = None
        self._running = False
        # parse mode
        self._mode = self._config['mode'].strip().lower()
        # store last command
        self._last_cmd = LowCmdDefault()
        self._last_q_cmd = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        self._last_dq_cmd = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        self._last_tau_cmd = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        self._last_kp_cmd = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        self._last_kd_cmd = np.zeros((MOTOR_COUNT,), dtype=np.float32)
        self._desired_cmd_full = make_estop_cmd(self._last_cmd)
        self._desired_cmd_lower = make_estop_cmd(self._last_cmd)
        self._desired_cmd_upper = make_estop_cmd(self._last_cmd)
        # every watchdog and weight ramp is timed on watchdog_time()
        # (CLOCK_BOOTTIME), so neither a wall-clock step nor a suspend can fake
        # or hide a gap
        self._last_upper_msg_time: float | None = None
        # leg input (low_cmd_lower_in, or low_cmd_in in full_body_mode)
        self._last_lower_msg_time: float | None = None
        self._lower_stale_seconds = float(
            self._config['control']['lower_stale_estop_seconds']
        )
        # latest rt/lowstate: receive time and q, for the engage gate and, once
        # engaged, the state watchdog
        self._last_state_time: float | None = None
        self._state_q: np.ndarray | None = None
        self._state_stale_seconds = float(
            self._config['control']['state_stale_estop_seconds']
        )
        self._last_refusal_log: float | None = None
        # DDS publication handle of each input's first writer; a sample from
        # any other writer latches the estop
        self._writers: dict[str, int] = {}
        # split_mode and full_body_mode stay silent until every input has sent a
        # command, so the zeroed boot command never reaches rt/lowcmd
        self._lowcmd_engaged = False
        # their estop command: every motor off (estop_response.mode zero), or
        # every motor on as a pure damper with this per-joint kd (damp)
        response = self._config['estop_response']
        self._estop_kd = response['kd'] if response['mode'] == 'damp' else None
        self._reported_gain_clips: set[tuple[int, str]] = set()
        # rt/lowstate's mode_machine, echoed into rt/lowcmd headers
        self._mode_machine: int | None = None
        # arm_sdk_mode: waist + arms are shared with Unitree's controller through
        # the weight slot; ramp it in once a command passes the engage gate, out
        # on estop or exit
        self._arm_sdk = self._mode == 'arm_sdk_mode'
        self._engage_seconds = float(self._config['arm_sdk']['engage_seconds'])
        self._release_seconds = float(self._config['arm_sdk']['release_seconds'])
        self._engage_start: float | None = None
        # the release ramp only advances while rt/lowstate flows: start time
        # (None while held), the ramp time already done, and whether it is
        # ramping (None before the first decision), for logging changes
        self._release_start: float | None = None
        self._release_elapsed = 0.0
        self._release_ramping: bool | None = None
        self._release_from = 0.0
        self._weight = 0.0
        # a stray publisher's default command has kp 0 on the waist
        self._waist_kp_min = float(self._config['arm_sdk']['waist_kp_min'])
        # shutdown(): hand control back, then stop
        self._exiting = False
        self._exit_hold_start: float | None = None
        self._exit_done = threading.Event()
        self._stopped = False
        # events decided in the publisher loop, printed after its write
        self._pending_events: list[dict[str, Any]] = []
        # arm_sdk_mode only checks the slots it commands; the legs are Unitree's
        self._check_slots = ARM_SDK_SLOTS if self._arm_sdk else range(MOTOR_COUNT)
        # init publishers and subscribers
        self._crc = CRC()
        self._cmd_sub_full = None
        self._cmd_sub_lower = None
        self._cmd_sub_upper = None
        self._state_sub = ChannelSubscriber(
            self._config['topics']['low_state'], LowState_
        )
        self._cmd_pub = ChannelPublisher(self._config['topics']['low_cmd_out'], LowCmd_)
        self._publish_hz = float(self._config['control']['publish_hz'])
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop, name='low_cmd_publisher', daemon=True
        )
        # setup async logger
        self._logger = None
        if self._config['logging']['enabled']:
            self._logger = ChunkLogger(
                base_dir=self._config['logging']['base_dir'],
                chunk_prefix=self._config['logging']['chunk_prefix'],
                write_hz=LOG_WRITE_HZ,
                max_queue_size=LOG_MAX_QUEUE_SIZE,
                samples_per_chunk=LOG_SAMPLES_PER_CHUNK,
            )
            self._logger.start()

    def start(self) -> None:
        '''Start subscriptions and keep relay alive'''
        self._validate_mode()
        self._running = True
        # initialize subscribers and publishers
        self._init_cmd_subscribers()
        self._state_sub.Init(self._on_low_state, 10)
        self._cmd_pub.Init()
        self._publisher_thread.start()
        # successfully started, log event
        started = {'event': 'relay_started', 'mode': self._mode}
        if self._logger:
            started['log_dir'] = self._config['logging']['base_dir']
        self._log_event(started)

        while self._running:
            # a dead publisher loop writes nothing: fail instead of idling
            if not self._publisher_thread.is_alive() and self._running:
                raise RuntimeError('the publisher loop stopped unexpectedly')
            time.sleep(0.2)

    def shutdown(self, reason: str) -> bool:
        '''
        Hand control back, then stop(); return False if that did not finish in
        time. In arm_sdk_mode the last good command is held while the weight
        ramps to 0 over release_seconds, even with rt/lowstate stale: nothing
        is published after this, and a ramp that might be lost is no worse than
        stopping at the current weight. split_mode and full_body_mode write
        the estop command. Either way it is written for EXIT_HOLD_SECONDS; a
        layer that never engaged writes nothing
        '''
        with self._lock:
            first = not self._exiting
            self._exiting = True
            engaged = self._engaged_locked()
            if first and engaged:
                self._trigger_estop(reason)
        if first:
            self._log_event({'event': 'exit_requested', 'reason': reason, 'engaged': engaged})
        finished = True
        if self._publisher_thread.is_alive():
            timeout = self._release_seconds + EXIT_HOLD_SECONDS + 1.0
            finished = self._exit_done.wait(timeout=timeout)
            if not finished:
                self._log_event({'event': 'exit_hand_back_timeout', 'seconds': timeout})
        self.stop()
        return finished

    def stop(self) -> None:
        '''Stop relay at once, without handing control back (see shutdown), and flush logger'''
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        if self._publisher_thread.ident is not None:
            self._publisher_thread.join(timeout=1.0)
        self._close_cmd_subscribers()
        self._state_sub.Close()
        self._cmd_pub.Close()
        self._log_event({'event': 'relay_stopped'})
        if self._logger:
            self._logger.stop()
        # every event is out before stop() returns, unless stdout is stalled
        console.drain(console.EXIT_DRAIN_SECONDS)

    def _validate_mode(self) -> None:
        '''Validate config mode and raise if unsupported'''
        if self._mode in MODES:
            return
        mode_name = self._config['mode']
        raise ValueError(f'Unknown mode: {mode_name}')

    def _init_cmd_subscribers(self) -> None:
        '''Init low_cmd subscribers based on configured mode'''
        if self._mode == 'full_body_mode':
            self._cmd_sub_full = ChannelSubscriber(
                self._config['topics']['low_cmd_in'], LowCmd_
            )
            self._cmd_sub_full.Init(self._on_low_cmd_full, 10)
        elif self._arm_sdk:
            self._cmd_sub_upper = ChannelSubscriber(
                self._config['topics']['low_cmd_upper_in'], LowCmd_
            )
            self._cmd_sub_upper.Init(self._on_low_cmd_upper, 10)
        # split mode
        else:
            self._cmd_sub_lower = ChannelSubscriber(
                self._config['topics']['low_cmd_lower_in'], LowCmd_
            )
            self._cmd_sub_upper = ChannelSubscriber(
                self._config['topics']['low_cmd_upper_in'], LowCmd_
            )
            self._cmd_sub_lower.Init(self._on_low_cmd_lower, 10)
            self._cmd_sub_upper.Init(self._on_low_cmd_upper, 10)

    def _close_cmd_subscribers(self) -> None:
        '''Close any active low_cmd subscribers'''
        if self._cmd_sub_full is not None:
            self._cmd_sub_full.Close()
            self._cmd_sub_full = None
        if self._cmd_sub_lower is not None:
            self._cmd_sub_lower.Close()
            self._cmd_sub_lower = None
        if self._cmd_sub_upper is not None:
            self._cmd_sub_upper.Close()
            self._cmd_sub_upper = None

    def _clip_cmd_or_estop(self, msg: LowCmd_, source: str) -> LowCmd_ | None:
        '''
        Clip incoming command and trigger estop on validation failures or a
        second writer (lock acquired in caller method)
        '''
        if not self._check_single_writer_locked(msg, source):
            return None
        self._last_cmd = msg
        if self._estop:
            return None

        try:
            out = clip_low_cmd(msg, self._config['limits'])
            if self._arm_sdk:
                check_waist_kp(out, self._waist_kp_min)
        except CmdValidationError as e:
            self._trigger_estop(f'Command validation failed on {source}: {e}')
            return None
        self._report_clipped_gains_locked(msg, source)
        return out

    def _check_single_writer_locked(self, msg: LowCmd_, source: str) -> bool:
        '''
        Latch the estop and reject msg if it comes from another DDS writer than
        this input's first one, so two publishers never interleave commands
        (lock acquired in caller method)
        '''
        # cyclonedds attaches its SampleInfo to every sample it delivers; the
        # publication handle identifies the writer
        info = getattr(msg, 'sample_info', None)
        handle = getattr(info, 'publication_handle', None)
        if handle is None:
            return True
        first = self._writers.setdefault(source, handle)
        if handle == first:
            return True
        topic = self._config['topics'][SOURCE_TOPICS[source]]
        self._trigger_estop(f'Second writer on {topic} ({source}): two publishers')
        return False

    def _report_clipped_gains_locked(self, msg: LowCmd_, source: str) -> None:
        '''
        Log each kp/kd clipped to kp_max/kd_max once per joint and gain, so a gain
        mismatch with the upstream controller is visible
        (lock acquired in caller method)
        '''
        slots = GAIN_REPORT_SLOTS[source]
        for i, gain, value, limit in clipped_gains(msg, self._config['limits'], slots):
            if (i, gain) in self._reported_gain_clips:
                continue
            self._reported_gain_clips.add((i, gain))
            self._log_event({
                'event': 'gain_clipped',
                'source': source,
                'joint': JOINT_NAMES[i],
                'gain': gain,
                'commanded': value,
                'max': limit,
            })

    def _on_low_cmd_full(self, msg: LowCmd_) -> None:
        '''Update full-body desired command from incoming low_cmd'''
        with self._lock:
            out = self._clip_cmd_or_estop(msg, source='full_body_mode')
            if out is not None:
                self._desired_cmd_full = out
                self._last_lower_msg_time = watchdog_time()
                self._engage_lowcmd_if_ready_locked(self._last_lower_msg_time)

    def _on_low_cmd_lower(self, msg: LowCmd_) -> None:
        '''Update split-mode lower-body desired command from incoming low_cmd'''
        with self._lock:
            out = self._clip_cmd_or_estop(msg, source='split_mode_lower')
            if out is not None:
                self._desired_cmd_lower = out
                self._last_lower_msg_time = watchdog_time()
                self._engage_lowcmd_if_ready_locked(self._last_lower_msg_time)

    def _on_low_cmd_upper(self, msg: LowCmd_) -> None:
        '''Update the upper-body desired command (arm_sdk_mode, split_mode) from low_cmd'''
        source = 'arm_sdk_mode' if self._arm_sdk else 'split_mode_upper'
        with self._lock:
            out = self._clip_cmd_or_estop(msg, source=source)
            if out is not None:
                if self._arm_sdk:
                    # the legs are Unitree's: rt/arm_sdk carries zeros there
                    zero_legs(out)
                self._desired_cmd_upper = out
                self._last_upper_msg_time = watchdog_time()
                if self._arm_sdk:
                    self._engage_arm_sdk_if_ready_locked(self._last_upper_msg_time)
                else:
                    self._engage_lowcmd_if_ready_locked(self._last_upper_msg_time)

    def _engage_arm_sdk_if_ready_locked(self, now: float) -> None:
        '''
        Start the rt/arm_sdk weight ramp once a command passes the engage gate
        (lock acquired in caller method)
        '''
        if self._engage_start is not None or self._exiting:
            return
        refusal = self._engage_refusal_locked(self._desired_cmd_upper, ARM_SDK_SLOTS, now)
        if refusal is not None:
            self._report_engage_refusal_locked(now, refusal)
            return
        self._engage_start = now
        self._log_event({'event': 'arm_sdk_engage'})

    def _engage_lowcmd_if_ready_locked(self, now: float) -> None:
        '''
        Start publishing on rt/lowcmd once every input of split_mode or
        full_body_mode has sent a command and the command to publish passes
        the engage gate on all 29 slots (lock acquired in caller method)
        '''
        if self._arm_sdk or self._lowcmd_engaged or self._exiting:
            return
        if self._last_lower_msg_time is None:
            return
        if self._mode == 'split_mode' and self._last_upper_msg_time is None:
            return
        if self._mode == 'full_body_mode':
            cmd = self._desired_cmd_full
        else:
            cmd = self._build_split_cmd_locked()
        refusal = self._engage_refusal_locked(cmd, range(MOTOR_COUNT), now)
        if refusal is not None:
            self._report_engage_refusal_locked(now, refusal)
            return
        self._lowcmd_engaged = True
        self._log_event({'event': 'lowcmd_engage', 'mode': self._mode})

    def _engage_refusal_locked(self, cmd: LowCmd_, slots, now: float) -> str | None:
        '''Why cmd must not be engaged yet, or None (lock acquired in caller method)'''
        return engage_refusal(
            cmd,
            slots,
            self._state_q,
            self._state_age_locked(now),
            self._mode_machine,
            self._config['engage'],
        )

    def _report_engage_refusal_locked(self, now: float, reason: str) -> None:
        '''
        Log why the layer does not engage, at most once every
        ENGAGE_REFUSAL_LOG_SECONDS (lock acquired in caller method)
        '''
        last = self._last_refusal_log
        if last is not None and now - last < ENGAGE_REFUSAL_LOG_SECONDS:
            return
        self._last_refusal_log = now
        self._log_event({'event': 'engage_refused', 'mode': self._mode, 'reason': reason})

    def _engaged_locked(self) -> bool:
        '''Whether the layer has taken control (lock acquired in caller method)'''
        if self._arm_sdk:
            return self._engage_start is not None
        return self._lowcmd_engaged

    def _state_age_locked(self, now: float) -> float | None:
        '''Seconds since the last rt/lowstate, or None (lock acquired in caller method)'''
        if self._last_state_time is None:
            return None
        return now - self._last_state_time

    def _build_split_cmd_locked(self) -> LowCmd_:
        '''
        Merge split commands into one full-body command
        (lock acquired in caller method)
        '''
        merged = copy.deepcopy(self._desired_cmd_upper)
        # keep waist + arms (12:29) from upper and overwrite legs (0:12) from lower
        for i in range(SPLIT_UPPER_START):
            merged.motor_cmd[i] = copy.deepcopy(self._desired_cmd_lower.motor_cmd[i])
        return merged

    def _get_publish_cmd_locked(self, now: float) -> LowCmd_ | None:
        '''
        Get command to publish in current mode, or None to stay silent
        (lock acquired in caller method)
        '''
        if not self._arm_sdk and not self._lowcmd_engaged:
            # never publish the zeroed boot command: wait for every input
            return None
        if self._mode == 'full_body_mode':
            return copy.deepcopy(self._desired_cmd_full)
        if self._arm_sdk:
            # silent until a command passes the engage gate, so Unitree keeps
            # waist + arms
            if self._engage_start is None:
                return None
            self._weight = engage_weight(now, self._engage_start, self._engage_seconds)
            return with_weight(self._desired_cmd_upper, self._weight)
        return self._build_split_cmd_locked()

    def _get_estop_cmd_locked(self, now: float) -> LowCmd_ | None:
        '''
        Get command to publish while estopped
        (lock acquired in caller method)
        '''
        if not self._arm_sdk:
            # zero or damp every motor; stay silent if we never took control
            if not self._lowcmd_engaged:
                return None
            return make_estop_cmd(self._last_cmd, self._estop_kd)
        # hand waist + arms back to Unitree: hold the last good command while the
        # weight ramps to 0; stay silent if we never took control
        if self._engage_start is None:
            return None
        # the ramp advances only while rt/lowstate flows: with the link down
        # nothing reaches the robot, which would then see the weight jump when
        # it came back; on exit it runs anyway (see shutdown)
        age = self._state_age_locked(now)
        flowing = self._exiting or (age is not None and age <= self._state_stale_seconds)
        if flowing and self._release_start is None:
            self._release_start = now - self._release_elapsed
        elif not flowing and self._release_start is not None:
            self._release_elapsed = now - self._release_start
            self._release_start = None
        ramping = self._release_start is not None
        if ramping != self._release_ramping:
            self._release_ramping = ramping
            self._pending_events.append({
                'event': 'arm_sdk_release_ramping' if ramping else 'arm_sdk_release_held',
                'weight': round(self._weight, 4),
                'state_age': None if age is None else round(age, 4),
            })
        if ramping:
            self._weight = release_weight(
                now, self._release_start, self._release_from, self._release_seconds
            )
        return with_weight(self._desired_cmd_upper, self._weight)

    def _stamp_header_locked(self, msg: LowCmd_) -> None:
        '''Set the rt/lowcmd header fields the G1 checks (lock acquired in caller)'''
        if self._arm_sdk:
            return
        msg.mode_pr = MODE_PR
        if self._mode_machine is not None:
            msg.mode_machine = self._mode_machine

    def _publisher_loop(self) -> None:
        dt = 1.0 / self._publish_hz
        while self._running:
            # watchdog_time(), not time.time(): a wall-clock step back would
            # also stall this loop's sleep by the size of the step
            start_time = watchdog_time()
            with self._lock:
                if self._estop:
                    out = self._get_estop_cmd_locked(start_time)
                else:
                    self._check_upper_watchdog_locked(start_time)
                    self._check_lower_watchdog_locked(start_time)
                    self._check_state_watchdog_locked(start_time)
                    if self._estop:
                        out = self._get_estop_cmd_locked(start_time)
                    else:
                        out = self._get_publish_cmd_locked(start_time)
                if out is not None:
                    self._stamp_header_locked(out)
                # taken together with out, so the report matches this write
                report = self._pending_estop_report
                self._pending_estop_report = None
                events = self._pending_events
                self._pending_events = []
                exit_done = self._exit_finished_locked(start_time, out)
            # nothing to write before engage, estop or not
            action_ms = None if out is None else self._publish_checked(out)
            if report is not None:
                self._log_event(self._complete_estop_report(report, action_ms))
            for event in events:
                self._log_event(event)
            if exit_done:
                self._exit_done.set()
            time.sleep(max(0.0, dt - (watchdog_time() - start_time)))

    def _exit_finished_locked(self, now: float, out: LowCmd_ | None) -> bool:
        '''
        While exiting, whether the hand-back is complete: nothing to hand back,
        or its final command (weight 0, or the estop command) already written
        for EXIT_HOLD_SECONDS once out is written (lock acquired in caller)
        '''
        if not self._exiting:
            return False
        if out is None:
            # never took control
            return True
        final = self._weight == 0.0 if self._arm_sdk else self._estop
        if not final:
            return False
        if self._exit_hold_start is None:
            self._exit_hold_start = now
        return now - self._exit_hold_start >= EXIT_HOLD_SECONDS

    def _check_upper_watchdog_locked(self, now: float) -> None:
        '''
        Escalate to estop if the upper-body input fell silent after its first
        command. Active in split_mode and arm_sdk_mode; never fires before that
        first command (split_mode stays silent without it anyway)
        '''
        if self._mode not in ('split_mode', 'arm_sdk_mode'):
            return
        if self._last_upper_msg_time is None:
            return
        if now - self._last_upper_msg_time > UPPER_STALE_ESTOP_SECONDS:
            self._trigger_estop(
                f'Upper-body command stale for '
                f'{now - self._last_upper_msg_time:.2f}s '
                f'(> {UPPER_STALE_ESTOP_SECONDS}s)'
            )

    def _check_lower_watchdog_locked(self, now: float) -> None:
        '''
        Escalate to estop if the leg input (split_mode lower or full_body_mode)
        fell silent after its first command, instead of replaying its last one
        (lock acquired in caller method)
        '''
        if self._last_lower_msg_time is None:
            return
        stale = now - self._last_lower_msg_time
        if stale > self._lower_stale_seconds:
            name = 'Full-body' if self._mode == 'full_body_mode' else 'Lower-body'
            self._trigger_estop(
                f'{name} command stale for {stale:.2f}s '
                f'(> {self._lower_stale_seconds}s)'
            )

    def _check_state_watchdog_locked(self, now: float) -> None:
        '''
        Once engaged, escalate to estop if rt/lowstate is older than
        state_stale_estop_seconds: the state checks would be blind
        (lock acquired in caller method)
        '''
        if not self._engaged_locked():
            return
        age = self._state_age_locked(now)
        if age is None:
            self._trigger_estop('No rt/lowstate received')
        elif age > self._state_stale_seconds:
            self._trigger_estop(
                f'rt/lowstate stale for {age:.3f}s (> {self._state_stale_seconds}s)'
            )

    def _on_low_state(self, msg: LowState_) -> None:
        '''Check state limits for estop and log sample'''
        received = watchdog_time()
        state_q = np.asarray(
            [float(msg.motor_state[i].q) for i in range(MOTOR_COUNT)], dtype=np.float64
        )
        diagnostic: dict[str, np.ndarray] | None = None
        with self._lock:
            self._mode_machine = int(msg.mode_machine)
            # kept while estopped too: the arm_sdk_mode release ramp waits for it
            self._last_state_time = received
            self._state_q = state_q
            if self._estop:
                return
            # nothing of ours to guard until we engage: in arm_sdk_mode Unitree
            # drives waist + arms until then, and split_mode and full_body_mode
            # write nothing (a limp joint past its limit must not latch)
            guarding = self._engaged_locked()
            try:
                if guarding:
                    check_estop_limits(msg, self._config['limits'], self._check_slots)
            except EStopTriggered as e:
                q = np.asarray(
                    [float(msg.motor_state[i].q) for i in range(MOTOR_COUNT)],
                    dtype=np.float32,
                )
                dq = np.asarray(
                    [float(msg.motor_state[i].dq) for i in range(MOTOR_COUNT)],
                    dtype=np.float32,
                )
                ddq = np.asarray(
                    [float(msg.motor_state[i].ddq) for i in range(MOTOR_COUNT)],
                    dtype=np.float32,
                )
                diagnostic = {
                    'q_cmd': self._last_q_cmd.copy(),
                    'dq_cmd': self._last_dq_cmd.copy(),
                    'tau_cmd': self._last_tau_cmd.copy(),
                    'q': q,
                    'dq': dq,
                    'ddq': ddq,
                }
                self._trigger_estop(str(e))

        if diagnostic is not None:
            for name, values in diagnostic.items():
                console.write_line(f'{name}: {values}')
        self._log_sample(msg)

    def _publish_checked(self, msg: LowCmd_) -> int:
        # write message
        msg.crc = self._crc.Crc(msg)
        self._cmd_pub.Write(msg)
        action_ms = int(time.time() * 1000)
        # write local copy for logging
        self._last_q_cmd = np.asarray(
            [float(msg.motor_cmd[i].q) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        self._last_dq_cmd = np.asarray(
            [float(msg.motor_cmd[i].dq) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        self._last_tau_cmd = np.asarray(
            [float(msg.motor_cmd[i].tau) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        self._last_kp_cmd = np.asarray(
            [float(msg.motor_cmd[i].kp) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        self._last_kd_cmd = np.asarray(
            [float(msg.motor_cmd[i].kd) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        return action_ms

    def _trigger_estop(
        self, reason: str, trigger_stamp_ms: int | None = None
    ) -> None:
        '''Set estop state while holding _lock; defer its log until command write'''
        if self._estop:
            return
        self._estop = True
        self._estop_reason = reason
        decision_ms = int(time.time() * 1000)
        self._pending_estop_report = {
            'event': 'estop_command_written',
            'reason': reason,
            'trigger_stamp_ms': trigger_stamp_ms,
            'decision_ms': decision_ms,
        }
        if self._arm_sdk:
            # keep the last good upper command to hold during the release ramp,
            # which the publisher loop starts once rt/lowstate flows
            self._release_start = None
            self._release_elapsed = 0.0
            self._release_from = self._weight
            return
        # overwrite desired commands with estop command
        self._desired_cmd_full = make_estop_cmd(self._last_cmd, self._estop_kd)
        self._desired_cmd_lower = make_estop_cmd(self._last_cmd, self._estop_kd)
        self._desired_cmd_upper = make_estop_cmd(self._last_cmd, self._estop_kd)

    def _complete_estop_report(
        self, report: dict[str, Any], action_ms: int | None
    ) -> dict[str, Any]:
        '''
        Complete the one-time estop report with the write timing, or rename it
        estop_latched_silent when nothing was written (action_ms None: the layer
        never engaged)
        '''
        if action_ms is None:
            report['event'] = 'estop_latched_silent'
            return report

        report['action_ms'] = action_ms
        report['decision_to_action_ms'] = action_ms - report['decision_ms']
        trigger_stamp_ms = report['trigger_stamp_ms']
        if trigger_stamp_ms is not None:
            # this requires the estop publisher and safety host clocks to be
            # synchronized, and a negative value signals clock skew
            report['trigger_to_action_ms'] = action_ms - trigger_stamp_ms
        return report

    def _log_event(self, data: dict[str, Any]) -> None:
        # queued, never written here: a closed, hung-up or stalled stdout must
        # not stop the publisher loop, block under _lock or end the hand-back
        console.write_line(json.dumps(data, sort_keys=True))

    def _log_sample(self, msg: LowState_) -> None:
        '''Build logging dict from low_state message and log via logger'''
        time_stamp = time.time()

        mode = np.asarray(
            [int(msg.motor_state[i].mode) for i in range(MOTOR_COUNT)], dtype=np.uint8
        )
        q = np.asarray(
            [float(msg.motor_state[i].q) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        dq = np.asarray(
            [float(msg.motor_state[i].dq) for i in range(MOTOR_COUNT)], dtype=np.float32
        )
        ddq = np.asarray(
            [float(msg.motor_state[i].ddq) for i in range(MOTOR_COUNT)],
            dtype=np.float32,
        )
        tau_est = np.asarray(
            [float(msg.motor_state[i].tau_est) for i in range(MOTOR_COUNT)],
            dtype=np.float32,
        )
        temperature = np.asarray(
            [msg.motor_state[i].temperature for i in range(MOTOR_COUNT)], dtype=np.int16
        )
        vol = np.asarray(
            [float(msg.motor_state[i].vol) for i in range(MOTOR_COUNT)],
            dtype=np.float32,
        )
        sensor = np.asarray(
            [msg.motor_state[i].sensor for i in range(MOTOR_COUNT)], dtype=np.uint32
        )
        motorstate = np.asarray(
            [int(msg.motor_state[i].motorstate) for i in range(MOTOR_COUNT)],
            dtype=np.uint32,
        )
        reserve = np.asarray(
            [msg.motor_state[i].reserve for i in range(MOTOR_COUNT)], dtype=np.uint32
        )

        imu_quaternion = np.asarray(msg.imu_state.quaternion, dtype=np.float32)
        imu_gyroscope = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)
        imu_accelerometer = np.asarray(msg.imu_state.accelerometer, dtype=np.float32)
        imu_rpy = np.asarray(msg.imu_state.rpy, dtype=np.float32)
        imu_temperature = np.int16(msg.imu_state.temperature)

        sample = {
            'time_stamp': time_stamp,
            'mode': mode,
            'q': q,
            'dq': dq,
            'ddq': ddq,
            'tau_est': tau_est,
            'temperature': temperature,
            'vol': vol,
            'sensor': sensor,
            'motorstate': motorstate,
            'reserve': reserve,
            'imu_quaternion': imu_quaternion,
            'imu_gyroscope': imu_gyroscope,
            'imu_accelerometer': imu_accelerometer,
            'imu_rpy': imu_rpy,
            'imu_temperature': imu_temperature,
            'q_cmd': self._last_q_cmd.copy(),
            'dq_cmd': self._last_dq_cmd.copy(),
            'tau_cmd': self._last_tau_cmd.copy(),
            'kp_cmd': self._last_kp_cmd.copy(),
            'kd_cmd': self._last_kd_cmd.copy(),
        }

        if self._logger:
            self._logger.log_sample(sample)
