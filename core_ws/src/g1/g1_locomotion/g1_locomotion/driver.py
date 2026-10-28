'''DDS side of the walking node, with no ROS: rt/lowstate in, leg commands out'''

import threading
import time

from cyclonedds.core import InstanceState, ReadCondition, SampleState, ViewState, WaitSet
from cyclonedds.domain import DomainParticipant
from cyclonedds.internal import InvalidSample
from cyclonedds.qos import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_description.layout import MODE_PR

from g1_locomotion.locomotion import WAITING, SourceClock, read_state

# how long the worker waits for rt/lowstate before checking that it is not stale (s)
WATCHDOG_PERIOD = 0.01
# the readers keep only the newest sample: a backlog is dropped inside DDS, and
# nothing is decoded until the worker takes it
NEWEST_ONLY = Qos(Policy.History.KeepLast(1), Policy.Reliability.BestEffort)
ANY_SAMPLE = SampleState.Any | ViewState.Any | InstanceState.Any


class LocomotionDriver:
    '''
    Runs a LocomotionController on Unitree DDS. A worker thread takes the
    newest rt/lowstate sample whenever one arrives and gives it to the
    controller with its tick; samples that arrived while it was busy are gone.
    Each leg target the controller returns is checked for latency
    (LocomotionController.latency_ok, from the sample's DDS write, see
    SourceClock) and written at once to topics['low_cmd']
    (golem_safety's lower-body input, which holds it at 500 Hz): slots 0-11
    with mode 1, dq = tau = 0 and the training gains, every other slot zero.
    topics['upper_cmd'] is only read while waiting to start, to know the arm
    controller is running. Nothing is written on a fault or on shutdown.

    The SDK's subscribers decode every sample in the DDS receive thread, under
    the GIL, so a node that falls behind would first have to decode its
    backlog; the readers here are plain DDS readers on a participant of the
    node's own (domain_id, after ChannelFactoryInitialize). The publisher is
    the SDK's
    '''

    def __init__(self, controller, topics, domain_id):
        self._controller = controller
        # serializes every call into the controller, and its writes
        self._lock = threading.Lock()
        self._closed = False
        self._source_clock = SourceClock()

        self._cmd = LowCmdDefault()
        # golem_safety stamps the G1 profile's mode_pr and mode_machine (0 and 5)
        # again before anything reaches the robot
        self._cmd.mode_pr = MODE_PR
        for i, slot in enumerate(controller.leg_slots):
            motor = self._cmd.motor_cmd[slot]
            motor.mode = 1
            motor.kp = float(controller.kp[i])
            motor.kd = float(controller.kd[i])
        self._crc = CRC()

        self._pub = ChannelPublisher(topics['low_cmd'], LowCmd_)
        self._pub.Init()
        self._participant = DomainParticipant(domain_id)
        self._topics = (Topic(self._participant, topics['low_state'], LowState_),
                        Topic(self._participant, topics['upper_cmd'], LowCmd_))
        self._state_reader = DataReader(self._participant, self._topics[0], NEWEST_ONLY)
        self._upper_reader = DataReader(self._participant, self._topics[1], NEWEST_ONLY)
        self._state_ready = ReadCondition(self._state_reader, ANY_SAMPLE)
        self._upper_ready = ReadCondition(self._upper_reader, ANY_SAMPLE)
        self._waitset = WaitSet(self._participant)
        self._waitset.attach(self._state_ready)
        self._waitset.attach(self._upper_ready)
        self._worker = threading.Thread(target=self._run, name='locomotion', daemon=True)
        self._worker.start()

    @property
    def phase(self):
        with self._lock:
            return self._controller.phase

    @property
    def fault(self):
        with self._lock:
            return self._controller.fault

    @property
    def command(self):
        '''The (vx, vy, yaw) and height the policy got last'''
        with self._lock:
            return self._controller.command

    def set_cmd_vel(self, vx, vy, yaw):
        with self._lock:
            self._controller.set_cmd_vel(vx, vy, yaw, time.monotonic())

    def set_height(self, height):
        with self._lock:
            self._controller.set_height(height)

    def _run(self):
        '''Take the newest samples as they arrive; while none do, check for staleness'''
        while True:
            # the GIL is released while waiting
            self._waitset.wait(duration(seconds=WATCHDOG_PERIOD))
            with self._lock:
                if self._closed:
                    return
                try:
                    self._step()
                except Exception as e:  # noqa: BLE001 - never leave the legs on a broken node
                    self._controller.latch(f'unexpected error: {e!r}')

    def _step(self):
        now = time.monotonic()
        if self._upper_reader is not None:
            if any(not isinstance(m, InvalidSample) for m in self._upper_reader.take(N=1)):
                self._controller.on_upper_cmd(now)
            if self._controller.phase != WAITING:
                # only the start needs the arm controller: stop waking on it
                self._waitset.detach(self._upper_ready)
                self._upper_ready = self._upper_reader = None
        samples = self._state_reader.take(N=1)
        if not samples or isinstance(samples[0], InvalidSample):
            self._controller.check_stale(now)
            return
        msg = samples[0]
        source = msg.sample_info.source_timestamp
        written = self._source_clock.written(source * 1e-9, now) if source > 0 else now
        state = read_state(msg, self._controller.slots)
        targets = self._controller.on_lowstate(state, msg.tick, now, now - written)
        if targets is None or self._closed:
            return
        for i, slot in enumerate(self._controller.leg_slots):
            self._cmd.motor_cmd[slot].q = float(targets[i])
        self._cmd.crc = self._crc.Crc(self._cmd)
        sent = time.monotonic()
        if self._controller.latency_ok(sent - written):
            self._pub.Write(self._cmd)
            # after the write, so the timing log never delays a command
            self._controller.log_timing(sent)

    def shutdown(self):
        '''Stop without writing anything more: golem_safety owns the estop'''
        with self._lock:
            self._closed = True
        self._worker.join(timeout=1.0)
        self._pub.Close()
        # dropping the last references deletes the DDS entities; the participant
        # last, which also deletes anything left under it
        self._waitset = self._state_ready = self._upper_ready = None
        self._state_reader = self._upper_reader = self._topics = None
        self._participant = None
