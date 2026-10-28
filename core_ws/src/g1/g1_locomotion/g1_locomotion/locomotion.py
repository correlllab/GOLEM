'''
The walking node's logic with no ROS or DDS: loading the policy, the joint
slots, the observed state, the velocity and height commands, and the startup
and fault handling
'''

import collections
import importlib.util
import os
from typing import NamedTuple

import numpy as np

from g1_description.layout import BODY_JOINT_NAMES, SLOT_BY_NAME

# HOMIE was trained at 50 Hz. The node steps on a 20 ms grid of rt/lowstate's
# tick, so it follows the robot's clock (the sim's, in the sim) rather than the
# laptop's, however many samples it sees
POLICY_HZ = 50.0
POLICY_DT = 1.0 / POLICY_HZ
# rt/lowstate's tick is a uint32 that wraps
TICK_MODULO = 2 ** 32
# rt/lowstate slots of the policy's 27 joints: legs 0-11, waist yaw 12 and the
# arms 15-28. Waist roll and pitch (13, 14) are not observed
POLICY_SLOTS = tuple(range(0, 13)) + tuple(range(15, 29))
# gravity's z in the pelvis frame is -1 upright. Starting needs it below -0.9
# (tilted less than about 25 deg); above -0.5 (more than 60 deg) is a fall
UPRIGHT_GRAVITY_Z = -0.9
FALLEN_GRAVITY_Z = -0.5
# how far the IMU quaternion's norm may be from 1
QUAT_NORM_TOLERANCE = 0.05
# the arm controller counts as running while its last command is this recent (s)
UPPER_FRESH_SECONDS = 0.5
# the start conditions must hold this long, in rt/lowstate's tick (s): ten of
# the kernel's 100 ms CPU-quota periods, so a container that is being
# throttled shows it before the legs move
STEADY_SECONDS = 1.0
# SourceClock keeps the fastest delivery over the last WINDOW_BUCKETS buckets
# of BUCKET_SECONDS each (s): about 2 s
BUCKET_SECONDS = 0.5
WINDOW_BUCKETS = 4
# how often the timing (latency, policy step spacing) is logged, s
STATS_SECONDS = 10.0
# a reason to wait is logged when it changes, each at most this often (s)
WAIT_LOG_SECONDS = 5.0

# phases: silent until the start conditions hold, then a ramp of the legs to
# the default pose, then the policy; a fault is silent for good
WAITING = 'waiting'
RAMPING = 'ramping'
WALKING = 'walking'
FAULTED = 'faulted'


def load_policy(policy_dir):
    '''
    HomiePolicy from the policy's homie_policy.py, g1.yaml and deploy.onnx in
    policy_dir (this package's policy/), used in place (nothing is copied)
    '''
    module_path = os.path.join(policy_dir, 'homie_policy.py')
    if not os.path.isfile(module_path):
        raise FileNotFoundError(
            f'{module_path} not found: policy_dir must be the policy directory (g1_locomotion/policy)')
    spec = importlib.util.spec_from_file_location('homie_policy', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # HomiePolicy's session (homie_policy.policy_session) runs on the calling
    # thread: one intra-op and one inter-op thread, no spinning, nothing pinned.
    # onnxruntime's default pool (as wide as the machine, pinned, spinning)
    # burned about two cores at 50 Hz, and its pinned threads could be starved
    # by another process's on the same CPUs, which hung the node's startup
    policy = module.HomiePolicy(os.path.join(policy_dir, 'g1.yaml'))
    # onnxruntime's first run is slow: take it now rather than in the first
    # walking step, then start again from an empty history
    policy.step(np.zeros(3), policy.config['height_cmd'], np.zeros(3), [1.0, 0.0, 0.0, 0.0],
                policy.default_q, np.zeros(len(policy.policy_joints)))
    policy.reset()
    return policy


def policy_slots(joint_names):
    '''The rt/lowstate motor slot of each policy joint, looked up by name in g1_description'''
    unknown = [name for name in joint_names if name not in SLOT_BY_NAME]
    if unknown:
        raise ValueError(f'policy joints not in g1_description: {unknown}')
    slots = tuple(SLOT_BY_NAME[name] for name in joint_names)
    if slots != POLICY_SLOTS:
        raise ValueError(f'policy joints map to slots {list(slots)}, expected 0-12 and 15-28')
    return slots


class State(NamedTuple):
    '''What the policy observes of one rt/lowstate sample'''
    quat_wxyz: np.ndarray  # pelvis orientation, w first: the rt/lowstate IMU
    gyro: np.ndarray  # pelvis angular velocity in the pelvis frame, rad/s
    q: np.ndarray  # positions of the policy joints, rad
    dq: np.ndarray  # velocities of the policy joints, rad/s


def read_state(msg, slots):
    '''State from a unitree_hg LowState_; slots from policy_slots()'''
    motors = msg.motor_state
    return State(
        quat_wxyz=np.array(msg.imu_state.quaternion, dtype=np.float64),
        gyro=np.array(msg.imu_state.gyroscope, dtype=np.float64),
        q=np.array([motors[i].q for i in slots], dtype=np.float64),
        dq=np.array([motors[i].dq for i in slots], dtype=np.float64),
    )


def tick_delta(tick, previous):
    '''tick - previous for rt/lowstate's uint32 tick, across its wrap; negative if it went back'''
    delta = (int(tick) - int(previous)) % TICK_MODULO
    return delta - TICK_MODULO if delta >= TICK_MODULO // 2 else delta


class SourceClock:
    '''
    When each rt/lowstate sample was written, on the node's clock, from its DDS
    source timestamp (the writer's clock, s). The two clocks differ by an
    unknown offset but run at the same rate, real time, in the sim too (its
    writes carry the host's clock, not sim time). So this keeps the smallest
    `received - source` of the last ~2 s, the fastest delivery, and puts each
    sample's write at its source timestamp plus that offset. A latency
    measured from there counts any delivery delay beyond the fastest one, the
    time the sample waited for the node and the node's own work
    '''

    def __init__(self):
        # [start, smallest offset] per bucket, newest last
        self._buckets = collections.deque(maxlen=WINDOW_BUCKETS)

    def written(self, source, received):
        '''The node-clock time a sample with this source timestamp was written; received is now'''
        offset = received - source
        if self._buckets and received - self._buckets[-1][0] < BUCKET_SECONDS:
            self._buckets[-1][1] = min(self._buckets[-1][1], offset)
        else:
            self._buckets.append([received, offset])
        return source + min(bucket[1] for bucket in self._buckets)


def gravity_z(quat_wxyz):
    '''z of gravity in the pelvis frame, -1 upright (homie_policy's get_gravity_orientation)'''
    w, x, y, z = quat_wxyz
    return -(w * w - x * x - y * y + z * z)


class CommandFilter:
    '''
    The velocity and height commands the policy gets: clipped to the trained
    ranges, then to the deployment limits, and moved toward the request by at
    most the configured rates. Starts at zero velocity and the standing height
    '''

    def __init__(self, policy, commands):
        self._policy = policy
        self._low = np.array([commands[name][0] for name in ('vx', 'vy', 'yaw')])
        self._high = np.array([commands[name][1] for name in ('vx', 'vy', 'yaw')])
        self._max_step = np.array(commands['accel']) * POLICY_DT
        self._max_height_step = commands['height_rate'] * POLICY_DT
        self.cmd = np.zeros(3)
        self.height = float(policy.config['height_cmd'])

    def update(self, cmd, height):
        '''One policy step toward cmd (vx, vy, yaw) and height; returns the new (cmd, height)'''
        cmd, height = self._policy.clip_command(cmd, height)
        cmd = np.clip(np.asarray(cmd, dtype=np.float64), self._low, self._high)
        self.cmd = self.cmd + np.clip(cmd - self.cmd, -self._max_step, self._max_step)
        step = min(max(height - self.height, -self._max_height_step), self._max_height_step)
        self.height += step
        return self.cmd.copy(), self.height


class LocomotionController:
    '''
    HOMIE on the legs, fed the newest rt/lowstate sample with its tick. Not
    thread-safe: the caller serializes the calls. The ramp and the policy step
    on a grid of ticks one policy period (20 ms of robot time) apart, from the
    first ramp step: a step is due at the first sample at or past its grid
    tick, and the next one at the first grid tick at least half a period after
    that sample. Samples the caller drops never change the rate: 50 Hz of
    robot time on average, steps never closer than 10 ms.

    - waiting: silent until rt/lowstate has been steady, on time (no sample
      older than half faults.max_latency when taken, no gap over that), finite and
      upright for STEADY_SECONDS, its tick has kept to the wall clock within
      startup.tick_rate over that time (a wrong tick_seconds would change the
      policy's rate), and the arm controller is commanding (golem_safety's
      split mode starts on the first leg command only if the arms are already
      commanded)
    - ramping: the legs move from their measured positions to the policy's
      default pose over startup.ramp_seconds
    - walking: the policy runs, starting at zero velocity and the standing
      height, with an empty history
    - faulted: once the legs are commanded, a stale, non-finite or fallen
      state, a tick that goes back, a bad IMU quaternion, a policy step that
      fails, or a command later than faults.max_latency (latency_ok) stops all
      output for good. golem_safety then keeps relaying the last leg command,
      which does not balance the robot, and reports the quiet input; it does
      not estop on it. The node never sends a zero-gain or damping command
      itself

    log(level, message) reports phase changes, why it is waiting, faults and,
    every STATS_SECONDS (log_timing), the timing; level is 'info', 'warn' or
    'error'. The `now` arguments are one monotonic clock, in s
    '''

    def __init__(self, policy, config, log):
        self.policy = policy
        self.slots = policy_slots(policy.policy_joints)
        self.leg_slots = self.slots[:len(policy.leg_joints)]
        self.kp, self.kd = policy.gains(policy.leg_joints)
        self._default = np.asarray(policy.default_angles, dtype=np.float64)
        self._ramp_steps = max(1, int(round(config['startup']['ramp_seconds'] * POLICY_HZ)))
        self._tick_seconds = config['tick_seconds']
        # ticks per policy step (20 for a tick of 1 ms)
        self._period_ticks = int(round(POLICY_DT / self._tick_seconds))
        self._steady_ticks = int(round(STEADY_SECONDS / self._tick_seconds))
        self._tick_rate = config['startup']['tick_rate']
        self._max_gap = config['faults']['max_state_gap']
        self._max_latency = config['faults']['max_latency']
        self._cmd_timeout = config['commands']['timeout']
        self._upper_topic = config['topics']['upper_cmd']
        self._filter = CommandFilter(policy, config['commands'])
        self._log = log

        self.phase = WAITING
        self.fault = None
        self._last_tick = None  # the newest tick taken
        self._last_sample = None  # when it was taken
        # tick from which every sample was usable, while waiting, and when it was taken
        self._good_since = self._good_wall = None
        self._due_tick = None  # grid tick of the next ramp or policy step
        self._step_tick = None  # tick of the last one
        self._wait_reason = None
        self._wait_logged = {}  # when each kind of reason to wait was last logged
        # for the timing log: latencies (s) and policy step spacing (ticks)
        self._latencies = []
        self._step_gaps = []
        self._stats_start = None
        self._upper_time = None
        self._cmd_vel = np.zeros(3)
        self._cmd_vel_time = None
        self._height = self._filter.height
        self._ramp_start = None
        self._ramp_step = 0

    @property
    def command(self):
        '''The (vx, vy, yaw) and height the policy got last'''
        return self._filter.cmd.copy(), self._filter.height

    def on_upper_cmd(self, now):
        '''The arm controller commanded (a sample on topics.upper_cmd) at now (s)'''
        self._upper_time = now

    def set_cmd_vel(self, vx, vy, yaw, now):
        '''A /cmd_vel request in the base frame (m/s, m/s, rad/s), received at now (s)'''
        cmd = np.array([vx, vy, yaw], dtype=np.float64)
        if not np.all(np.isfinite(cmd)):
            self._log('warn', f'/cmd_vel is not finite ({cmd.tolist()}): stopping')
            cmd = np.zeros(3)
        self._cmd_vel, self._cmd_vel_time = cmd, now

    def set_height(self, height):
        '''A /locomotion/height request, the pelvis height target (m)'''
        if not np.isfinite(height):
            self._log('warn', f'ignoring a /locomotion/height that is not finite ({height})')
            return
        self._height = float(height)

    def check_stale(self, now):
        '''Latch a fault if rt/lowstate has gone quiet while the legs are commanded'''
        if self.phase not in (RAMPING, WALKING):
            return
        age = now - self._last_sample
        if age > self._max_gap:
            self._latch(f'rt/lowstate stale: no sample for {age * 1000:.0f} ms '
                        f'(> {self._max_gap * 1000:.0f} ms)')

    def latch(self, reason):
        '''Stop commanding the legs for good'''
        if self.phase != FAULTED:
            self._latch(reason)

    def on_lowstate(self, state, tick, now, age=0.0):
        '''
        Take an rt/lowstate sample with its tick, taken at now (s), age s after
        it was written (SourceClock); returns the 12 leg position targets to
        publish, or None to stay silent. A sample whose tick did not advance
        is ignored
        '''
        delta = None if self._last_tick is None else tick_delta(tick, self._last_tick)
        if delta == 0:
            return None
        gap = None if self._last_sample is None else now - self._last_sample
        went_back = delta is not None and delta < 0
        previous_tick = self._last_tick
        self._last_sample, self._last_tick = now, tick
        if self.phase == FAULTED:
            return None
        if self.phase == WAITING:
            return self._wait(state, tick, now, gap, age, previous_tick if went_back else None)

        # the legs are commanded: every problem is a fault
        if went_back:
            return self._latch(f'rt/lowstate tick went back from {previous_tick} to {tick}')
        if gap is not None and gap > self._max_gap:
            return self._latch(f'rt/lowstate stale: {gap * 1000:.0f} ms between samples '
                               f'(> {self._max_gap * 1000:.0f} ms)')
        problem = self._state_problem(state)
        if problem is not None:
            return self._latch(problem)
        gz = gravity_z(state.quat_wxyz)
        if gz > FALLEN_GRAVITY_Z:
            return self._latch(
                f'fell: gravity z {gz:.2f} in the pelvis frame (> {FALLEN_GRAVITY_Z})')
        if tick_delta(tick, self._due_tick) < 0:
            return None
        self._step_at(tick)
        if self.phase == RAMPING:
            return self._ramp()
        return self._walk(state, now)

    def _step_at(self, tick):
        '''A ramp or policy step on this tick: set the grid tick of the next one'''
        period = self._period_ticks
        if self._due_tick is None:
            # the first step sets the grid
            self._due_tick, late = tick, 0
        else:
            late = tick_delta(tick, self._due_tick)
            self._step_gaps.append(tick_delta(tick, self._step_tick))
        # the first grid tick at least half a period after this one: the
        # smallest k >= 1 with k * period >= late + period / 2
        periods = (2 * late + 3 * period - 1) // (2 * period)
        self._due_tick = (self._due_tick + periods * period) % TICK_MODULO
        self._step_tick = tick

    def latency_ok(self, latency):
        '''
        Check a command just before it is published: latency (s) is the time
        from the rt/lowstate sample it answers (when it was written,
        SourceClock) to now. False, and a latched fault, if it is over
        faults.max_latency
        '''
        if self.phase == FAULTED:
            return False
        if latency > self._max_latency:
            self._latch(f'command late: {latency * 1000:.1f} ms after its rt/lowstate sample '
                        f'(> {self._max_latency * 1000:.0f} ms)')
            return False
        self._latencies.append(latency)
        return True

    def log_timing(self, now):
        '''
        Log the timing (latency_ok's latencies, the policy steps' spacing) once
        every STATS_SECONDS; the caller calls it after a publish, not before
        '''
        if self._stats_start is None:
            self._stats_start = now
            return
        if now - self._stats_start < STATS_SECONDS or not self._latencies:
            return
        latency = np.array(self._latencies) * 1000
        timing = (f'timing over the last {now - self._stats_start:.0f} s: {latency.size} '
                  f'commands, latency median {np.median(latency):.1f} ms, '
                  f'max {latency.max():.1f} ms (limit {self._max_latency * 1000:.0f} ms)')
        if self._step_gaps:
            gaps = np.array(self._step_gaps) * self._tick_seconds * 1000
            timing += (f'; steps {gaps.min():.0f}-{gaps.max():.0f} ms of rt/lowstate time apart, '
                       f'mean {gaps.mean():.1f} ms')
        self._log('info', timing)
        self._latencies, self._step_gaps, self._stats_start = [], [], now

    def _state_problem(self, state):
        '''Why a state cannot be used (values not finite, a bad IMU quaternion), or None'''
        for name, values in zip(State._fields, state):
            bad = np.flatnonzero(~np.isfinite(values))
            if bad.size:
                # joint names for q and dq, indices for the IMU
                where = bad.tolist()
                if name in ('q', 'dq'):
                    where = [BODY_JOINT_NAMES[self.slots[i]] for i in bad]
                return f'rt/lowstate {name} is not finite (at {where})'
        norm = float(np.linalg.norm(state.quat_wxyz))
        if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
            return f'the rt/lowstate IMU quaternion is not a unit quaternion (norm {norm:.3f})'
        return None

    def _wait(self, state, tick, now, gap, age, went_back_from):
        '''Start the ramp once the start conditions have held for STEADY_SECONDS of ticks'''
        reason = self._state_problem(state)
        if reason is None:
            gz = gravity_z(state.quat_wxyz)
            if gz >= UPRIGHT_GRAVITY_Z:
                reason = (f'the robot is not upright (gravity z {gz:.2f} in the pelvis frame, '
                          f'needs < {UPRIGHT_GRAVITY_Z})')
        if reason is None and went_back_from is not None:
            reason = f'rt/lowstate tick went back (from {went_back_from} to {tick})'
        if reason is None and gap is not None and gap > self._max_gap:
            reason = (f'rt/lowstate is not steady ({gap * 1000:.0f} ms between samples, '
                      f'max {self._max_gap * 1000:.0f} ms)')
        # a command answering a sample this late, or due in such a gap, would
        # be late too: the laptop is not keeping up. Half faults.max_latency,
        # so that a start does not run straight into the latency fault
        late = max(age, 0.0 if gap is None else gap)
        if reason is None and late > self._max_latency / 2:
            reason = (f'rt/lowstate reaches the node late ({late * 1000:.0f} ms, '
                      f'max {self._max_latency * 500:.1f} ms to start)')
        if reason is not None:
            self._good_since = None
        elif self._good_since is None:
            self._good_since, self._good_wall = tick, now
        upper_age = None if self._upper_time is None else now - self._upper_time
        if reason is None and (upper_age is None or upper_age > UPPER_FRESH_SECONDS):
            reason = f'no commands from the arm controller on {self._upper_topic}'
        steady = reason is None and tick_delta(tick, self._good_since) >= self._steady_ticks
        if steady:
            # s of tick per s of wall clock over the steady samples
            rate = (tick_delta(tick, self._good_since) * self._tick_seconds /
                    max(now - self._good_wall, 1e-9))
            low, high = self._tick_rate
            if not low <= rate <= high:
                reason = (f"rt/lowstate's tick is off the wall clock ({rate:.2f} s of tick per s, "
                          f'needs {low:.2f}-{high:.2f}; is tick_seconds right?)')
                self._good_since = None
        if reason is not None:
            # logged when the kind of reason (the text before any ' (') changes,
            # so a steady one is logged once, and each kind at most every
            # WAIT_LOG_SECONDS, so two that alternate do not flood the log
            kind = reason.split(' (')[0]
            last = self._wait_logged.get(kind)
            if kind != self._wait_reason and (last is None or now - last >= WAIT_LOG_SECONDS):
                self._wait_logged[kind] = now
                self._log('info', f'waiting to start: {reason}')
            self._wait_reason = kind
            return None
        if not steady:
            return None
        self.phase = RAMPING
        self._step_at(tick)
        self._ramp_start = state.q[:len(self.leg_slots)].copy()
        self._ramp_step = 0
        self._log('info', f'moving the legs from where they are to the default pose over '
                          f'{self._ramp_steps / POLICY_HZ:.1f} s (rt/lowstate\'s tick ran at '
                          f'{rate:.2f} s per s of wall clock)')
        return self._ramp()

    def _ramp(self):
        '''The next target on the line from the measured legs to the default pose'''
        alpha = self._ramp_step / self._ramp_steps
        targets = self._ramp_start + alpha * (self._default - self._ramp_start)
        self._ramp_step += 1
        if self._ramp_step > self._ramp_steps:
            # the default pose was just sent: the policy takes over on the next step
            self.phase = WALKING
            self.policy.reset()
            self._log('info', f'walking: policy started at zero velocity and '
                              f'{self._filter.height:.2f} m height')
        return targets

    def _walk(self, state, now):
        '''One policy step on the filtered command'''
        fresh = self._cmd_vel_time is not None and now - self._cmd_vel_time <= self._cmd_timeout
        cmd, height = self._filter.update(self._cmd_vel if fresh else np.zeros(3), self._height)
        try:
            targets = self.policy.step(cmd, height, state.gyro, state.quat_wxyz, state.q, state.dq)
        except Exception as e:  # noqa: BLE001 - any policy failure stops the legs
            return self._latch(f'policy step failed: {e}')
        targets = np.asarray(targets, dtype=np.float64)
        if targets.shape != (len(self.leg_slots),) or not np.all(np.isfinite(targets)):
            return self._latch(f'policy output is not 12 finite targets: {targets.tolist()}')
        return targets

    def _latch(self, reason):
        self.phase = FAULTED
        self.fault = reason
        self._log('error', f'fault: {reason}. Stopped commanding the legs; golem_safety keeps '
                           'relaying their last command, which does not balance the robot. '
                           'Restart the node to clear')
        return None
