'''
LowCmdHandler: the controller's command, kept for every body joint and published on the
profile's upper slots to golem_safety's input (topics.low_cmd) by a CommandPublisher.

- Clipping: every q, dq and tau is clipped to the controller config's limits.clip, the q
  again after the follow gate, so every published q is within the clip limits.
- Held joints (profile.held_joints: the H1-2's torso, the G1's waist): every command carries
  their hold angles, never the q the controller passes for them. start() sets the hold at the
  measured angles; it moves only through set_hold(), and controller.hold_targets is where the
  controller's start-up moves it.
- Follow gate (FollowGate, controller.follow_tolerance): until golem_safety is seen relaying
  the commands on topics.safety_out, every upper-body command stays within the tolerance of
  the measured joints, and the hold restarts from the command the gate let through. Between
  control steps a thread re-checks at frequency.check_hz: it opens the gate on a relayed
  sample and keeps the stored command near a drifting pose. Once open, the gate and its
  thread are gone for good. The measured joints are the robot model's state source; without
  one (detached, or before its first sample) nothing is gated: a call that needs the measured
  pose raises StaleStateError, and the gate thread stops, leaving the last command standing.
- No estop: golem_safety's limits and estop are the check. A measured state past a limit
  changes nothing here.
- shutdown() sends one last command and stops publishing: the IK joints mode 0 with zero
  gains and torque, and the held joints the same, except where the profile keeps them
  (shutdown_keeps_held_joints_on_lowcmd, the G1) and golem_safety writes /lowcmd: there they
  keep their hold angle, gains and torque.
'''
import math
import threading
import time
from typing import List, Optional

import numpy as np

from golem_upper_ik.config import SAFETY_OUTPUT_TOPICS
from golem_upper_ik.core.follow_gate import FollowGate, relays_commands
from golem_upper_ik.core.ros_io import CommandPublisher, SafetyOutputSubscriber, StaleStateError
from golem_upper_ik.profile import RobotProfile

__all__ = ['LowCmdHandler']


class LowCmdHandler:
    def __init__(self, node, profile: RobotProfile, config: dict, robot_model):
        '''
        node: the ROS node the publisher and the safety-output subscriber live on
        profile: the robot's profile
        config: its controller config (golem_upper_ik.config.load_controller_config)
        robot_model: the RobotModel whose state source gives the measured state, which the
            follow gate and start() read
        '''
        for what, robot in (('controller config', config['robot']),
                            ('robot model', robot_model.profile.robot)):
            if robot != profile.robot:
                raise ValueError(f'the {what} is for robot {robot}, not {profile.robot}')
        self._node = node
        self._profile = profile
        self._robot_model = robot_model
        count = len(profile.body_joints)
        self._count = count
        self._upper = np.array(profile.upper_slots, dtype=np.int64)
        self._held = np.array([profile.body_joints.index(joint) for joint in profile.held_joints],
                              dtype=np.int64)

        limits = config['limits']
        self._q_clip_limits = np.array(limits['q_clip_limits'], dtype=np.float64)
        self._dq_clip_limits = np.array(limits['dq_clip_limits'], dtype=np.float64)
        self._tau_clip_limits = np.array(limits['tau_clip_limits'], dtype=np.float64)

        # the command of every body joint; only the upper slots are published
        self._q = np.zeros(count, dtype=np.float32)
        self._dq = np.zeros(count, dtype=np.float32)
        self._tau = np.zeros(count, dtype=np.float32)
        self._kp = np.array(config['gains']['kp'], dtype=np.float32)
        self._kd = np.array(config['gains']['kd'], dtype=np.float32)
        targets = config['controller']['hold_targets']
        self._hold_targets = np.array([targets[joint] for joint in profile.held_joints],
                                      dtype=np.float64)
        self._hold = np.zeros(len(self._held))  # start() sets it to the measured angles

        self._safety_output_topic = config['topics']['safety_out']
        # the held joints stay on at shutdown: a walking policy may be on the legs
        self._shutdown_keeps_held = (
            profile.shutdown_keeps_held_joints_on_lowcmd
            and self._safety_output_topic in (SAFETY_OUTPUT_TOPICS['split_mode'],
                                              SAFETY_OUTPUT_TOPICS['full_body_mode']))
        self._weight_slot = None
        if self._safety_output_topic == SAFETY_OUTPUT_TOPICS['arm_sdk_mode']:
            if profile.arm_sdk_weight_slot is None:
                raise ValueError(f'robot {profile.robot} has no arm_sdk weight slot, so its '
                                 f'safety output cannot be {self._safety_output_topic}')
            self._weight_slot = profile.arm_sdk_weight_slot
        self._follow_gate = FollowGate(profile.upper_slots,
                                       config['controller']['follow_tolerance'])
        self._check_dt = 1.0 / float(config['frequency']['check_hz'])

        self._data_lock = threading.Lock()
        self._started = False
        self._shut_down = False
        self._gate_error = None
        self._gate_stop = threading.Event()
        self._gate_thread = None
        self._safety_output_lock = threading.Lock()

        self._publisher = CommandPublisher(
            node, config['topics']['low_cmd'], config['frequency']['pub_hz'], profile)
        self._safety_output = None
        if not self._follow_gate.following:
            self._safety_output = SafetyOutputSubscriber(node, self._safety_output_topic)

    @property
    def q_cmd(self) -> np.ndarray:
        '''The commanded q of every body joint (float32, rad); the upper slots are sent'''
        with self._data_lock:
            return self._q.copy()

    @property
    def hold(self) -> np.ndarray:
        '''The held joints' hold angles, rad, in profile.held_joints order'''
        with self._data_lock:
            return self._hold.copy()

    @property
    def hold_targets(self) -> np.ndarray:
        '''controller.hold_targets, rad, in profile.held_joints order'''
        return self._hold_targets.copy()

    @property
    def following(self) -> bool:
        '''False while the follow gate holds the commands near the measured pose'''
        return self._follow_gate.following

    @property
    def safety_output_topic(self) -> str:
        return self._safety_output_topic

    def _clip_q(self, q, joint_ids=None):
        limits = self._q_clip_limits
        if joint_ids is not None:
            limits = limits[joint_ids]
        return np.clip(q, limits[:, 0], limits[:, 1])

    def _clip_dq(self, dq, joint_ids=None):
        limits = self._dq_clip_limits
        if joint_ids is not None:
            limits = limits[joint_ids]
        return np.clip(dq, -limits, limits)

    def _clip_tau(self, tau, joint_ids=None):
        limits = self._tau_clip_limits
        if joint_ids is not None:
            limits = limits[joint_ids]
        return np.clip(tau, -limits, limits)

    def _values(self, name, values, count):
        '''values as count finite float64 numbers; ValueError otherwise'''
        try:
            values = np.broadcast_to(np.asarray(values, dtype=np.float64), (count,))
        except ValueError:
            raise ValueError(f'the {name} command must hold {count} values, not '
                             f'{np.shape(values)}') from None
        if not np.all(np.isfinite(values)):
            raise ValueError(f'the {name} command is not finite: {values}')
        return values

    def _check_running(self):
        if self._shut_down:
            raise RuntimeError('the command handler is shut down')
        if not self._started:
            raise RuntimeError('the command handler sends nothing before start()')
        if self._gate_error is not None:
            raise RuntimeError(f'the follow gate check failed: {self._gate_error!r}')

    def _measured_q(self):
        '''
        The measured q, from the robot model's state source. StaleStateError without a
        source, or before its first sample: RobotModel then reports the all-zero rest pose,
        and gating toward it would pull every command toward q = 0
        '''
        source = self._robot_model.state_subscriber
        if source is None:
            raise StaleStateError('no measured state: the robot model has no state source')
        if not math.isfinite(source.staleness_ms):
            raise StaleStateError('no measured state: the state source has no sample yet')
        return np.asarray(source.state['q'], dtype=np.float64)

    def _push_locked(self):
        upper = self._upper
        self._publisher.set(self._q[upper], self._dq[upper], self._tau[upper],
                            self._kp[upper], self._kd[upper])

    def _commit_locked(self, q, dq, tau, measured_q):
        '''
        Store and publish q, dq and tau (float64, every body joint, the held joints at their
        hold). While the follow gate is shut, q is first held near measured_q, clipped again,
        and the hold restarts from the result; nothing is stored if it is not finite
        '''
        gated = measured_q is not None and not self._follow_gate.following
        if gated:
            q = self._clip_q(self._follow_gate.limit(q, measured_q))
            if not np.all(np.isfinite(q)):
                raise ValueError('the measured q is not finite, so the follow gate cannot '
                                 'bound the command; the last command stands')
        self._q[:] = q
        self._dq[:] = dq
        self._tau[:] = tau
        if gated:
            self._hold = self._q[self._held].astype(np.float64)
        self._push_locked()

    def start(self, q_init=None, tau=None):
        '''
        Command q_init (default: the measured q) with dq 0 and tau (default 0), the hold at
        q_init's held joints, and start publishing. While the follow gate is shut, q_init is
        held near the measured pose like any command, so the measured pose is needed:
        StaleStateError without one. ValueError if q_init is not finite. Either way nothing
        is published
        '''
        with self._data_lock:
            if self._shut_down or self._started:
                raise RuntimeError('the command handler starts once')
        gated = not self._follow_gate.following
        measured_q = self._measured_q() if gated or q_init is None else None
        q_init = self._values('initial q', measured_q if q_init is None else q_init,
                              self._count)
        tau = self._values('initial tau', 0.0 if tau is None else tau, self._count)
        with self._data_lock:
            if self._shut_down or self._started:
                raise RuntimeError('the command handler starts once')
            q = self._clip_q(q_init)
            self._hold = q[self._held].astype(np.float32).astype(np.float64)
            self._commit_locked(q, np.zeros(self._count), self._clip_tau(tau),
                                measured_q if gated else None)
            self._started = True
        if not self._follow_gate.following:
            self._gate_thread = threading.Thread(
                target=self._check_follow_gate, name='follow_gate', daemon=True)
            self._gate_thread.start()

    def set_joint_commands(self,
                           q: Optional[np.ndarray] = None,
                           dq: Optional[np.ndarray] = None,
                           tau: Optional[np.ndarray] = None,
                           joint_ids: Optional[List[int]] = None):
        '''
        Set the command of every body joint, or of joint_ids, atomically: clipped, the held
        joints at their hold, and, while the follow gate is shut and q is given, held near
        the measured pose. A q for a held joint named in joint_ids, or a non-finite value,
        raises ValueError and changes nothing; without a measured pose to gate a q against,
        StaleStateError
        '''
        self._check_running()
        if joint_ids is not None:
            joint_ids = np.asarray(joint_ids, dtype=np.int64).reshape(-1)
            held = sorted(set(joint_ids.tolist()) & set(self._held.tolist()))
            if held and q is not None:
                names = ', '.join(self._profile.body_joints[slot] for slot in held)
                raise ValueError(f'{names}: held joints move only through set_hold')
        count = self._count if joint_ids is None else len(joint_ids)
        given = {name: self._values(name, values, count)
                 for name, values in (('q', q), ('dq', dq), ('tau', tau)) if values is not None}
        # read before taking the data lock; needed only while the gate holds the commands
        measured_q = None if self.following or q is None else self._measured_q()

        with self._data_lock:
            self._check_running()
            ids = slice(None) if joint_ids is None else joint_ids
            new_q = self._q.astype(np.float64)
            new_dq = self._dq.astype(np.float64)
            new_tau = self._tau.astype(np.float64)
            if 'q' in given:
                new_q[ids] = self._clip_q(given['q'], joint_ids)
                new_q[self._held] = self._hold
            if 'dq' in given:
                new_dq[ids] = self._clip_dq(given['dq'], joint_ids)
            if 'tau' in given:
                new_tau[ids] = self._clip_tau(given['tau'], joint_ids)
            self._commit_locked(new_q, new_dq, new_tau, measured_q)

    def set_hold(self, q_held):
        '''
        Hold the held joints at q_held (rad, profile.held_joints order), clipped, from this
        command on; while the follow gate is shut, near the measured pose
        '''
        self._check_running()
        q_held = self._values('hold', q_held, len(self._held))
        measured_q = None if self.following else self._measured_q()
        with self._data_lock:
            self._check_running()
            self._hold = self._clip_q(q_held, self._held)
            new_q = self._q.astype(np.float64)
            new_q[self._held] = self._hold
            self._commit_locked(new_q, self._dq.astype(np.float64),
                                self._tau.astype(np.float64), measured_q)

    def _check_follow_gate(self):
        '''
        Until the gate opens: open it on a new relayed sample, otherwise keep the stored
        command within tolerance of the measured pose; then stop reading the safety output
        '''
        relayed = None
        try:
            while not self._gate_stop.is_set():
                start = time.monotonic()
                sample = self._safety_output.latest()
                if sample is not None and sample[0] is not relayed:
                    relayed = sample[0]
                    self._open_if_relayed(relayed)
                if self._follow_gate.following:
                    break
                measured_q = self._measured_q()
                with self._data_lock:
                    if not (self._shut_down or self._follow_gate.following):
                        gated = self._clip_q(self._follow_gate.limit(self._q, measured_q))
                        # a non-finite measured pose leaves the last command standing
                        if np.all(np.isfinite(gated)) and not np.array_equal(
                                gated.astype(np.float32), self._q):
                            self._q[:] = gated
                            self._push_locked()
                time.sleep(max(0.0, self._check_dt - (time.monotonic() - start)))
        except Exception as error:
            # nothing is written after an error: the last command stands, and
            # set_joint_commands raises the error from now on
            self._gate_error = error
            self._node.get_logger().error(f'the follow gate check failed: {error!r}')
            return
        if self._follow_gate.following:
            self._close_safety_output()

    def _open_if_relayed(self, msg):
        '''Open the follow gate if msg, a safety output sample, relays the commands'''
        motors = msg.motor_cmd
        count = self._count
        sample_q = np.fromiter((motors[i].q for i in range(count)), np.float64, count)
        sample_kp = np.fromiter((motors[i].kp for i in range(count)), np.float64, count)
        weight = None if self._weight_slot is None else float(motors[self._weight_slot].q)
        gate = self._follow_gate
        with self._data_lock:
            if self._shut_down or not relays_commands(
                    sample_q, sample_kp, weight, self._q, gate.joint_ids, gate.tolerance):
                return
            gate.open()
        at = '' if weight is None else ' at weight 1'
        self._node.get_logger().info(
            f'golem_safety relays the commands on {self._safety_output_topic}{at}; '
            'the follow gate is open')

    def _close_safety_output(self):
        '''Stop reading the safety layer's output (never with the data lock held)'''
        with self._safety_output_lock:
            subscriber, self._safety_output = self._safety_output, None
        if subscriber is not None:
            subscriber.shutdown()

    def shutdown(self):
        '''
        Send the IK joints mode 0 with zero gains and torque (q and dq as they were), and the
        held joints the same unless the profile keeps them on /lowcmd, where their command
        stands; make sure that command went out, and stop publishing.

        Call it before the robot model's state source is detached (RobotModel.shutdown()):
        while the follow gate is shut, a detached source stops the gate's check with an error
        rather than gating toward the model's all-zero rest pose
        '''
        with self._data_lock:
            if self._shut_down:
                return
            self._shut_down = True
        self._gate_stop.set()
        if self._gate_thread is not None and self._gate_thread is not threading.current_thread():
            self._gate_thread.join()
        self._close_safety_output()
        try:
            with self._data_lock:
                if self._started:
                    upper = self._upper
                    off = upper
                    if self._shutdown_keeps_held:
                        off = np.setdiff1d(upper, self._held)
                    self._tau[off] = 0.0
                    self._kp[off] = 0.0
                    self._kd[off] = 0.0
                    self._publisher.set(self._q[upper], self._dq[upper], self._tau[upper],
                                        self._kp[upper], self._kd[upper],
                                        enabled=~np.isin(upper, off))
        finally:
            self._publisher.stop()
