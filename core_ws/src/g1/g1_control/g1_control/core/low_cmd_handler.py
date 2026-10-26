import os
import time
import threading
import numpy as np
from typing import Optional, List

from g1_control.utility.joint_definition import BODY_JOINTS, UPPER_BODY_INDEX
from g1_control.utility.joint_limits import setup_gains
from g1_control.utility.controller_config import load_controller_config
from g1_control.core.robot_model import RobotModel
from g1_control.core.follow_gate import FollowGate, relays_commands
from g1_control.core.channel_interface import LowCmdPublisher, SafetyOutputSubscriber

class LowCmdHandler:
    def __init__(self,
                 robot_model: RobotModel,
                 config: dict=None):
        self._robot_model = robot_model
        self._config = config if config is not None else load_controller_config()
        frequency_cfg = self._config.get('frequency', {})
        checker_dt = 1.0 / float(frequency_cfg.get('check_hz', 1000.0))
        publisher_dt = 1.0 / float(frequency_cfg.get('pub_hz', 500.0))

        self._checker_dt = checker_dt
        limits_cfg = self._config.get('limits', {})

        self._q_clip_limits = limits_cfg.get('q_clip_limits')
        self._dq_clip_limits = limits_cfg.get('dq_clip_limits')
        self._tau_clip_limits = limits_cfg.get('tau_clip_limits')
        self._q_estop_limits = limits_cfg.get('q_estop_limits')
        self._dq_estop_limits = limits_cfg.get('dq_estop_limits')
        self._tau_estop_limits = limits_cfg.get('tau_estop_limits')

        topics_cfg = self._config.get('topics', {})
        low_cmd_topic = topics_cfg.get('low_cmd', 'rt/safety/lowcmd_upper_in')
        clip_limits = None
        if self._q_clip_limits is not None and self._dq_clip_limits is not None and self._tau_clip_limits is not None:
            clip_limits = {
                'position_clip': [
                    {'low': float(limit[0]), 'high': float(limit[1])}
                    for limit in self._q_clip_limits
                ],
                'velocity_clip': self._dq_clip_limits,
                'torque_clip': self._tau_clip_limits,
            }
        # intialize command publisher (to g1_safety_layer's upper-body input)
        self._command_publisher = LowCmdPublisher(
            publisher_dt,
            low_cmd_topic=low_cmd_topic,
            clip_limits=clip_limits,
        )
        setup_gains(self._command_publisher, self._config.get('gains'))
        # set by arm_follow_gate: holds commands near the measured pose until
        # g1_safety_layer's output (read by _safety_output) relays them
        self._follow_gate = None
        self._safety_output = None
        self._safety_output_topic = None
        self._safety_output_lock = threading.Lock()

        # background safety monitoring thread
        self._estopped = False
        self._checking_safety = True
        self._safety_thread = threading.Thread(
            target=self._safety_checker,
            name='safety_thread',
            daemon=True
        )

        # variables recording states
        self._recording = False
        self._com_arr = []
        self._q_arr = []
        self._dq_arr = []
        self._tau_arr = []
        self._q_cmd_arr = []
        self._dq_cmd_arr = []
        self._tau_cmd_arr = []
        self._torque_cmd_arr = []

    @property
    def q_cmd(self):
        with self._command_publisher.data_lock:
            return self._command_publisher.q.copy()

    @property
    def dq_cmd(self):
        with self._command_publisher.data_lock:
            return self._command_publisher.dq.copy()

    @property
    def tau_cmd(self):
        with self._command_publisher.data_lock:
            return self._command_publisher.tau.copy()

    def _clip_q_cmd(self, q, joint_ids=None):
        limits = self._q_clip_limits
        if joint_ids is not None:
            limits = limits[joint_ids]
        return np.clip(q, limits[:, 0], limits[:, 1])

    def _clip_dq_cmd(self, dq, joint_ids=None):
        limits = self._dq_clip_limits
        if joint_ids is not None:
            limits = limits[joint_ids]
        return np.clip(dq, -limits, limits)

    def _clip_tau_cmd(self, tau, joint_ids=None):
        limits = self._tau_clip_limits
        if joint_ids is not None:
            limits = limits[joint_ids]
        return np.clip(tau, -limits, limits)

    def set_joint_commands(self,
                           q: Optional[np.ndarray]=None,
                           dq: Optional[np.ndarray]=None,
                           tau: Optional[np.ndarray]=None,
                           joint_ids: Optional[List[int]]=None):
        '''Set multiple joint commands atomically'''
        # check estop status
        if self._estopped:
            raise Exception('Emergency stop triggered!')
        # read before taking the data lock; only needed while the gate holds commands
        measured_q = None if self.following else self._robot_model.state['q']

        # write to command publisher
        with self._command_publisher.data_lock:
            if joint_ids is None:
                if q is not None:
                    self._command_publisher.q[:] = self._clip_q_cmd(q)
                if dq is not None:
                    self._command_publisher.dq[:] = self._clip_dq_cmd(dq)
                if tau is not None:
                    self._command_publisher.tau[:] = self._clip_tau_cmd(tau)
            else:
                if q is not None:
                    self._command_publisher.q[joint_ids] = self._clip_q_cmd(q, joint_ids)
                if dq is not None:
                    self._command_publisher.dq[joint_ids] = self._clip_dq_cmd(dq, joint_ids)
                if tau is not None:
                    self._command_publisher.tau[joint_ids] = self._clip_tau_cmd(tau, joint_ids)
            if q is not None and measured_q is not None:
                self._gate_command_locked(measured_q)

        # record state if recording is enabled
        if self._recording:
            self._record_state()

    def set_joint_gains(self,
                        kp: Optional[np.ndarray]=None,
                        kd: Optional[np.ndarray]=None,
                        joint_ids: Optional[List[int]]=None):
        '''Set multiple joint gains atomically'''
        with self._command_publisher.data_lock:
            if joint_ids is None:
                if kp is not None:
                    self._command_publisher.kp[:] = kp
                if kd is not None:
                    self._command_publisher.kd[:] = kd
            else:
                if kp is not None:
                    self._command_publisher.kp[joint_ids] = kp
                if kd is not None:
                    self._command_publisher.kd[joint_ids] = kd

    def enable_motors(self, motor_ids, init_q=None):
        if init_q is None:
            init_q = self._robot_model.state['q'][motor_ids]
        self._command_publisher.enable_motors(motor_ids, init_q)

    def arm_follow_gate(self, tolerance, safety_out_topic, joint_ids=UPPER_BODY_INDEX):
        '''
        Hold the joints' commands within tolerance of their measured q until
        g1_safety_layer is seen relaying them on safety_out_topic (read only)
        '''
        self._follow_gate = FollowGate(joint_ids, tolerance)
        self._safety_output_topic = safety_out_topic
        if not self._follow_gate.following:
            self._safety_output = SafetyOutputSubscriber(
                safety_out_topic, self._on_safety_output
            )

    @property
    def following(self):
        '''False while the follow gate holds commands near the measured pose'''
        return self._follow_gate is None or self._follow_gate.following

    @property
    def safety_output_topic(self):
        return self._safety_output_topic

    def _on_safety_output(self, sample_q, sample_kp, weight):
        '''Open the follow gate on a safety layer output sample relaying our commands'''
        gate = self._follow_gate
        if gate is None or gate.following:
            return
        with self._command_publisher.data_lock:
            if self._estopped or not relays_commands(
                sample_q, sample_kp, weight, self._command_publisher.q,
                gate.joint_ids, gate.tolerance
            ):
                return
            gate.open()
        at = '' if weight is None else ' at weight 1'
        print(f'g1_safety_layer relays the commands on {self._safety_output_topic}{at}; '
              'the follow gate is open', flush=True)

    def _close_safety_output(self):
        '''Stop reading the safety layer's output (never with the data lock held)'''
        with self._safety_output_lock:
            subscriber, self._safety_output = self._safety_output, None
        if subscriber is not None:
            subscriber.shutdown()

    def _gate_command_locked(self, measured_q):
        '''Apply the follow gate to the stored q command (data lock held by the caller)'''
        gate = self._follow_gate
        if gate is None or gate.following:
            return
        self._command_publisher.q[:] = gate.limit(self._command_publisher.q, measured_q)

    def start(self):
        self._command_publisher.start()
        self._safety_thread.start()

    def shutdown(self):
        self._close_safety_output()
        self._command_publisher.shutdown()
        self._checking_safety = False

    def estop(self):
        self._command_publisher.estop()
        self._estopped = True

    def start_recording(self, ids, save_path, filename, record_interval=1.0):
        '''Start recording with background saving'''
        self._recording = True
        self._record_ids = ids
        self._record_path = save_path
        self._record_filename = filename
        self._record_interval = record_interval
        self._record_lock = threading.Lock()
        # start recording thread
        threading.Thread(
            target=self._save_recording,
            name='save_recording_thread',
            daemon=True
        ).start()

    def stop_recording(self):
        '''Stop recording'''
        if self._recording:
            self._recording = False
            print(f'Recording stopped; data saved to {self._record_path}/{self._record_filename}.npz')

    def clear_recording(self):
        with self._record_lock:
            self._com_arr = []
            self._q_arr = []
            self._dq_arr = []
            self._tau_arr = []
            self._q_cmd_arr = []
            self._dq_cmd_arr = []
            self._tau_cmd_arr = []
            self._torque_cmd_arr = []

    def _safety_checker(self):
        '''Background thread for safety monitoring'''
        while self._checking_safety and not self._estopped:
            start_time = time.time()
            state = self._robot_model.state
            if not self.following:
                # between control steps too: keep a held command near a drifting
                # pose, until the safety layer relays it (then never again)
                with self._command_publisher.data_lock:
                    self._gate_command_locked(state['q'])
            elif self._safety_output is not None:
                self._close_safety_output()
            for i in UPPER_BODY_INDEX:
                joint_name = BODY_JOINTS[i]
                # check position limits
                if (state['q'][i] < self._q_estop_limits[i][0] or
                    state['q'][i] > self._q_estop_limits[i][1]):
                    print(f'Position limit exceeded on joint {i} {joint_name}: {state["q"][i]:.3f} rad')
                    print(f'q cmd at {i} {self.q_cmd[i]}')
                    self.estop()
                # check velocity limits
                if abs(state['dq'][i]) > self._dq_estop_limits[i]:
                    print(f'Velocity limit exceeded on joint {i} {joint_name}: {state["dq"][i]:.3f} rad/s')
                    self.estop()
                # check torque limits
                if abs(state['tau'][i]) > self._tau_estop_limits[i]:
                    print(f'Torque limit exceeded on joint {i} {joint_name}: {state["tau"][i]:.3f} Nm')
                    self.estop()
            time.sleep(max(0, self._checker_dt - (time.time() - start_time)))

    def _record_state(self):
        if self._recording:
            # record center of mass
            com = self._robot_model.dynamics.get_com()
            # get values for upper body joints only
            state = self._robot_model.state
            q = state['q'][self._record_ids]
            dq = state['dq'][self._record_ids]
            tau = state['tau'][self._record_ids]
            with self._command_publisher.data_lock:
                q_cmd = self._command_publisher.q[self._record_ids]
                dq_cmd = self._command_publisher.dq[self._record_ids]
                tau_cmd = self._command_publisher.tau[self._record_ids]
                kp = self._command_publisher.kp[self._record_ids]
                kd = self._command_publisher.kd[self._record_ids]
            # record with thread-safe access
            with self._record_lock:
                self._com_arr.append(com)
                self._q_arr.append(q)
                self._dq_arr.append(dq)
                self._tau_arr.append(tau)
                self._q_cmd_arr.append(q_cmd)
                self._dq_cmd_arr.append(dq_cmd)
                self._tau_cmd_arr.append(tau_cmd)
                self._torque_cmd_arr.append(
                    tau_cmd + kp * (q_cmd - q) + kd * (dq_cmd - dq)
                )

    def _save_recording(self):
        '''Background worker thread for constant saving'''
        while self._recording:
            start_time = time.time()
            try:
                filename = f'{self._record_path}/{self._record_filename}.npz'
                # save logic moved here
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                # acquire lock to save copy of the arrays
                with self._record_lock:
                    # create copies of the arrays for safe access outside the lock
                    com = np.array(self._com_arr)
                    q = np.array(self._q_arr)
                    dq = np.array(self._dq_arr)
                    tau = np.array(self._tau_arr)
                    q_cmd = np.array(self._q_cmd_arr)
                    dq_cmd = np.array(self._dq_cmd_arr)
                    tau_cmd = np.array(self._tau_cmd_arr)
                    torque_cmd = np.array(self._torque_cmd_arr)
                # save to the file
                np.savez(filename,
                         ids=np.array(self._record_ids),
                         com=com,
                         q=q,
                         dq=dq,
                         tau=tau,
                         q_cmd=q_cmd,
                         dq_cmd=dq_cmd,
                         tau_cmd=tau_cmd,
                         torque_cmd=torque_cmd)
            except Exception as e:
                print(f'Failed to auto-save recording: {str(e)}')

            time.sleep(max(0, self._record_interval - (time.time() - start_time)))
