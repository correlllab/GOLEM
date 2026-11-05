"""Unitree DDS for the simulated H1-2: the topics its own computers serve.

  rt/lowstate  published after every physics step
  rt/lowcmd    subscribed; PD on the 27 body motors

Every motor is off (zero torque) until the first rt/lowcmd frame with a valid
CRC, mode_machine 6, mode_pr 0 and finite values on the 27 body motors; other
frames are dropped. From then on rt/lowcmd drives the body: motor mode 1 turns
a motor's PD on, any other mode turns it off. Until a command drives it the
robot hangs from the sim's elastic band (h12_mujoco).

Every torque is recomputed every physics step from the latched command and the
current state (write_ctrl), which emulates the motors' ~1 kHz servo loop.
Computing it only per frame (50 Hz) holds a stale torque for ~20 ms, so the kd
damping lags the true velocity and FAME oscillates and topples ~2.8 s after
standing free. rt/lowcmd silent for GOLEM_CMD_TIMEOUT (0.5 s of sim time, so a
sim running slower than real time does not trip it) turns every motor off until
frames return: the robot goes limp and falls, which is preferable to holding a
snapshot of a pose mid-fall against contact dynamics. The slot layout and
header values match golem_safety's H1-2 profile
(core_ws/src/golem_safety/robots/h12.yaml).
"""
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

import unitree_common
from unitree_common import (MODE_PR, RateLimitedWarnings, domain_id, env_seconds, fast_crc, log,
                            parse_motor_cmds, rpy_from_quat)

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
# 27 motors on the main body, in golem_safety's H1-2 slot order (sim_names.ROS_MOTOR_ORDER)
MOTOR_NUM = 27
# what the real H1-2 reports in every LowState_ (bags of real runs); the IDL default is 0
MODE_MACHINE = 6
# Sim seconds without a frame before rt/lowcmd, once it has driven the motors,
# turns every motor off
LOWCMD_TIMEOUT = env_seconds('GOLEM_CMD_TIMEOUT', 0.5)


def lowcmd_error(msg, crc):
    """Why an rt/lowcmd frame must be dropped, or None to accept it"""
    return unitree_common.lowcmd_error(msg, crc, MODE_MACHINE, 'H1-2')


def lowcmd_motor_on(mode):
    """rt/lowcmd: motor mode 1 applies the PD, every other mode is off"""
    return mode == 1


class SimInterface:
    """The H1-2's Unitree DDS topics on the robosuite-merged MuJoCo model.

    The resolver (sim_names.NameResolver) maps the 27 motors to their qpos,
    qvel, jointactuatorfrc sensor and actuator, and the IMU sensors, in the
    merged model's prefixed names. The main loop calls write_ctrl() before and
    after_step() after every mj_step, holding the sim lock. The DDS handler only
    validates and latches commands (one reference swap) and never touches MjData.
    """

    def __init__(self, model, data, resolver=None):
        if resolver is None:
            raise ValueError("SimInterface requires a NameResolver")
        self.model = model
        self.data = data
        self.resolver = resolver
        self.have_imu = "imu_quat" in resolver.sensor_adr
        self.lowcmd_timeout = LOWCMD_TIMEOUT
        self._crc = fast_crc()
        # sim time of the last step, for the handler (a float swap, no MjData access)
        self.sim_time = float(data.time)

        # latched command (None until the first valid frame)
        self._lowcmd = None
        # True from the first valid rt/lowcmd frame on; until then every motor is off
        self.commanded = False
        self._lowcmd_timed_out = False
        self._warnings = RateLimitedWarnings()

        ChannelFactoryInitialize(id=domain_id())
        self.low_state = LowState_default()
        self.low_state.mode_pr = MODE_PR
        self.low_state.mode_machine = MODE_MACHINE
        self._low_state_pub = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self._low_state_pub.Init()
        self._lowcmd_sub = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self._lowcmd_sub.Init(self._on_lowcmd, 10)

    # ------------------------------------------------------------------ #
    # DDS handler: validate and latch
    # ------------------------------------------------------------------ #
    def _on_lowcmd(self, msg: LowCmd_):
        error = lowcmd_error(msg, self._crc)
        cmd = None if error else parse_motor_cmds(msg.motor_cmd, range(MOTOR_NUM),
                                                  self.sim_time, lowcmd_motor_on)
        if cmd is None:
            error = error or 'a value is not finite'
            self._warnings.warn(f'lowcmd {error}', f'dropped rt/lowcmd: {error}')
            return
        self._lowcmd = cmd
        if not self.commanded:
            self.commanded = True
            log(f't={cmd.time:.3f}s: first valid rt/lowcmd; it drives all {MOTOR_NUM} body motors')

    # ------------------------------------------------------------------ #
    # main loop (the caller holds the sim lock)
    # ------------------------------------------------------------------ #
    def _lowcmd_torques(self, q, dq, now):
        cmd = self._lowcmd
        if now - cmd.time > self.lowcmd_timeout:
            if not self._lowcmd_timed_out:
                self._lowcmd_timed_out = True
                log(f't={now:.3f}s: rt/lowcmd silent for {self.lowcmd_timeout:.2f}s; '
                    'every motor off (zero torque)')
            return np.zeros_like(q)
        if self._lowcmd_timed_out:
            self._lowcmd_timed_out = False
            log(f't={now:.3f}s: rt/lowcmd back')
        return cmd.torques(q, dq)

    def write_ctrl(self):
        """The torque of every body motor for the next physics step"""
        d = self.data
        r = self.resolver
        if self.commanded:
            d.ctrl[r.motor_ctrl] = self._lowcmd_torques(d.qpos[r.motor_qpos], d.qvel[r.motor_qvel],
                                                        float(d.time))
        else:   # never commanded: every motor off
            d.ctrl[r.motor_ctrl] = 0.0

    def after_step(self):
        """Publish rt/lowstate after a physics step"""
        d = self.data
        r = self.resolver
        sd = d.sensordata
        self.sim_time = float(d.time)
        msg = self.low_state
        msg.tick = int(round(d.time * 1000.0)) & 0xFFFFFFFF  # ms of sim time
        # tau_est from the <jointactuatorfrc> sensor (the torque clamped to the
        # joint's actuatorfrcrange), the real robot's measured torque. The raw,
        # unclamped data.actuator_force would spuriously trip golem_safety's estop.
        q, dq, tau = d.qpos[r.motor_qpos], d.qvel[r.motor_qvel], sd[r.motor_tau]
        for i, motor in enumerate(msg.motor_state[:MOTOR_NUM]):
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.tau_est = float(tau[i])
        if self.have_imu:
            imu = msg.imu_state
            adr = {name: a for name, (a, _dim) in r.sensor_adr.items()}
            quat = [float(v) for v in sd[adr['imu_quat']:adr['imu_quat'] + 4]]  # w, x, y, z
            imu.quaternion = quat
            imu.gyroscope = [float(v) for v in sd[adr['imu_gyro']:adr['imu_gyro'] + 3]]
            imu.accelerometer = [float(v) for v in sd[adr['imu_acc']:adr['imu_acc'] + 3]]
            imu.rpy = rpy_from_quat(*quat)
        msg.crc = self._crc.Crc(msg)
        self._low_state_pub.Write(msg)

    def shutdown(self):
        self._lowcmd_sub.Close()
        self._low_state_pub.Close()
