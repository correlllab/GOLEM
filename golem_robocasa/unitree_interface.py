"""Serve the simulated H1-2's Unitree DDS topics, as its own computers do.

  rt/lowstate  published after every physics step
  rt/lowcmd    subscribed; PD on the 27 body motors

Every motor is off (zero torque) until the first rt/lowcmd frame with a valid
CRC, mode_machine 6, mode_pr 0 and finite values on the 27 body motors; other
frames are dropped. From then on rt/lowcmd drives the body: motor mode 1 turns
a motor's PD on, any other mode turns it off. Until a command drives it the
robot hangs from the sim's elastic band (h12_mujoco).

Every torque is recomputed every physics step from the latched command and the
current state (write_ctrl), as the motors' ~1 kHz servo loop does: a torque
held for a whole 50 Hz command frame lags the true velocity in its kd term, and
FAME then oscillates and topples ~2.8 s after standing free.

rt/lowcmd silent for GOLEM_CMD_TIMEOUT (0.5 s of sim time, so a sim running
slower than real time does not trip it) turns every motor off until frames
return: the robot goes limp and falls rather than holding a mid-fall pose
against contact. The command path (validation, latch, timeout) is
unitree_common.LowCmdLatch, shared with the G1's sim. The slot layout and
header values match golem_safety's H1-2 profile
(core_ws/src/golem_safety/robots/h12.yaml).
"""
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from unitree_common import LowCmdLatch, LowStateWriter, domain_id

TOPIC_LOWCMD = 'rt/lowcmd'
TOPIC_LOWSTATE = 'rt/lowstate'
# Body motors, in sim_names.ROS_MOTOR_ORDER.
MOTOR_NUM = 27


class SimInterface(LowCmdLatch):
    """Drive the H1-2's body motors from rt/lowcmd and publish rt/lowstate.

    The main loop calls write_ctrl() before and after_step() after every
    mj_step, holding the sim lock. The DDS handler only validates and latches
    commands (one reference swap) and never touches MjData.
    """

    # The real H1-2's LowState_ mode_machine (recorded from the robot; the IDL default is 0).
    MODE_MACHINE = 6
    ROBOT = 'H1-2'

    @staticmethod
    def lowcmd_motor_on(mode):
        """Return whether an rt/lowcmd motor mode applies the PD: only mode 1 does."""
        return mode == 1

    def __init__(self, model, data, resolver=None):
        """Open the DDS topics on ROS_DOMAIN_ID; resolver is the model's sim_names.NameResolver."""
        if resolver is None:
            raise ValueError("SimInterface requires a NameResolver")
        self.model = model
        self.data = data
        self.resolver = resolver
        self.have_imu = "imu_quat" in resolver.sensor_adr
        # Every motor is off until the first valid rt/lowcmd frame.
        self._init_lowcmd(MOTOR_NUM, float(data.time))
        # The sim's ElasticBand, set by sim_loop for the tests that record the
        # sim from inside its process; this class never reads it.
        self.band = None

        ChannelFactoryInitialize(id=domain_id())
        self._low_state_pub = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self._low_state_pub.Init()
        self._low_state = LowStateWriter(self._low_state_pub, self.MODE_MACHINE, MOTOR_NUM)
        if self.have_imu:
            adr = {name: a for name, (a, _dim) in resolver.sensor_adr.items()}
            self._imu = (slice(adr['imu_quat'], adr['imu_quat'] + 4),   # w, x, y, z
                         slice(adr['imu_gyro'], adr['imu_gyro'] + 3),
                         slice(adr['imu_acc'], adr['imu_acc'] + 3))
        self._lowcmd_sub = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self._lowcmd_sub.Init(self._on_lowcmd, 10)

    def write_ctrl(self):
        """Write every body motor's torque for the next physics step into data.ctrl."""
        d = self.data
        r = self.resolver
        if self.commanded:
            d.ctrl[r.motor_ctrl] = self._lowcmd_torques(d.qpos[r.motor_qpos], d.qvel[r.motor_qvel],
                                                        float(d.time))
        else:
            d.ctrl[r.motor_ctrl] = 0.0

    def after_step(self):
        """Publish rt/lowstate of the state mj_step just produced."""
        d = self.data
        r = self.resolver
        sd = d.sensordata
        self.sim_time = float(d.time)
        tick = int(round(d.time * 1000.0)) & 0xFFFFFFFF  # ms of sim time
        # tau_est is the <jointactuatorfrc> sensor, the torque after the joint's
        # actuatorfrcrange clamp, as the real motor measures it. The unclamped
        # data.actuator_force would trip golem_safety's estop limits.
        q, dq, tau = d.qpos[r.motor_qpos], d.qvel[r.motor_qvel], sd[r.motor_tau]
        imu = [sd[part] for part in self._imu] if self.have_imu else [None] * 3
        self._low_state.write(tick, q, dq, tau, *imu)

    def shutdown(self):
        """Close the DDS topics."""
        self._lowcmd_sub.Close()
        self._low_state_pub.Close()
