"""Check the H1-2 RoboCasa sim's Unitree DDS command path (golem_robocasa/unitree_interface.py).

rt/lowcmd frames are validated and latched whole, every motor is off until the
first valid frame, the command timeout runs in sim time inside the physics step,
and rt/lowstate is published after every physics step. The model is a stand-in
for the robot (27 hinge motors and a free-floating IMU body, no kitchen), and
the DDS channels and cyclonedds' ddspy_write are in-process fakes for the
length of each test, so no DDS participant opens. It runs in the standard runner
(tests/robocasa/run_in_container.sh); see test_h12_mode_machine.py for a
throwaway-container command.
"""
import contextlib
import io
import math
import os
import subprocess
import sys
import time
import types
import unittest
from unittest import mock

import mujoco
import numpy as np

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC

import unitree_common
import unitree_interface as ui

N = ui.MOTOR_NUM
CRC_ = CRC()
# rad: the IMU body's orientation, so that rpy is checked against known angles
ROLL, PITCH, YAW = 0.1, -0.2, 0.3


def quat_from_rpy(roll, pitch, yaw):
    """Return the WXYZ quaternion of yaw about z, then pitch about y, then roll about x."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def model_and_resolver():
    """Return a stand-in robot model and its resolver.

    N hinge motors with jointactuatorfrc sensors and a free body carrying the IMU sensors,
    resolved as sim_names.NameResolver resolves the robot.
    """
    # 0.1 m spheres give enough inertia for the tests' gains to stay stable at 2 ms steps.
    bodies = ''.join(f'<body><joint name="j{i}" axis="0 0 1"/><geom size="0.1"/></body>'
                     for i in range(N))
    quat = ' '.join(str(v) for v in quat_from_rpy(ROLL, PITCH, YAW))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002"><flag contact="disable"/></option>'
        f'<worldbody>{bodies}'
        f'<body name="pelvis" quat="{quat}"><freejoint/><geom size="0.05"/><site name="imu"/></body>'
        '</worldbody>'
        '<actuator>' + ''.join(f'<motor joint="j{i}"/>' for i in range(N)) + '</actuator>'
        '<sensor>' + ''.join(f'<jointactuatorfrc joint="j{i}"/>' for i in range(N))
        + '<framequat name="imu_quat" objtype="site" objname="imu"/>'
          '<gyro name="imu_gyro" site="imu"/><accelerometer name="imu_acc" site="imu"/>'
          '</sensor></mujoco>')

    def sensor(name):
        """Return a sensor's (sensordata address, dimension)."""
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return (int(model.sensor_adr[sid]), int(model.sensor_dim[sid]))

    resolver = types.SimpleNamespace(
        motor_qpos=model.jnt_qposadr[:N].copy(), motor_qvel=model.jnt_dofadr[:N].copy(),
        motor_tau=model.sensor_adr[:N].copy(), motor_ctrl=np.arange(N),
        sensor_adr={name: sensor(name) for name in ('imu_quat', 'imu_gyro', 'imu_acc')})
    return model, resolver


class FakePublisher:
    """Stand in for ChannelPublisher, keeping the samples written to its DataWriter.

    It has the private attribute chain of unitree_sdk2py's ChannelPublisher, ending in a
    stand-in DataWriter, so that LowStateWriter writes through fake_ddspy_write.
    """
    by_ref = {}

    def __init__(self, topic, kind):
        """Keep the topic; open nothing."""
        self.topic, self.sent, self.closed = topic, [], False
        writer = types.SimpleNamespace(_ref=len(FakePublisher.by_ref) + 1, _use_version_2=None)
        FakePublisher.by_ref[writer._ref] = self
        self._ChannelPublisher__channel = types.SimpleNamespace(
            _Channel__writer=types.SimpleNamespace(_Writer__writer=writer))

    def Init(self):
        """Do nothing: there is no channel."""

    def Write(self, sample):
        """Fail: the sim writes rt/lowstate's bytes, not SDK samples."""
        raise AssertionError('rt/lowstate took the SDK path')

    def Close(self):
        """Mark the publisher closed."""
        self.closed = True

    def last(self):
        """Return the last sample written, decoded by the SDK's LowState_."""
        return LowState_.deserialize(self.sent[-1])


def fake_ddspy_write(ref, data):
    """Keep the bytes written to a FakePublisher's DataWriter, and report success."""
    FakePublisher.by_ref[ref].sent.append(bytes(data))
    return 0


class FakeSubscriber:
    """Stand in for ChannelSubscriber, keeping the handler so that a test delivers frames through it."""
    by_topic = {}

    def __init__(self, topic, kind):
        """Register the subscriber by topic; open nothing."""
        self.topic, self.handler, self.closed = topic, None, False
        FakeSubscriber.by_topic[topic] = self

    def Init(self, handler, queue_size):
        """Keep the handler."""
        self.handler = handler

    def Close(self):
        """Mark the subscriber closed."""
        self.closed = True


def lowcmd(q=0.3, dq=0.1, kp=40.0, kd=2.0, tau=0.5, mode=1, mode_machine=6, mode_pr=0):
    """Return a LowCmd_ giving every body motor the same command, with a valid CRC."""
    msg = LowCmd_default()
    msg.mode_machine, msg.mode_pr = mode_machine, mode_pr
    for c in msg.motor_cmd[:N]:
        c.mode, c.q, c.dq, c.tau, c.kp, c.kd = mode, q, dq, tau, kp, kd
    msg.crc = CRC_.Crc(msg)
    return msg


class H12UnitreeInterfaceTests(unittest.TestCase):
    """Run SimInterface on the stand-in model with the DDS channels faked."""

    def setUp(self):
        """Fake the DDS channels, make a SimInterface and keep its rt/lowcmd handler."""
        env = mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '199'})  # reaches no network: the fakes below
        env.start()
        self.addCleanup(env.stop)
        FakeSubscriber.by_topic = {}
        FakePublisher.by_ref = {}
        for module, name, fake in ((ui, 'ChannelFactoryInitialize', lambda id: None),
                                   (ui, 'ChannelPublisher', FakePublisher),
                                   (ui, 'ChannelSubscriber', FakeSubscriber),
                                   (unitree_common, 'ddspy_write', fake_ddspy_write)):
            patcher = mock.patch.object(module, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model, self.r = model_and_resolver()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.iface = ui.SimInterface(self.model, self.data, resolver=self.r)
        self.deliver = FakeSubscriber.by_topic['rt/lowcmd'].handler

    def ctrl(self):
        """Return the motors' ctrl (torque, N*m)."""
        return self.data.ctrl[self.r.motor_ctrl]

    def state(self):
        """Return copies of the motors' positions and velocities."""
        return self.data.qpos[self.r.motor_qpos].copy(), self.data.qvel[self.r.motor_qvel].copy()

    def quiet(self, fn, *args):
        """Call fn(*args) with its printed lines captured, and return them."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fn(*args)
        return out.getvalue().splitlines()

    def test_motors_off_until_first_valid_lowcmd(self):
        """Every motor is off until the first valid rt/lowcmd, whatever the sim time."""
        self.data.qpos[self.r.motor_qpos] = np.linspace(-0.3, 0.3, N)
        self.data.qvel[self.r.motor_qvel] = 0.5
        self.data.ctrl[:] = 1.0  # whatever a previous step left
        for t in (0.0, 1.0, 100.0):
            self.data.time = t
            self.iface.write_ctrl()
            self.assertFalse(self.iface.commanded)
            self.assertTrue(np.all(self.ctrl() == 0.0))

    def test_invalid_lowcmd_ignored(self):
        """A bad CRC, the wrong mode_machine, mode_pr 1 or a non-finite value is dropped with a warning."""
        bad_crc = lowcmd()
        bad_crc.crc ^= 1
        bad = {'bad CRC': bad_crc, 'mode_machine 5': lowcmd(mode_machine=5),
               'mode_pr 1': lowcmd(mode_pr=1), 'NaN q': lowcmd(q=math.nan),
               'inf kp': lowcmd(kp=math.inf)}
        lines = []
        for reason, msg in bad.items():
            with self.subTest(reason):
                lines += self.quiet(self.deliver, msg)
                self.iface.write_ctrl()
                self.assertFalse(self.iface.commanded)
                self.assertIsNone(self.iface._lowcmd)
                self.assertTrue(np.all(self.ctrl() == 0.0))
        # One warning per reason; NaN and inf share "a value is not finite".
        self.assertEqual(len(lines), 4, lines)
        self.assertTrue(all('dropped rt/lowcmd' in line for line in lines), lines)
        self.quiet(self.deliver, lowcmd())
        self.assertTrue(self.iface.commanded)

    def test_drop_warnings_rate_limited_per_reason(self):
        """Drop warnings print once per reason per WARN_PERIOD_S, then count what they skipped."""
        bad_crc = lowcmd()
        bad_crc.crc ^= 1
        lines = self.quiet(lambda: [self.deliver(m) for m in [bad_crc] * 5 + [lowcmd(mode_pr=1)] * 3])
        self.assertEqual(len(lines), 2, lines)  # one per reason within WARN_PERIOD_S
        later = time.monotonic() + unitree_common.WARN_PERIOD_S + 1.0
        with mock.patch('time.monotonic', lambda: later):
            lines = self.quiet(self.deliver, bad_crc)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('(5 since the last warning)', lines[0])

    def test_first_valid_lowcmd_drives_motors(self):
        """The first valid frame drives each motor at tau + kp*(q - q_meas) + kd*(dq - dq_meas), unclipped, in mode 1 only."""
        self.data.qpos[self.r.motor_qpos] = np.linspace(-0.3, 0.3, N)
        self.data.qvel[self.r.motor_qvel] = 0.2
        lines = self.quiet(self.deliver, lowcmd(q=0.3, dq=0.1, kp=4000.0, kd=2.0, tau=0.5))
        self.assertTrue(self.iface.commanded)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('first valid rt/lowcmd', lines[0])
        self.iface.write_ctrl()
        q, dq = self.state()
        # No torque limit: kp 4000 over 0.6 rad is 2400 N*m.
        expected = 0.5 + 4000.0 * (0.3 - q) + 2.0 * (0.1 - dq)
        np.testing.assert_allclose(self.ctrl(), expected)
        self.assertGreater(np.max(np.abs(self.ctrl())), 1000.0)
        # Motor mode 1 is on, every other mode off.
        for mode in (0, 2):
            self.quiet(self.deliver, lowcmd(mode=mode))
            self.iface.write_ctrl()
            self.assertTrue(np.all(self.ctrl() == 0.0), mode)

    def test_valid_frame_latched_whole_without_mjdata(self):
        """The handler swaps in a whole new command stamped with the last step's sim time.

        It reads nothing from MjData and never changes a latched command.
        """
        self.iface.sim_time = 1.25
        data, self.iface.data = self.iface.data, None  # any MjData access would raise
        try:
            self.quiet(self.deliver, lowcmd(q=0.3))
            first = self.iface._lowcmd
            self.quiet(self.deliver, lowcmd(q=-0.4))
            second = self.iface._lowcmd
            bad = lowcmd(q=0.9)
            bad.crc ^= 1
            self.quiet(self.deliver, bad)
        finally:
            self.iface.data = data
        self.assertIsNot(first, second)
        self.assertIs(self.iface._lowcmd, second)
        self.assertTrue(np.all(first.q == 0.3) and np.all(second.q == -0.4))
        self.assertEqual((first.time, second.time), (1.25, 1.25))
        with self.assertRaises(AttributeError):  # an immutable tuple
            second.q = first.q

    def test_lowcmd_timeout_turns_motors_off(self):
        """Motors turn off after the command timeout in sim time and on again when commands resume, each logged once."""
        timeout = self.iface.lowcmd_timeout
        self.assertEqual(timeout, ui.LOWCMD_TIMEOUT)
        self.quiet(self.deliver, lowcmd())
        self.data.time = timeout - 0.01  # s of sim time, however long it took on the wall
        self.iface.write_ctrl()
        self.assertTrue(np.any(self.ctrl() != 0.0))
        self.data.time = timeout + 0.01
        lines = self.quiet(lambda: [self.iface.write_ctrl() for _ in range(3)])
        self.assertTrue(np.all(self.ctrl() == 0.0))
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('silent', lines[0])
        self.iface.sim_time = float(self.data.time)
        self.quiet(self.deliver, lowcmd())  # commands resume
        lines = self.quiet(lambda: [self.iface.write_ctrl() for _ in range(3)])
        self.assertTrue(np.any(self.ctrl() != 0.0))
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('back', lines[0])

    def test_timeout_knob_must_be_finite_and_positive(self):
        """GOLEM_CMD_TIMEOUT must be finite and positive, and a bad value stops the sim at import."""
        name = 'GOLEM_CMD_TIMEOUT'
        self.assertEqual(ui.env_seconds(name, 0.5, {}), 0.5)
        self.assertEqual(ui.env_seconds(name, 0.5, {name: ' '}), 0.5)
        self.assertEqual(ui.env_seconds(name, 0.5, {name: '0.25'}), 0.25)
        for bad in ('nan', 'inf', '-inf', '-0.5', '0', 'fast'):
            with self.subTest(bad), self.assertRaises(ValueError) as caught:
                ui.env_seconds(name, 0.5, {name: bad})
            self.assertIn(name, str(caught.exception))
            self.assertIn(repr(bad), str(caught.exception))
        # The sim refuses a bad value when it starts, which is when it imports the interface.
        env = dict(os.environ, GOLEM_CMD_TIMEOUT='nan', PYTHONDONTWRITEBYTECODE='1')
        run = subprocess.run([sys.executable, '-c', 'import unitree_interface'], env=env,
                             cwd=os.path.dirname(os.path.abspath(ui.__file__)),
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('GOLEM_CMD_TIMEOUT', run.stderr)

    def test_lowstate_per_step(self):
        """Each physics step publishes a LowState with a millisecond tick, mode_machine 6, a valid CRC, the motors and the IMU."""
        pub = self.iface._low_state_pub
        self.quiet(self.deliver, lowcmd(q=0.2, kp=30.0, kd=1.0, tau=0.3))
        for _ in range(5):
            self.iface.write_ctrl()
            mujoco.mj_step(self.model, self.data)
            self.iface.after_step()
        self.assertTrue(self.iface._low_state.fast)
        self.assertEqual(len(pub.sent), 5)
        msg = pub.last()
        self.assertEqual(msg.tick, round(self.data.time * 1000))
        self.assertEqual(msg.tick, 10)
        self.assertEqual((msg.mode_machine, msg.mode_pr), (6, 0))
        self.assertEqual(CRC_.Crc(msg), msg.crc)
        self.assertEqual(self.iface.sim_time, self.data.time)
        sd = self.data.sensordata
        np.testing.assert_allclose([m.q for m in msg.motor_state[:N]],
                                   self.data.qpos[self.r.motor_qpos], atol=1e-6)
        np.testing.assert_allclose([m.dq for m in msg.motor_state[:N]],
                                   self.data.qvel[self.r.motor_qvel], atol=1e-6)
        tau_est = [m.tau_est for m in msg.motor_state[:N]]
        np.testing.assert_allclose(tau_est, sd[self.r.motor_tau], atol=1e-4)
        self.assertTrue(np.any(np.abs(tau_est) > 1.0))
        adr = {k: v[0] for k, v in self.r.sensor_adr.items()}
        np.testing.assert_allclose(msg.imu_state.quaternion, sd[adr['imu_quat']:][:4], atol=1e-6)
        np.testing.assert_allclose(msg.imu_state.gyroscope, sd[adr['imu_gyro']:][:3], atol=1e-5)
        np.testing.assert_allclose(msg.imu_state.accelerometer, sd[adr['imu_acc']:][:3], atol=1e-5)
        np.testing.assert_allclose(msg.imu_state.rpy, [ROLL, PITCH, YAW], atol=1e-5)

    def test_tick_wraps_at_32_bits(self):
        """The millisecond tick wraps at 2**32."""
        self.data.time = (2 ** 32 + 7) / 1000.0
        self.iface.after_step()
        self.assertEqual(self.iface._low_state_pub.last().tick, 7)

    def test_shutdown_closes_channels(self):
        """Shutdown closes both DDS channels."""
        self.iface.shutdown()
        self.assertTrue(self.iface._low_state_pub.closed)
        self.assertTrue(FakeSubscriber.by_topic['rt/lowcmd'].closed)

    def test_no_background_threads(self):
        """rt/lowstate and the command timeout run in the physics loop, not in threads."""
        self.assertFalse(hasattr(ui, 'RecurrentThread'))
        self.assertFalse([t for t in vars(self.iface).values() if hasattr(t, 'Start')])


if __name__ == '__main__':
    unittest.main()
