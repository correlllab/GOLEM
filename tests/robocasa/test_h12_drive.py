"""Check the H1-2 sim's opt-in drive model (golem_robocasa/h12_drive.py, GOLEM_H12_DRIVE_MODEL=1).

The series-elastic hip roll's response against its closed form, the command delay, the reported rotor state, the
drivers' velocity low-pass and tau_est, on test_h12_unitree_interface's stand-in model with the DDS faked. Runs in
the standard runner (tests/robocasa/run_in_container.sh).
"""
import contextlib
import io
import os
import unittest
from unittest import mock

import mujoco
import numpy as np

import sys
import types

import h12_drive
import unitree_interface as ui
from dds_fakes import FakeSubscriber, fake_dds
from test_h12_unitree_interface import N, lowcmd, model_and_resolver


class SeriesDriveTests(unittest.TestCase):
    """SeriesDrive alone, its links held still."""

    def test_rotor_follows_the_closed_form_at_20_hz(self):
        """With the link held, the rotor's response to a 20 Hz target is kp / (kp + k - J w^2 + j w (kd + c))."""
        drive = h12_drive.SeriesDrive(slots=(0,))
        drive.reset(np.zeros(1), np.zeros(1))
        kp, kd, f, dt, amp = 250.0, 6.5, 20.0, 0.002, 0.01
        w = 2 * np.pi * f
        n = int(3.0 / dt)
        th = []
        for i in range(n):
            c = _Cmd(q=amp * np.sin(w * i * dt), kp=kp, kd=kd)
            drive.step(c, np.zeros(1), np.zeros(1), np.array([1e4]), dt)
            th.append(drive.th[0])
        th = np.array(th[n // 2:])
        t = (np.arange(n)[n // 2:] + 1) * dt   # th is recorded at the end of its step
        # least-squares amplitude and phase of the rotor at w
        a, b = np.linalg.lstsq(np.stack([np.sin(w * t), np.cos(w * t)], 1), th, rcond=None)[0]
        got = complex(a, b) / amp
        J, k, c = h12_drive.DRIVE_J, h12_drive.DRIVE_K, h12_drive.DRIVE_C
        # the target held over each step lags it half a step
        want = kp / (kp + k - J * w * w + 1j * w * (kd + c)) * np.exp(-0.5j * w * dt)
        self.assertAlmostEqual(abs(got), abs(want), delta=0.05 * abs(want))
        self.assertAlmostEqual(np.angle(got), np.angle(want), delta=0.05)

    def test_motors_off_means_no_pd_torque(self):
        """Without a command the PD torque is zero and the rotor only follows the spring."""
        drive = h12_drive.SeriesDrive(slots=(0,))
        drive.reset(np.zeros(1), np.zeros(1))
        drive.step(None, np.zeros(1), np.zeros(1), np.array([1e4]), 0.002)
        self.assertEqual(drive.tau_pd[0], 0.0)


class _Cmd:
    """A one-motor MotorCommand stand-in."""

    def __init__(self, q, kp, kd):
        self.on = np.array([True])
        self.q, self.dq, self.tau = np.array([q]), np.zeros(1), np.zeros(1)
        self.kp, self.kd = np.array([kp]), np.array([kd])


class DriveInterfaceTests(unittest.TestCase):
    """The drive's SimInterface on the stand-in model."""

    def setUp(self):
        """Fake the DDS channels and make the drive's interface."""
        fake_dds(self, ui)
        self.model, self.r = model_and_resolver()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        with contextlib.redirect_stdout(io.StringIO()):
            self.iface = h12_drive.make_interface(self.model, self.data, self.r)
        self.deliver = FakeSubscriber.by_topic['rt/lowcmd'].handler

    def step(self, n=1):
        """Run n physics steps through the interface."""
        for _ in range(n):
            self.iface.write_ctrl()
            mujoco.mj_step(self.model, self.data)
            self.iface.after_step()

    def test_command_reaches_the_motors_delay_steps_late(self):
        """A first command drives the motors only after DELAY_STEPS physics steps."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.deliver(lowcmd(q=0.3, dq=0.0, kp=40.0, kd=2.0, tau=0.0))
        ctrl = []
        for _ in range(h12_drive.DELAY_STEPS + 1):
            self.iface.write_ctrl()
            ctrl.append(self.data.ctrl[self.r.motor_ctrl].copy())
        others = [s for s in range(N) if s not in h12_drive.DRIVE_SLOTS]
        for c in ctrl[:-1]:
            self.assertTrue(np.all(c[others] == 0.0))
        self.assertTrue(np.all(ctrl[-1][others] != 0.0))

    def test_lowstate_reports_rotors_lagged_dq_and_pd_torque(self):
        """rt/lowstate carries the hip rolls' rotor angles and PD torque, and LAG_SLOTS' low-passed velocity."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.deliver(lowcmd(q=0.3, dq=0.0, kp=40.0, kd=2.0, tau=0.0))
        self.step(60)
        msg = self.iface._low_state_pub.last()
        q = np.array([m.q for m in msg.motor_state[:N]])
        dq = np.array([m.dq for m in msg.motor_state[:N]])
        tau = np.array([m.tau_est for m in msg.motor_state[:N]])
        slots = list(h12_drive.DRIVE_SLOTS)
        np.testing.assert_allclose(q[slots], self.iface.drive.th, atol=1e-6)
        np.testing.assert_allclose(tau[slots], self.iface.drive.tau_pd, atol=1e-3)
        link = self.data.qpos[self.r.motor_qpos][slots]
        self.assertTrue(np.all(np.abs(q[slots] - link) > 0.0))   # the rotor is not the link
        lagged = [s for s in h12_drive.LAG_SLOTS if s not in slots]
        unlagged = [s for s in range(N) if s not in h12_drive.LAG_SLOTS]
        true = self.data.qvel[self.r.motor_qvel]
        np.testing.assert_allclose(dq[unlagged], true[unlagged], atol=1e-5)
        self.assertGreater(np.max(np.abs(dq[lagged] - true[lagged])), 1e-3)   # still accelerating: lagged

    def test_sim_tests_recorder_sees_every_step(self):
        """The drive publishes through SimInterface.after_step, which the sim tests' recorder wraps."""
        calls, after_step = [], ui.SimInterface.after_step
        with mock.patch.object(ui.SimInterface, 'after_step', lambda s: (after_step(s), calls.append(1))):
            self.step(3)
        self.assertEqual(len(calls), 3)

    def test_physics_reset_restarts_the_drive(self):
        """After a physics reset the rotors sit on the links again and the delay line is empty."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.deliver(lowcmd(q=0.3))
        self.step(20)
        mujoco.mj_resetData(self.model, self.data)
        self.iface.physics_reset()
        self.assertTrue(all(c is None for c in self.iface.cmds))   # the delay line holds no command
        self.iface.write_ctrl()
        np.testing.assert_allclose(self.iface.drive.th, self.data.qpos[self.r.motor_qpos][list(h12_drive.DRIVE_SLOTS)])


class KnobTests(unittest.TestCase):
    """GOLEM_H12_DRIVE_MODEL picks the interface; off by default."""

    def test_off_by_default_on_with_the_knob(self):
        # h12_robot imports the gripper bridge, which needs the ROS workspace's magpie_msgs: a stand-in, as
        # test_sim_rate's gripper tests use one
        stand_in = types.ModuleType('magpie_hand_bridge')
        stand_in.MagpieHandBridge = type('MagpieHandBridge', (), {})
        with mock.patch.dict(sys.modules, {'magpie_hand_bridge': sys.modules.get('magpie_hand_bridge', stand_in)}):
            import h12_robot
        robot = h12_robot.H12Robot()
        with mock.patch.object(h12_robot, 'SimInterface') as plain, \
                mock.patch.object(h12_drive, 'make_interface') as drive:
            with mock.patch.dict(os.environ, {'GOLEM_H12_DRIVE_MODEL': ''}):
                robot.make_interface('m', 'd', 'r')
            self.assertEqual((plain.call_count, drive.call_count), (1, 0))
            with mock.patch.dict(os.environ, {'GOLEM_H12_DRIVE_MODEL': '1'}):
                robot.make_interface('m', 'd', 'r')
            self.assertEqual((plain.call_count, drive.call_count), (1, 1))


if __name__ == '__main__':
    unittest.main()
