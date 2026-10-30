"""Exercise real Unitree IDL/CRC and bridge methods without live DDS participants."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'
sys.path.insert(0, str(ROOT))
from src.python.dds.specialized.h12_robot_dds import H12RobotDDS
from src.python.dds.specialized.inspire_dds import InspireDDS
from src.python.dds.common.sharedmemorymanager import SharedMemoryManager
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_


class DDSMotorTests(unittest.TestCase):
    def setUp(self):
        self.allocations = []
        def memory(bridge, **kwargs):
            for side in ('input', 'output'):
                shm = SharedMemoryManager(size=kwargs[side + '_size'])
                setattr(bridge, side + '_shm', shm)
                self.allocations.append(shm)
        with patch.object(H12RobotDDS, 'setup_shared_memory', memory), patch.object(InspireDDS, 'setup_shared_memory', memory):
            self.bridge = H12RobotDDS()
            self.hand = InspireDDS()
        self.published = []
        self.bridge.publisher = type('Capture', (), {'Write': lambda _, msg: self.published.append(copy.deepcopy(msg))})()

    def tearDown(self):
        for shm in self.allocations:
            shm.cleanup()

    def lowcmd(self):
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr, msg.mode_machine = 0, 1
        for i, motor in enumerate(msg.motor_cmd):
            motor.mode = i % 2
            motor.q, motor.dq, motor.tau, motor.kp, motor.kd = i * .0123456789, -i * .023456789, i * .03456789, 21.23456789 + i, .123456789 + i
        msg.crc = self.bridge.crc.Crc(msg)
        return msg

    def test_lowstate_preserves_wxyz_and_motor_order(self):
        q = np.arange(27, dtype=np.float32) / 10
        imu = np.array([1, 2, 3, .5, -.5, .5, -.5, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        self.bridge.write_robot_state(q, q + 1, q + 2, imu)
        self.bridge.dds_publisher()
        self.assertEqual(len(self.published), 1)
        msg = self.published[0]
        np.testing.assert_allclose(msg.imu_state.quaternion, imu[3:7])
        np.testing.assert_allclose(msg.imu_state.accelerometer, imu[7:10])
        np.testing.assert_allclose(msg.imu_state.gyroscope, imu[10:13])
        np.testing.assert_allclose([m.q for m in msg.motor_state[:27]], q)
        np.testing.assert_allclose([m.dq for m in msg.motor_state[:27]], q + 1)
        np.testing.assert_allclose([m.tau_est for m in msg.motor_state[:27]], q + 2)
        self.assertEqual(msg.crc, self.bridge.crc.Crc(msg))
        self.assertEqual(msg.tick, 1)

    def test_command_all_fields_modes_sequence_and_large_shared_buffer(self):
        msg = self.lowcmd()
        self.bridge.dds_subscriber(msg)
        first = self.bridge.get_robot_command()
        self.assertIsNotNone(first)
        self.assertEqual(first['sequence'], 1)
        self.assertEqual(first['mode_machine'], 1)
        for field, attribute in [('mode', 'mode'), ('positions', 'q'), ('velocities', 'dq'), ('torques', 'tau'), ('kp', 'kp'), ('kd', 'kd')]:
            np.testing.assert_allclose(first['motor_cmd'][field], [getattr(m, attribute) for m in msg.motor_cmd])
        self.bridge.dds_subscriber(msg)
        self.assertEqual(self.bridge.get_robot_command()['sequence'], 2)

    def test_bad_crc_cannot_replace_latest_valid_command(self):
        msg = self.lowcmd()
        self.bridge.dds_subscriber(msg)
        msg.motor_cmd[0].q += 1
        with patch('builtins.print'):
            self.bridge.dds_subscriber(msg)
        saved = self.bridge.get_robot_command()
        self.assertEqual(saved['sequence'], 1)
        self.assertEqual(saved['motor_cmd']['positions'][0], 0.)

    def test_full_precision_hand_state_fits_shared_memory(self):
        values = np.linspace(.123456789123456, 1.65432198765432, 12)
        self.hand.write_inspire_state(values, values + 1, values + 2)
        result = self.hand.input_shm.read_data()
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['positions'], values)
        np.testing.assert_allclose(result['velocities'], values + 1)
        np.testing.assert_allclose(result['torques'], values + 2)

if __name__ == '__main__':
    unittest.main()
