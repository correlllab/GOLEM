"""DDS-to-action tests on CPU, with no DDS participant or simulator."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
ROOT = Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'
sys.path.insert(0, str(ROOT))
from src.python.control.motor_contract import BODY_ACTION_DIM, MOTOR_FIELDS

stub = ModuleType('src.python.dds.common.dds_master')
stub.dds_manager = Mock()
spec = importlib.util.spec_from_file_location('src.python.control._tested_action_provider_dds', ROOT / 'src/python/control/action_provider_dds.py')
provider_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {stub.__name__: stub}):
    spec.loader.exec_module(provider_module)


def command(sequence=1):
    return {'sequence': sequence, 'motor_cmd': {field: [float(i + 1)] * 27 for i, field in enumerate(MOTOR_FIELDS)}}


class ActionProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleNamespace(device='cpu', physics_dt=.01, _sim_step_counter=0)
        self.robot, self.hand = Mock(), Mock()
        self.robot.get_robot_command.return_value = command()
        self.hand.get_inspire_hand_command.return_value = None
        stub.dds_manager.get_object.side_effect = lambda name: self.robot if name == 'h1_2' else self.hand
        with patch.dict(os.environ, {'GOLEM_CMD_TIMEOUT': '.5'}):
            self.provider = provider_module.DDSActionProvider(self.env, SimpleNamespace(robot_type='h1_2', enable_inspire_dds=True))

    def test_preserves_every_field_and_reuses_output(self):
        action = self.provider.get_action(self.env)
        expected = torch.tensor([[float(i + 1)] * 27 for i in range(6)])
        torch.testing.assert_close(action[0, :BODY_ACTION_DIM].reshape(6, 27), expected)
        self.assertEqual(action.data_ptr(), self.provider.get_action(self.env).data_ptr())

    def test_stale_sequence_expires_in_simulation_time(self):
        self.provider.get_action(self.env)
        self.env._sim_step_counter = 50
        self.assertTrue(self.provider.get_action(self.env)[0, :BODY_ACTION_DIM].any())
        self.env._sim_step_counter = 51
        self.assertFalse(self.provider.get_action(self.env)[0, :BODY_ACTION_DIM].any())
        self.robot.get_robot_command.return_value = command(2)
        self.assertTrue(self.provider.get_action(self.env)[0, :BODY_ACTION_DIM].any())

    def test_time_reversal_releases_old_command(self):
        self.env._sim_step_counter = 20
        self.provider.get_action(self.env)
        self.env._sim_step_counter = 0
        self.assertFalse(self.provider.get_action(self.env)[0, :BODY_ACTION_DIM].any())

    def test_mode_zero_reaches_the_drive(self):
        self.robot.get_robot_command.return_value['motor_cmd']['mode'] = [0] * 27
        body = self.provider.get_action(self.env)[0, :BODY_ACTION_DIM].reshape(6, 27)
        self.assertFalse(body[5].any())

    def test_malformed_command_releases_previous_command(self):
        for bad in ({}, {'positions': [1]}, {'positions': [float('nan')] * 27, **{k: [1.] * 27 for k in MOTOR_FIELDS[1:]}}):
            self.robot.get_robot_command.return_value = command(10)
            self.provider._sequence = None
            self.provider.get_action(self.env)
            self.robot.get_robot_command.return_value = {'sequence': 11, 'motor_cmd': bad}
            self.assertFalse(self.provider.get_action(self.env)[0, :BODY_ACTION_DIM].any())

    def test_hand_positions_survive_body_timeout(self):
        positions = [i * .1 for i in range(12)]
        self.hand.get_inspire_hand_command.return_value = {'positions': positions}
        self.provider.get_action(self.env)
        self.env._sim_step_counter = 100
        result = self.provider.get_action(self.env)
        torch.testing.assert_close(result[0, BODY_ACTION_DIM:], torch.tensor(positions))
        self.assertFalse(result[0, :BODY_ACTION_DIM].any())
        self.hand.get_inspire_hand_command.return_value = {'positions': [float('nan')] * 12}
        torch.testing.assert_close(self.provider.get_action(self.env)[0, BODY_ACTION_DIM:], torch.tensor(positions))

    def test_start_and_cleanup_have_no_worker_thread(self):
        self.provider.start()
        self.assertTrue(self.provider.is_running)
        self.assertIsNone(self.provider._thread)
        self.provider.cleanup()
        self.assertFalse(self.provider.is_running)
        self.robot.stop_communication.assert_called_once()
        self.hand.stop_communication.assert_called_once()

if __name__ == '__main__':
    unittest.main()
