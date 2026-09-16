"""Drive regression without Kit; actual physics response is checked by run_robot_interop.sh."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch

ROOT = Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'
sys.path.insert(0, str(ROOT))
from src.python.control.motor_contract import BODY_ACTION_DIM, ACTION_DIM

managers = ModuleType('isaaclab.managers')
managers.ActionTerm = object
managers.ActionTermCfg = object
utils = ModuleType('isaaclab.utils')
utils.configclass = lambda cls: cls
with patch.dict(sys.modules, {'isaaclab.managers': managers, 'isaaclab.utils': utils}):
    spec = importlib.util.spec_from_file_location('src.python.control.test_unitree_action', ROOT/'src/python/control/unitree_action.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class ImplicitActionTests(unittest.TestCase):
    def test_targets_and_gains_reach_solver_without_explicit_pd_torque(self):
        action = object.__new__(module.UnitreeMotorAction)
        action.num_envs = 1
        action._body_ids = list(range(27))
        action._hand_ids = list(range(27,39))
        action._coupled_ids = list(range(39,51))
        action._hand_sources = torch.arange(12)
        action._hand_scales = torch.ones(12)
        action._raw_actions = torch.zeros(1,ACTION_DIM)
        action._kp_cache = torch.full((1,27), float('nan'))
        action._kd_cache = torch.full((1,27), float('nan'))
        action._efforts = torch.zeros(1,27)
        model = SimpleNamespace(joint_indices=list(range(27)),stiffness=torch.zeros(1,27),damping=torch.zeros(1,27))
        data = SimpleNamespace(joint_stiffness=torch.zeros(1,51),joint_damping=torch.zeros(1,51),joint_pos=torch.zeros(1,51),joint_vel=torch.zeros(1,51))
        asset = Mock(data=data,actuators={'body':model})
        asset.write_joint_stiffness_to_sim.side_effect = lambda value,joint_ids: data.joint_stiffness.__setitem__((slice(None),joint_ids),value)
        asset.write_joint_damping_to_sim.side_effect = lambda value,joint_ids: data.joint_damping.__setitem__((slice(None),joint_ids),value)
        action._asset = asset
        # Targets q=1,dq=2, feedforward3,kp4,kd5, enabledmode1.
        command=torch.zeros(1,ACTION_DIM)
        for field,value in enumerate((1,2,3,4,5,1)): command[:,field*27:(field+1)*27]=value
        action.process_actions(command)
        action.apply_actions()
        torch.testing.assert_close(asset.set_joint_effort_target.call_args.args[0],torch.full((1,27),3.))
        torch.testing.assert_close(model.stiffness,torch.full((1,27),4.))
        torch.testing.assert_close(model.damping,torch.full((1,27),5.))
        torch.testing.assert_close(asset.set_joint_velocity_target.call_args.args[0],torch.full((1,27),2.))
        action.process_actions(command)
        self.assertEqual(asset.write_joint_stiffness_to_sim.call_count,1)
        command[:,5*27:BODY_ACTION_DIM]=0
        action.process_actions(command);action.apply_actions()
        self.assertFalse(model.stiffness.any())
        self.assertFalse(model.damping.any())
        self.assertFalse(asset.set_joint_effort_target.call_args.args[0].any())
        action.reset()
        self.assertTrue(torch.isnan(action._kp_cache).all())
