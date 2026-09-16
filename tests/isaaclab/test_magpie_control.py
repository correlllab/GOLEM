"""Magpie aperture/force commands reach the physical joint drives."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'))
from src.python.control.magpie_control import MagpieController
from src.python.control.robot_variant import robot_variant


class MagpieControlTests(unittest.TestCase):
    def test_linkage_speed_force_and_measured_state(self):
        names=list(robot_variant().hand_joints)
        data=SimpleNamespace(joint_names=names,joint_pos=torch.zeros(1,12),joint_vel=torch.zeros(1,12),applied_torque=torch.zeros(1,12),joint_effort_limits=torch.zeros(1,12))
        actuator=SimpleNamespace(effort_limit=torch.zeros(1,12),joint_indices=list(range(12)))
        robot=Mock(data=data,device='cpu',actuators={'hands':actuator})
        robot.write_joint_effort_limit_to_sim.side_effect=lambda v,joint_ids: data.joint_effort_limits.copy_(v)
        commands={side:{'aperture':55.,'force':4.,'speed':.5,'sequence':1} for side in ('left','right')}
        bridge=Mock();bridge.get_commands.return_value=commands
        control=MagpieController(robot,bridge)
        q=control.get_joint_targets(.1)
        torch.testing.assert_close(q[:6],torch.tensor([1.,-1.,1.,-1.,1.,-1.])* .157)
        torch.testing.assert_close(robot.write_joint_effort_limit_to_sim.call_args.args[0],torch.tensor([[.2,0.,0.]*4]))
        torch.testing.assert_close(actuator.effort_limit,torch.tensor([[.2,0.,0.]*4]))
        data.joint_pos[0]=torch.tensor([1.,-1.,1.,-1.,1.,-1.]*2)*1.025
        control.publish(1.)
        states=bridge.publish.call_args.args[1]
        self.assertAlmostEqual(states['left']['position'],55.,places=4)
        self.assertEqual(states['left']['finger_positions'],states['right']['finger_positions'])
        bridge.acknowledge.assert_called_once_with(commands)
