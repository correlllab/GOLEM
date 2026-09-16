"""CPU regressions for wire joint ordering, cache lifetime and IMU timing."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'
sys.path.insert(0, str(ROOT))
dds = types.ModuleType('src.python.dds.common.dds_master')
dds.dds_manager = Mock()

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tasks/common_observations' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

body = load('h12_27dof_state')
hand = load('inspire_state')

def env_for(names):
    n = len(names)
    data = types.SimpleNamespace(joint_names=names, joint_pos=torch.arange(n).float()[None],
        joint_vel=torch.arange(n).float()[None] + 100, applied_torque=torch.arange(n).float()[None] + 200,
        body_names=[], root_state_w=torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
    return types.SimpleNamespace(scene={'robot': types.SimpleNamespace(data=data)},
        sim=types.SimpleNamespace(current_time=0.), physics_dt=.002, episode_length_buf=torch.tensor([0]))

class StateTests(unittest.TestCase):
    def test_body_wire_order(self):
        names = body.get_robot_boy_joint_names()
        self.assertEqual(names[:3], ['left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint'])
        self.assertEqual(names[6:9], ['right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint'])

    def test_body_reuses_buffers_without_printing(self):
        env = env_for(body.get_robot_boy_joint_names()[::-1])
        with patch('builtins.print') as printing:
            first = body.get_robot_boy_joint_states(env, False)
            second = body.get_robot_boy_joint_states(env, False)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first[0, :27], torch.arange(26, -1, -1).float())
        printing.assert_not_called()

    def test_hands_resolve_names_and_rebuild_for_other_scene(self):
        names = hand.get_robot_girl_joint_names()
        for order in [names[::-1], names]:
            env = env_for(order)
            result = hand.get_robot_inspire_joint_states(env, False)
            expected = torch.tensor([[order.index(n) for n in names]], dtype=torch.float32)
            torch.testing.assert_close(result, expected)

    def test_imu_elapsed_sim_time_and_env_isolation(self):
        env = env_for([])
        torch.testing.assert_close(body.get_robot_imu_data(env)[0, 7:10], torch.tensor([0., 0., 9.81]))
        env.sim.current_time = .02
        env.episode_length_buf += 1
        env.scene['robot'].data.root_state_w[0, 7] = .04
        torch.testing.assert_close(body.get_robot_imu_data(env)[0, 7:10], torch.tensor([2., 0., 9.81]))
        other = env_for([])
        other.scene['robot'].data.root_state_w[0, 7] = 30
        torch.testing.assert_close(body.get_robot_imu_data(other)[0, 7:10], torch.tensor([0., 0., 9.81]))
        env.sim.current_time = 0
        torch.testing.assert_close(body.get_robot_imu_data(env)[0, 7:10], torch.tensor([0., 0., 9.81]))

    def test_imu_reset_at_monotonic_time(self):
        env = env_for([])
        env.episode_length_buf[:] = 10
        body.get_robot_imu_data(env)
        env.sim.current_time = .1
        env.episode_length_buf[:] = 0
        env.scene['robot'].data.root_state_w[0, 7] = 50
        torch.testing.assert_close(body.get_robot_imu_data(env)[0, 7:10], torch.tensor([0., 0., 9.81]))

    def test_explicit_reset_invalidates_even_when_episode_counter_advanced(self):
        env = env_for([])
        body.get_robot_imu_data(env)
        body.reset_robot_imu_cache(env)
        env.sim.current_time = .02
        env.episode_length_buf[:] = 20
        env.scene['robot'].data.root_state_w[0, 7] = 10
        torch.testing.assert_close(body.get_robot_imu_data(env)[0, 7:10], torch.tensor([0., 0., 9.81]))

    def test_actual_imu_link_frame_is_used_instead_of_root(self):
        env = env_for([])
        data = env.scene['robot'].data
        data.body_names = ['pelvis', 'imu_link']
        data.body_link_pose_w = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.], [1., 2., 3., 0., 1., 0., 0.]]])
        data.body_link_vel_w = torch.tensor([[[0., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 2., 3.]]])
        result = body.get_robot_imu_data(env)
        torch.testing.assert_close(result[0, :7], data.body_link_pose_w[0, 1])
        torch.testing.assert_close(result[0, 7:10], torch.tensor([0., 0., -9.81]))
        torch.testing.assert_close(result[0, 10:], torch.tensor([1., -2., -3.]))

    def test_partial_reset_only_resets_requested_environment(self):
        env = env_for([])
        env.scene['robot'].data.root_state_w = env.scene['robot'].data.root_state_w.repeat(2, 1)
        env.episode_length_buf = torch.ones(2, dtype=torch.long)
        body.get_robot_imu_data(env)
        body.reset_robot_imu_cache(env, torch.tensor([1]))
        env.sim.current_time = .02
        env.episode_length_buf += 10
        env.scene['robot'].data.root_state_w[:, 7] = .04
        torch.testing.assert_close(body.get_robot_imu_data(env)[:, 7:10], torch.tensor([[2., 0., 9.81], [0., 0., 9.81]]))

    def test_cache_rebuilds_on_scene_and_dtype_changes(self):
        names = body.get_robot_boy_joint_names()
        env = env_for(names)
        body.get_robot_boy_joint_states(env, False)
        env.scene = env_for(names[::-1]).scene
        result = body.get_robot_boy_joint_states(env, False)
        torch.testing.assert_close(result[0, :27], torch.arange(26, -1, -1).float())
        data = env.scene['robot'].data
        data.joint_pos = data.joint_pos.double()
        data.joint_vel = data.joint_vel.double()
        data.applied_torque = data.applied_torque.double()
        self.assertEqual(body.get_robot_boy_joint_states(env, False).dtype, torch.float64)

    def test_imu_step_counter_fallback_and_duplicate_sample(self):
        env = env_for([])
        del env.sim
        env._sim_step_counter = 0
        body.get_robot_imu_data(env)
        env._sim_step_counter = 10
        env.episode_length_buf += 1
        env.scene['robot'].data.root_state_w[0, 7] = .04
        first = body.get_robot_imu_data(env)
        torch.testing.assert_close(first[0, 7:10], torch.tensor([2., 0., 9.81]))
        torch.testing.assert_close(body.get_robot_imu_data(env), first)

    def test_imu_wxyz_rotation_and_scene_reset(self):
        env = env_for([])
        # A half-turn about X maps world gravity into negative body Z.
        env.scene['robot'].data.root_state_w[0, 3:7] = torch.tensor([0., 1., 0., 0.])
        result = body.get_robot_imu_data(env)
        torch.testing.assert_close(result[0, 3:7], torch.tensor([0., 1., 0., 0.]))
        torch.testing.assert_close(result[0, 7:10], torch.tensor([0., 0., -9.81]))
        env.scene = env_for([]).scene
        env.sim.current_time = .1
        env.episode_length_buf += 1
        env.scene['robot'].data.root_state_w[0, 7] = 99.
        torch.testing.assert_close(body.get_robot_imu_data(env)[0, 7:10], torch.tensor([0., 0., 9.81]))

    def test_publishing_wire_arrays_and_twenty_ms_throttle(self):
        env = env_for(body.get_robot_boy_joint_names()[::-1])
        published = []
        interface = types.SimpleNamespace(write_robot_state=lambda *args: published.append([x.copy() for x in args]))
        with patch.object(body, '_h12_robot_dds', interface), patch.object(body.time, 'monotonic', side_effect=[0., .01, .03]):
            body.get_robot_boy_joint_states(env)
            env.sim.current_time = .01
            body.get_robot_boy_joint_states(env)
            env.sim.current_time = .04
            env.episode_length_buf += 1
            env.scene['robot'].data.root_state_w[0, 7] = .08
            body.get_robot_boy_joint_states(env)
        self.assertEqual(len(published), 2)
        torch.testing.assert_close(torch.from_numpy(published[0][0]), torch.arange(26, -1, -1).float())
        torch.testing.assert_close(torch.from_numpy(published[1][3][7:10]), torch.tensor([2., 0., 9.81]))

    def test_dds_late_init_does_not_register_cleanup_repeatedly(self):
        hand._inspire_dds = None
        dds.dds_manager.get_object.return_value = None
        with patch.dict(sys.modules, {dds.__name__: dds}), patch('atexit.register') as register:
            hand._get_inspire_dds_instance()
            hand._get_inspire_dds_instance()
        self.assertLessEqual(register.call_count, 1)
        expected = object()
        dds.dds_manager.get_object.return_value = expected
        with patch.dict(sys.modules, {dds.__name__: dds}):
            self.assertIs(hand._get_inspire_dds_instance(), expected)
        hand._inspire_dds = None

if __name__ == '__main__':
    unittest.main()
