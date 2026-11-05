"""Check that the Isaac task's reset restores the physics before it clears the IMU history."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'CL_isaaclab_sim/tasks/h1-2_tasks/stack_rgyblock_h12_27dof_inspire/stack_rgyblock_h12_27dof_inspire_joint_env_cfg.py'


class PhysicalResetTests(unittest.TestCase):
    """Run the task's reset events with stand-in functions."""

    def test_reset_restores_physics_before_clearing_imu_history(self):
        """The reset events restore the scene's default state, then clear the robot's IMU cache."""
        # The task's own EventCfg, compiled alone so that neither Kit nor a GPU is needed.
        tree = ast.parse(CONFIG.read_text())
        event_class = next(node for node in tree.body
                           if isinstance(node, ast.ClassDef) and node.name == 'EventCfg')
        calls = []
        restore = lambda env, env_ids: calls.append(('physics', env, env_ids))
        reset_imu = lambda env, env_ids: calls.append(('imu', env, env_ids))
        namespace = dict(configclass=lambda cls: cls, EventTermCfg=SimpleNamespace,
                         base_mdp=SimpleNamespace(reset_scene_to_default=restore),
                         reset_robot_imu_cache=reset_imu)
        exec(compile(ast.Module(body=[event_class], type_ignores=[]), str(CONFIG), 'exec'), namespace)
        events = [value for value in vars(namespace['EventCfg']).values()
                  if isinstance(value, SimpleNamespace) and value.mode == 'reset']
        env, ids = object(), object()
        for event in events:
            event.func(env, ids, **getattr(event, 'params', {}))
        self.assertEqual(calls, [('physics', env, ids), ('imu', env, ids)])


if __name__ == '__main__':
    unittest.main()
