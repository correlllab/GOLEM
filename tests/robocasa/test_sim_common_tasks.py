"""Task listing and resolution, and the viewer framing, that both sim runners share.

These need RoboCasa and MuJoCo, so they run in the robocasa container:
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
"""
import contextlib
import io
import types
import unittest

import numpy as np

import battery_env
import sim_common


def _resolve(*args, **kwargs):
    """(resolved task, what resolve_task printed)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        task = sim_common.resolve_task(*args, **kwargs)
    return task, out.getvalue()


class ResolveTaskTests(unittest.TestCase):
    # The expected names are the seeded picks from this image's 317 runnable tasks.
    # They are fixed values, not derived from the code under test.
    def test_abstract_base_picks_the_same_subclass_as_before(self):
        task, out = _resolve('OpenDoor', seed=1, tag='h12_mujoco')
        self.assertEqual(task, 'OpenCabinet')
        self.assertEqual(
            out, "[h12_mujoco] 'OpenDoor' is an abstract task base; randomly selected "
                 "concrete subclass 'OpenCabinet' (from 3 options)\n")
        task, _ = _resolve('ManipulateDoor', seed=1, tag='h12_mujoco')
        self.assertEqual(task, 'CloseFridge')

    def test_log_prefix_is_the_robots_tag(self):
        _, out = _resolve('OpenDoor', seed=1, tag='g1_mujoco')
        self.assertTrue(out.startswith("[g1_mujoco] 'OpenDoor' is an abstract task base"), out)

    def test_concrete_task_is_returned_unchanged(self):
        task, out = _resolve('OpenFridge', tag='g1_mujoco')
        self.assertEqual((task, out), ('OpenFridge', ''))

    def test_unknown_task_exits_with_the_tag(self):
        with self.assertRaises(SystemExit) as caught:
            sim_common.resolve_task('NoSuchTask', tag='g1_mujoco')
        self.assertEqual(caught.exception.code, "[g1_mujoco] unknown RoboCasa task 'NoSuchTask'")

    def test_robot_hook_claims_its_task_first(self):
        task, out = _resolve('battery', tag='h12_mujoco', robot_task=battery_env.canonical_task)
        self.assertEqual((task, out), ('Battery', ''))

    def test_battery_without_the_h12_hook_is_h12_only(self):
        for name in ('Battery', 'battery'):
            with self.assertRaises(SystemExit) as caught:
                sim_common.resolve_task(name, tag='g1_mujoco')
            self.assertEqual(caught.exception.code,
                             f"[g1_mujoco] task {name!r} is H1-2 only")

    def test_h12_only_tasks_are_the_battery_envs(self):
        self.assertEqual(sim_common.H12_ONLY_TASKS, battery_env.TASKS)


class RandomTaskTests(unittest.TestCase):
    def test_seeded_pick_is_the_one_h12_mujoco_made(self):
        self.assertEqual(sim_common.random_task(seed=42), 'CloseStandMixerHead')
        self.assertEqual(sim_common.random_task(seed=1), 'CupcakeCleanup')

    def test_pick_is_runnable(self):
        _, runnable = sim_common.runnable_tasks()
        self.assertIn(sim_common.random_task(seed=7), runnable)


class RunnableTasksTests(unittest.TestCase):
    def test_abstract_bases_are_registered_but_not_runnable(self):
        env_classes, runnable = sim_common.runnable_tasks()
        self.assertIn('OpenDoor', env_classes)
        self.assertNotIn('OpenDoor', runnable)
        self.assertIn('OpenFridge', runnable)
        self.assertTrue(runnable <= set(env_classes))


class FrameViewerTests(unittest.TestCase):
    @staticmethod
    def _frame(pos, quat):
        handle = types.SimpleNamespace(cam=types.SimpleNamespace(lookat=np.zeros(3)))
        data = types.SimpleNamespace(xpos=np.array([pos], dtype=float),
                                     xquat=np.array([quat], dtype=float))
        sim_common.frame_viewer_on_robot(handle, data, 0)
        return handle.cam

    def test_looks_straight_down_at_the_body(self):
        import mujoco
        cam = self._frame((1.0, 2.0, 0.5), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(cam.type, mujoco.mjtCamera.mjCAMERA_FREE)
        np.testing.assert_allclose(cam.lookat, (1.0, 2.0, 0.5 + sim_common.VIEW_CAM_LOOKAT_DZ))
        self.assertEqual(cam.distance, sim_common.VIEW_CAM_DISTANCE)
        self.assertEqual(cam.elevation, sim_common.VIEW_CAM_ELEVATION)
        self.assertAlmostEqual(cam.azimuth, sim_common.VIEW_CAM_AZIMUTH)

    def test_azimuth_follows_the_body_yaw(self):
        half = np.sqrt(0.5)   # +90 deg about z
        cam = self._frame((0.0, 0.0, 0.0), (half, 0.0, 0.0, half))
        self.assertAlmostEqual(cam.azimuth, 90.0 + sim_common.VIEW_CAM_AZIMUTH)


if __name__ == '__main__':
    unittest.main()
