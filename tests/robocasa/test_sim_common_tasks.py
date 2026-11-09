"""Check the task listing and resolution, and the viewer framing, that both sim runners share.

These need RoboCasa and MuJoCo, so they run in the robocasa container:
ROS_DOMAIN_ID=187 tests/robocasa/run_in_container.sh
"""
import argparse
import contextlib
import io
import types
import unittest

import numpy as np

import battery_env
import blank_env
import sim_common

# The --task hooks passed to resolve_task, one per runner, each claiming the Battery and Blank tasks.
OWN_TASKS = battery_env.TASKS + blank_env.TASKS
H12_TASKS = {'tag': 'h12_mujoco', 'robot_task': lambda name: sim_common.canonical_task(name, OWN_TASKS)}
G1_TASKS = {'tag': 'g1_mujoco', 'robot_task': lambda name: sim_common.canonical_task(name, OWN_TASKS)}


def _resolve(*args, **kwargs):
    """Return (resolved task, what resolve_task printed)."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        task = sim_common.resolve_task(*args, **kwargs)
    return task, out.getvalue()


class ResolveTaskTests(unittest.TestCase):
    """Check resolve_task on concrete, abstract, unknown and robot-specific tasks.

    The expected names are the seeded picks from the image's 317 runnable tasks: fixed values,
    not derived from the code under test.
    """

    def test_abstract_base_picks_the_same_subclass_as_before(self):
        """A seeded abstract task base resolves to a fixed concrete subclass and says so."""
        task, out = _resolve('OpenDoor', seed=1, tag='h12_mujoco')
        self.assertEqual(task, 'OpenCabinet')
        self.assertEqual(
            out, "[h12_mujoco] 'OpenDoor' is an abstract task base; randomly selected "
                 "concrete subclass 'OpenCabinet' (from 3 options)\n")
        task, _ = _resolve('ManipulateDoor', seed=1, tag='h12_mujoco')
        self.assertEqual(task, 'CloseFridge')

    def test_log_prefix_is_the_robots_tag(self):
        """resolve_task's message starts with the runner's tag."""
        _, out = _resolve('OpenDoor', seed=1, tag='g1_mujoco')
        self.assertTrue(out.startswith("[g1_mujoco] 'OpenDoor' is an abstract task base"), out)

    def test_concrete_task_is_returned_unchanged(self):
        """A concrete task resolves to itself, silently."""
        task, out = _resolve('OpenFridge', tag='g1_mujoco')
        self.assertEqual((task, out), ('OpenFridge', ''))

    def test_unknown_task_exits_with_the_tag(self):
        """An unknown task exits with a message under the runner's tag."""
        with self.assertRaises(SystemExit) as caught:
            sim_common.resolve_task('NoSuchTask', tag='g1_mujoco')
        self.assertEqual(caught.exception.code, "[g1_mujoco] unknown RoboCasa task 'NoSuchTask'")

    def test_robot_hook_claims_its_task_first(self):
        """A runner's task hook resolves its own task before RoboCasa's."""
        task, out = _resolve('battery', tag='h12_mujoco',
                             robot_task=lambda name: sim_common.canonical_task(name, battery_env.TASKS))
        self.assertEqual((task, out), ('Battery', ''))

    def test_blank_resolves_in_any_case_for_both_robots(self):
        """Blank resolves in any letter case for both runners."""
        for hooks in (H12_TASKS, G1_TASKS):
            for name in ('blank', 'Blank', 'BLANK'):
                self.assertEqual(_resolve(name, **hooks), ('Blank', ''), (hooks['tag'], name))

    def test_battery_resolves_in_any_case_for_both_robots(self):
        """Battery resolves in any letter case for both runners."""
        for hooks in (H12_TASKS, G1_TASKS):
            for name in ('battery', 'Battery', 'BATTERY'):
                self.assertEqual(_resolve(name, **hooks), ('Battery', ''), (hooks['tag'], name))


class RefuseKitchenArgsTests(unittest.TestCase):
    """Check that --layout and --style are refused for a task with no kitchen."""

    def _parser(self):
        """Return a parser with the common options."""
        parser = argparse.ArgumentParser(prog='h12_mujoco.py')
        sim_common.add_common_args(parser)
        return parser

    def _refuse(self, argv, task):
        """Assert that `argv` is refused for `task` with exit code 2 and return the message."""
        parser = self._parser()
        args = parser.parse_args(argv)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
            sim_common.refuse_kitchen_args(parser, args, task, blank_env.TASKS)
        self.assertEqual(caught.exception.code, 2)
        return err.getvalue()

    def test_layout_and_style_are_refused_for_blank(self):
        """--layout, --style or both are refused for Blank, naming the option."""
        self.assertIn("--layout chooses a RoboCasa kitchen; task 'Blank' has none",
                      self._refuse(['--layout', '9'], 'Blank'))
        self.assertIn("--style chooses", self._refuse(['--style', '0'], 'Blank'))
        self.assertIn("--layout and --style choose",
                      self._refuse(['--layout', '1', '--style', '2', '--g1'], 'Blank'))

    def test_blank_without_them_and_kitchens_with_them_pass(self):
        """Blank without them and a kitchen task with them pass."""
        parser = self._parser()
        sim_common.refuse_kitchen_args(parser, parser.parse_args(['--seed', '3']), 'Blank',
                                       blank_env.TASKS)
        sim_common.refuse_kitchen_args(parser, parser.parse_args(['--layout', '9', '--style', '9']),
                                       'OpenFridge', blank_env.TASKS)


class RandomTaskTests(unittest.TestCase):
    """Check random_task's seeded picks."""

    def test_seeded_pick_is_the_one_h12_mujoco_made(self):
        """A seeded random pick is a fixed task."""
        self.assertEqual(sim_common.random_task(seed=42), 'CloseStandMixerHead')
        self.assertEqual(sim_common.random_task(seed=1), 'CupcakeCleanup')

    def test_pick_is_runnable(self):
        """A random pick is a runnable task."""
        _, runnable = sim_common.runnable_tasks()
        self.assertIn(sim_common.random_task(seed=7), runnable)


class RunnableTasksTests(unittest.TestCase):
    """Check runnable_tasks' split of registered and runnable tasks."""

    def test_abstract_bases_are_registered_but_not_runnable(self):
        """Abstract task bases are registered but not runnable."""
        env_classes, runnable = sim_common.runnable_tasks()
        self.assertIn('OpenDoor', env_classes)
        self.assertNotIn('OpenDoor', runnable)
        self.assertIn('OpenFridge', runnable)
        self.assertTrue(runnable <= set(env_classes))


class FrameViewerTests(unittest.TestCase):
    """Check where frame_viewer_on_robot points the viewer's camera."""

    @staticmethod
    def _frame(pos, quat):
        """Frame a stand-in viewer on a body at `pos` (m) with WXYZ `quat` and return its camera."""
        handle = types.SimpleNamespace(cam=types.SimpleNamespace(lookat=np.zeros(3)))
        data = types.SimpleNamespace(xpos=np.array([pos], dtype=float),
                                     xquat=np.array([quat], dtype=float))
        sim_common.frame_viewer_on_robot(handle, data, 0)
        return handle.cam

    def test_looks_straight_down_at_the_body(self):
        """The viewer's free camera looks straight down at the body from VIEW_CAM_DISTANCE."""
        import mujoco
        cam = self._frame((1.0, 2.0, 0.5), (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(cam.type, mujoco.mjtCamera.mjCAMERA_FREE)
        np.testing.assert_allclose(cam.lookat, (1.0, 2.0, 0.5 + sim_common.VIEW_CAM_LOOKAT_DZ))
        self.assertEqual(cam.distance, sim_common.VIEW_CAM_DISTANCE)
        self.assertEqual(cam.elevation, sim_common.VIEW_CAM_ELEVATION)
        self.assertAlmostEqual(cam.azimuth, sim_common.VIEW_CAM_AZIMUTH)

    def test_azimuth_follows_the_body_yaw(self):
        """The camera's azimuth turns with the body's yaw."""
        half = np.sqrt(0.5)  # +90 deg about z
        cam = self._frame((0.0, 0.0, 0.0), (half, 0.0, 0.0, half))
        self.assertAlmostEqual(cam.azimuth, 90.0 + sim_common.VIEW_CAM_AZIMUTH)


if __name__ == '__main__':
    unittest.main()
