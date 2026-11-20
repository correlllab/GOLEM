"""Check the Isaac sim's elastic band: its force, its toggle and what it stores on the robot, without Kit."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'CL_isaaclab_sim'))
from src.python.control.elastic_band import (
    BAND_DAMPING, BAND_DROP, BAND_STIFFNESS, ElasticBand, band_force, world_to_body)


def stand_in_robot(position=(0., 0., 1.), velocity=(0., 0., 0.)):
    """Return a stand-in articulation with a pelvis and a torso_link at position, moving at velocity."""
    data = SimpleNamespace(body_names=['pelvis', 'torso_link'],
                           body_pos_w=torch.tensor([[[0., 0., 0.], list(position)]]),
                           body_quat_w=torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]),
                           body_lin_vel_w=torch.tensor([[[0., 0., 0.], list(velocity)]]))
    return Mock(data=data, device='cpu')


def stored_force(robot):
    """Return the force of the robot's last set_forces_and_torques call, (num_envs, bodies, 3)."""
    return robot.permanent_wrench_composer.set_forces_and_torques.call_args.kwargs['forces']


class BandForceTests(unittest.TestCase):
    """Run band_force on single bodies."""

    def test_spring_pulls_toward_the_anchor(self):
        """A body 0.1 m below the anchor is pulled up by stiffness times 0.1 m."""
        force = band_force(torch.tensor([[0., 0., .9]]), torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]), 1e4, 0.)
        torch.testing.assert_close(force, torch.tensor([[0., 0., 1000.]]))

    def test_damping_acts_on_the_whole_velocity(self):
        """A body at the anchor moving sideways is damped against its velocity."""
        force = band_force(torch.tensor([[0., 0., 1.]]), torch.tensor([[.5, 0., 0.]]), torch.tensor([[0., 0., 1.]]), 1e4, 2e3)
        torch.testing.assert_close(force, torch.tensor([[-1000., 0., 0.]]))

    def test_matches_the_robocasa_band(self):
        """The force equals golem_robocasa's ElasticBand.evaluate_force at a displaced, moving body."""
        x, v, anchor = torch.tensor([[.03, -.02, .9]]), torch.tensor([[.1, .2, -.3]]), torch.tensor([[0., 0., 1.]])
        offset = x - anchor
        expected = -BAND_STIFFNESS * offset.norm() * offset / offset.norm() - BAND_DAMPING * v
        torch.testing.assert_close(band_force(x, v, anchor, BAND_STIFFNESS, BAND_DAMPING), expected)


class ElasticBandTests(unittest.TestCase):
    """Run ElasticBand against a stand-in articulation."""

    def test_anchor_is_below_the_torso_spawn(self):
        """The anchor is the torso's position at creation, lowered by the drop."""
        band = ElasticBand(stand_in_robot())
        torch.testing.assert_close(band.anchor, torch.tensor([[0., 0., 1. - BAND_DROP]]))
        self.assertTrue(band.enabled)

    def test_a_robot_without_the_body_is_refused(self):
        """A missing band body raises ValueError naming it."""
        with self.assertRaisesRegex(ValueError, 'no_such_link'):
            ElasticBand(stand_in_robot(), body='no_such_link')

    def test_apply_stores_the_pull_on_the_torso(self):
        """While enabled, apply() stores the band's pull on the torso alone, as a body-frame force."""
        robot = stand_in_robot()
        band = ElasticBand(robot)
        robot.data.body_pos_w[0, 1, 2] = .9
        band.apply()
        call = robot.permanent_wrench_composer.set_forces_and_torques.call_args.kwargs
        torch.testing.assert_close(call['forces'], torch.tensor([[[0., 0., BAND_STIFFNESS * (.1 - BAND_DROP)]]]))
        torch.testing.assert_close(call['torques'], torch.zeros(1, 1, 3))
        self.assertEqual(call['body_ids'].tolist(), [1])
        self.assertFalse(call['is_global'])

    def test_the_pull_follows_a_turned_torso(self):
        """With the torso rolled 90 deg about x, the upward pull is stored along the body's +y."""
        robot = stand_in_robot()
        band = ElasticBand(robot)
        robot.data.body_pos_w[0, 1, 2] = .9
        robot.data.body_quat_w[0, 1] = torch.tensor([2 ** -.5, 2 ** -.5, 0., 0.])
        band.apply()
        torch.testing.assert_close(stored_force(robot), torch.tensor([[[0., BAND_STIFFNESS * (.1 - BAND_DROP), 0.]]]),
                                   atol=1e-3, rtol=1e-5)

    def test_world_to_body_inverts_the_orientation(self):
        """A body yawed 90 deg sees the world's +x along its own -y."""
        yaw = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]])
        torch.testing.assert_close(world_to_body(yaw, torch.tensor([[1., 0., 0.]])), torch.tensor([[0., -1., 0.]]),
                                   atol=1e-6, rtol=0)

    def test_release_zeroes_the_stored_force_once(self):
        """After a release, the next apply() stores zero and the ones after it write nothing."""
        robot = stand_in_robot()
        band = ElasticBand(robot)
        band.apply()
        self.assertFalse(band.toggle())
        band.apply()
        torch.testing.assert_close(stored_force(robot), torch.zeros(1, 1, 3))
        band.apply()
        self.assertEqual(robot.permanent_wrench_composer.set_forces_and_torques.call_count, 2)

    def test_toggle_back_on_pulls_again(self):
        """A second toggle enables the band, and apply() stores its pull again."""
        robot = stand_in_robot()
        band = ElasticBand(robot)
        robot.data.body_pos_w[0, 1, 2] = .9
        band.toggle()
        band.apply()
        self.assertTrue(band.toggle())
        band.apply()
        self.assertGreater(float(stored_force(robot)[0, 0, 2]), 0.)


if __name__ == '__main__':
    unittest.main()
