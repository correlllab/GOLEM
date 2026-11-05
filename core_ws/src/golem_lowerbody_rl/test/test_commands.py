"""
Test the command processing between /cmd_vel and a policy's velocity and height inputs.

The goldens hold commands inside every limit, so the clip order, the trained range, the
run config's limits and the height processing are pinned here. Expected values follow
the code each policy ran before: the H1-2 node's per-axis clip to +-1 then ALMI's
clip(cmd * cmd_gain, +-1) and its smoothing; g1_homie's CommandFilter (trained range,
the run config's limits, then the acceleration and height-rate limits at 50 Hz).
"""

import dataclasses

from golem_lowerbody_rl.policy import CommandProcessor, load_policy_spec
from golem_lowerbody_rl.run_config import load_run_config
import numpy as np
import pytest

DT = 0.02  # s, one 50 Hz policy step


def _processor(robot: str, policy: str, run: str = 'sim', **limits) -> CommandProcessor:
    """Return the command processor of robot's policy on its run config, limits replaced."""
    commands = load_run_config(robot, run).commands
    return CommandProcessor(load_policy_spec(robot, policy).command,
                            dataclasses.replace(commands, **limits))


def test_run_limits_clip_before_the_gain():
    """The run config's limits apply to the request, before ALMI's gain and +-1 clip."""
    commands = _processor('h12', 'almi', vx=(-0.5, 0.5))
    commands.reset([0.9, 0.0, 0.0])
    # 0.9 -> 0.5 (limit) -> 0.75 (gain 1.5); the gain first would give 1.0 -> 0.5
    np.testing.assert_allclose(commands.process([0.9, 0.0, 0.0], DT), [0.75, 0.0, 0.0])


def test_gain_then_clip_to_one():
    """ALMI multiplies by its gain and clips each axis to +-1."""
    commands = _processor('h12', 'almi')
    commands.reset([0.8, -0.3, 0.6])
    np.testing.assert_allclose(commands.process([0.8, -0.3, 0.6], DT), [1.0, -0.3, 1.0])


def test_run_limits_hold_without_a_gain():
    """The command WALK observes is the request clipped to the run config's +-1 per axis."""
    commands = _processor('h12', 'walk')
    np.testing.assert_allclose(commands.process([1.7, -0.4, -3.0], DT), [1.0, -0.4, -1.0])


def test_smoothing_moves_alpha_of_the_way():
    """The smoothing of ALMI moves 0.15 of the way to the gained command each step."""
    commands = _processor('h12', 'almi')
    np.testing.assert_allclose(commands.process([0.02, 0.0, 0.0], DT), [0.0045, 0.0, 0.0])
    np.testing.assert_allclose(commands.process([0.02, 0.0, 0.0], DT),
                               [0.0045 + 0.15 * (0.03 - 0.0045), 0.0, 0.0])


def test_trained_range_and_run_limits_then_acceleration():
    """The command HOMIE observes is clipped to the narrower range and moves at accel."""
    commands = _processor('g1', 'homie', run='real')  # vx [-0.3, 0.5], vy +-0.3, yaw +-0.6
    np.testing.assert_allclose(commands.process([2.0, -1.0, 1.0], DT), [0.01, -0.01, 0.02])
    for _ in range(100):
        cmd = commands.process([2.0, -1.0, 1.0], DT)
    np.testing.assert_allclose(cmd, [0.5, -0.3, 0.6])


def test_trained_range_holds_inside_wider_run_limits():
    """The trained range of HOMIE clips a request the run config's limits would pass."""
    commands = _processor('g1', 'homie', vx=(-2.0, 2.0), vy=(-2.0, 2.0), yaw=(-2.0, 2.0))
    for _ in range(200):
        cmd = commands.process([2.0, -2.0, 2.0], DT)
    np.testing.assert_allclose(cmd, [1.2, -0.5, 0.8])


def test_height_clips_to_its_range_and_moves_at_its_rate():
    """The height HOMIE observes starts at 0.74, clips to [0.24, 0.74] and moves 0.1 m/s."""
    commands = _processor('g1', 'homie')
    assert commands.process_height(0.9, DT) == pytest.approx(0.74)
    assert commands.process_height(0.1, DT) == pytest.approx(0.738)
    for _ in range(300):
        height = commands.process_height(0.1, DT)
    assert height == pytest.approx(0.24)


def test_height_without_range_or_rate_passes_through():
    """FAME observes the squat command as it comes: no range, no rate limit."""
    commands = _processor('h12', 'fame')
    assert commands.height == 1.0
    assert commands.process_height(0.8, DT) == 0.8
    assert commands.process_height(1.3, DT) == 1.3


def test_reset_reseeds_the_smoothing_filter():
    """A reset seeds ALMI's filter with the gained, clipped seed command."""
    commands = _processor('h12', 'almi')
    for _ in range(5):
        commands.process([0.0, 0.0, 0.0], DT)
    commands.reset([0.6, 0.0, 0.3])
    np.testing.assert_allclose(commands.process([0.6, 0.0, 0.3], DT), [0.9, 0.0, 0.6])


def test_reset_keeps_the_rate_limits():
    """A reset leaves HOMIE's rate-limited command and height where they are."""
    commands = _processor('g1', 'homie')
    for _ in range(3):
        cmd = commands.process([0.5, 0.0, 0.0], DT)
        height = commands.process_height(0.5, DT)
    np.testing.assert_allclose(cmd, [0.03, 0.0, 0.0])
    commands.reset([0.0, 0.0, 0.0])
    np.testing.assert_allclose(commands.process([0.5, 0.0, 0.0], DT), [0.04, 0.0, 0.0])
    assert commands.process_height(0.5, DT) == pytest.approx(height - 0.002)


@pytest.mark.parametrize('bad', [[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0]])
def test_bad_command_is_refused_without_changing_state(bad):
    """A non-finite or misshapen command raises ValueError and changes nothing."""
    commands = _processor('g1', 'homie')
    commands.process([0.5, 0.0, 0.0], DT)
    with pytest.raises(ValueError):
        commands.process(bad, DT)
    with pytest.raises(ValueError):
        commands.process_height(np.nan, DT)
    np.testing.assert_allclose(commands.process([0.5, 0.0, 0.0], DT), [0.02, 0.0, 0.0])
    assert commands.height == 0.74
