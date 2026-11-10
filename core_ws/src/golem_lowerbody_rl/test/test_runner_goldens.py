"""
Replay the golden traces in test/golden/ through the config-driven policy runner.

The traces hold what the policy classes the runner replaces computed from scripted inputs
(their recorder is test/golden/record_goldens.py at commit a4ea767; it needs those
classes). Each policy's runner, built from the robot's sim run config and its spec in
config/<robot>/policies/, gets the same inputs: the golden ``cmd`` and ``height`` as the
supervisor hands them on, a reset where ``reset`` is set (engage and band release, seeded
with that step's command), and ALMI's onset resets from its own command processing.

Arrays in every ``<policy>.npz`` (N = 120 steps, one per 20 ms; S = 27 body slots on the
H1-2, 29 on the G1):

- ``t`` (N,) float64, s: the node clock handed to the policy; 5.0 at step 0, 0.02 apart.
- ``q``, ``dq`` (N, S) float32, rad and rad/s: every body slot of ``rt/lowstate``.
- ``quat_wxyz`` (N, 4) float32: the IMU orientation, w first, a unit quaternion.
- ``gyro`` (N, 3) float32, rad/s.
- ``cmd`` (N, 3), m/s, m/s, rad/s: the raw velocity command; steps of zero, ramps up and
  down, and a hold, so ALMI's onset resets twice.
- ``height`` (N,) float64: the raw height command (1.0 on the H1-2; 0.74, then 0.60 from
  step 40 on the G1).
- ``reset`` (N,) bool: the policy is reset before this step (0, engage; 60, band release).
- ``net_input`` (N, I) and ``net_output`` (N, 12) float32: what the network got and
  returned; ``targets`` (N, 12) float32, rad: the leg targets in leg order.
- ``kp``, ``kd`` (12,): the leg gains.
- ALMI ``onset`` (N,) bool and ``cmd_policy`` (N, 3): its onset resets and the command the
  network observed before its scale. FAME ``enc_input`` (N, 21) and ``z`` (N, 8): its
  encoder's input and latent. HOMIE ``cmd_policy`` (N, 3) and ``height_policy`` (N,): the
  filtered commands its network observed.

The runner differs from the classes it replaces in two ways, and the goldens are
replayed bit for bit only with both reverted. Gravity comes from the normalised
quaternion (the one intended change): at the first step it moves the gravity inputs by
up to 6e-8. And every term and command is computed in float64 and stored as float32,
where the H1-2's classes computed in float32 and HOMIE's cast its command to float32:
that moves the other inputs by rounding, up to about 2e-9 at the first step (5e-8 in
ALMI's smoothed command). The policy's closed loop (its last action, its history and an
LSTM's memory) carries both through 120 steps to up to 3.3e-6 in the network's input
and output, so those are compared within LOOP_ATOL; the leg targets, gains and
processed commands stay within ATOL.
"""

import os

from ament_index_python.packages import get_package_share_directory
from golem_lowerbody_rl.policy import load_policy_spec, PolicyInput, PolicyRunner
from golem_lowerbody_rl.policy.observation import Latent
from golem_lowerbody_rl.profile import load_profile
from golem_lowerbody_rl.run_config import load_run_config
import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
ATOL = 1e-6
# the network input and output over a whole trace (module docstring)
LOOP_ATOL = 1e-5
# the first step's inputs: gravity's (normalisation and rounding), and every other one
# (rounding only)
GRAVITY_ATOL = 1e-7
ROUNDING_ATOL = 1e-8
POLICIES = ['walk', 'almi', 'fame', 'homie']
ROBOT = {'walk': 'h12', 'almi': 'h12', 'fame': 'h12', 'homie': 'g1'}


def _runner(name: str) -> PolicyRunner:
    """Return a fresh runner of policy name on its robot's sim run config."""
    if name == 'homie':
        pytest.importorskip('onnxruntime', reason='HOMIE runs on onnxruntime')
    robot = ROBOT[name]
    return PolicyRunner(load_profile(robot), load_run_config(robot, 'sim'),
                        load_policy_spec(robot, name),
                        get_package_share_directory('golem_lowerbody_rl'))


def _golden(name: str):
    """Return policy name's golden trace."""
    return np.load(os.path.join(GOLDEN_DIR, f'{name}.npz'))


def _input(golden, i: int) -> PolicyInput:
    """Return the golden trace's input at step i."""
    return PolicyInput(
        t=float(golden['t'][i]), q=golden['q'][i], dq=golden['dq'][i],
        quat_wxyz=golden['quat_wxyz'][i], gyro=golden['gyro'][i], cmd=golden['cmd'][i],
        height=float(golden['height'][i]))


def _close(actual, expected, what: str, atol: float = ATOL) -> None:
    """Assert actual equals the golden expected within atol."""
    actual = np.asarray(actual)
    assert actual.shape == expected.shape, what
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=what)


@pytest.mark.parametrize('name', POLICIES)
def test_runner_matches_golden(name: str, torch_threads) -> None:
    """The runner computes each golden step's network input and output, targets and gains."""
    golden = _golden(name)
    runner = _runner(name)
    assert runner.name == name
    latent = [term for term in runner.observation.terms if isinstance(term, Latent)]

    net_input, net_output, targets, onsets, commands, heights = [], [], [], [], [], []
    enc_input, z = [], []
    for i in range(golden['t'].size):
        if golden['reset'][i]:
            runner.reset(golden['cmd'][i])
        legs = runner.step(_input(golden, i))
        net_input.append(runner.last_input)
        net_output.append(runner.last_output)
        targets.append(legs.q)
        onsets.append(runner.onset)
        commands.append(runner.command)
        heights.append(runner.height)
        _close(legs.kp, golden['kp'], f'kp at step {i}')
        _close(legs.kd, golden['kd'], f'kd at step {i}')
        for term in latent:
            enc_input.append(term.last_input)
            z.append(term.last_latent)

    _close(np.stack(net_input), golden['net_input'], 'net_input', LOOP_ATOL)
    _close(np.stack(net_output), golden['net_output'], 'net_output', LOOP_ATOL)
    _close(np.stack(targets), golden['targets'], 'targets')
    if name == 'almi':
        np.testing.assert_array_equal(np.array(onsets), golden['onset'])
        _close(np.stack(commands), golden['cmd_policy'], 'cmd_policy')
    else:
        assert not any(onsets)
    if name == 'homie':
        _close(np.stack(commands), golden['cmd_policy'], 'cmd_policy')
        _close(np.array(heights), golden['height_policy'], 'height_policy')
    if name == 'fame':
        assert len(latent) == 1
        _close(np.stack(enc_input), golden['enc_input'], 'enc_input')
        _close(np.stack(z), golden['z'], 'z')
    else:
        assert not latent
    if runner.spec.runtime.kind == 'torchscript':
        import torch
        assert torch.get_num_threads() == runner.spec.runtime.threads == 1


@pytest.mark.parametrize('name', POLICIES)
def test_first_step_differs_by_gravity_and_rounding(name: str, torch_threads) -> None:
    """The first step's input differs from the golden's by gravity (< 1e-7), else rounding."""
    golden = _golden(name)
    runner = _runner(name)
    runner.reset(golden['cmd'][0])
    runner.step(_input(golden, 0))
    gravity = np.zeros(runner.observation.size, dtype=bool)
    for where in runner.observation.newest('projected_gravity'):
        gravity[where] = True
    assert gravity.sum() == 3
    _close(runner.last_input[gravity], golden['net_input'][0][gravity], 'gravity',
           GRAVITY_ATOL)
    _close(runner.last_input[~gravity], golden['net_input'][0][~gravity], 'the rest',
           ROUNDING_ATOL)
    _close(runner.last_output, golden['net_output'][0], 'net_output')
