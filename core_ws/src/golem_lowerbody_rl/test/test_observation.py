"""Test the observation terms, the runner's checks and ALMI's onset reset."""

import os
import subprocess
import sys
import textwrap

from ament_index_python.packages import get_package_share_directory
from golem_lowerbody_rl.policy import (
    CommandProcessor, load_policy_spec, PolicyError, PolicyInput, PolicyRunner)
from golem_lowerbody_rl.policy.observation import GaitPhase, Latent, StepData
from golem_lowerbody_rl.profile import load_profile
from golem_lowerbody_rl.run_config import load_run_config
import numpy as np
import pytest
import yaml

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
SOURCE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
DT = 0.02  # s, one 50 Hz policy step
ATOL = 1e-6
# FAME's latent at its golden step 0 with the hand forces FORCES, as its policy class
# computed it (policy_classes.FamePolicy._encode at commit a4ea767)
FORCES = [[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]]
OLD_FAME_Z = [0.03601088374853134, -1.5208951234817505, 1.9404932260513306,
              2.3350117206573486, 0.4966792166233063, 0.23099984228610992,
              0.6272552609443665, -0.8404679298400879]


def _runner(robot: str, policy: str) -> PolicyRunner:
    """Return a fresh runner of robot's policy (a name or a spec path) on its sim config."""
    return PolicyRunner(load_profile(robot), load_run_config(robot, 'sim'),
                        load_policy_spec(robot, policy),
                        get_package_share_directory('golem_lowerbody_rl'))


def _golden_input(name: str, step: int, **change) -> PolicyInput:
    """Return the golden trace's input at step, with the fields in change replaced."""
    golden = np.load(os.path.join(GOLDEN_DIR, f'{name}.npz'))
    fields = dict(t=float(golden['t'][step]), q=golden['q'][step].copy(),
                  dq=golden['dq'][step].copy(), quat_wxyz=golden['quat_wxyz'][step].copy(),
                  gyro=golden['gyro'][step].copy(), cmd=golden['cmd'][step].copy(),
                  height=float(golden['height'][step]))
    fields.update(change)
    return PolicyInput(**fields)


class CountingRuntime:
    """Stand in for a runner's runtime: run the real one and count its resets."""

    def __init__(self, runtime):
        """Wrap runtime."""
        self._runtime = runtime
        self.resets = 0

    def __call__(self, x):
        """Run the wrapped runtime on x."""
        return self._runtime(x)

    def reset(self):
        """Reset the wrapped runtime and count the call."""
        self.resets += 1
        self._runtime.reset()


def _with_slot(name: str, field: str, slot: int, value: float) -> np.ndarray:
    """Return golden step 0's field array with slot set to value."""
    array = getattr(_golden_input(name, 0), field).copy()
    array[slot] = value
    return array


@pytest.mark.parametrize('field, bad', [
    ('q', lambda: _with_slot('almi', 'q', 3, np.nan)),
    ('dq', lambda: _with_slot('almi', 'dq', 14, np.inf)),
    ('gyro', lambda: np.array([0.0, np.nan, 0.0], dtype=np.float32)),
    ('quat_wxyz', lambda: np.zeros(4, dtype=np.float32)),
    ('cmd', lambda: np.array([np.nan, 0.0, 0.0], dtype=np.float32)),
    ('height', lambda: float('inf')),
])
def test_observation_is_finite_or_raises(field, bad, torch_threads):
    """A non-finite input the policy uses raises PolicyError and leaves the runner as it was."""
    faulted, fresh = _runner('h12', 'almi'), _runner('h12', 'almi')
    with pytest.raises(PolicyError):
        faulted.step(_golden_input('almi', 0, **{field: bad()}))
    for runner in (faulted, fresh):
        runner.step(_golden_input('almi', 0))
        assert np.all(np.isfinite(runner.last_input))
    np.testing.assert_array_equal(faulted.last_input, fresh.last_input)
    np.testing.assert_array_equal(faulted.last_output, fresh.last_output)


def test_unobserved_joint_is_not_checked(torch_threads):
    """A non-finite value in a slot the policy does not observe (a wrist) is not its fault."""
    runner = _runner('h12', 'almi')
    runner.step(_golden_input('almi', 0, q=_with_slot('almi', 'q', 18, np.nan)))
    assert np.all(np.isfinite(runner.last_input))


class NanRuntime:
    """Stand in for a runtime whose network returns NaN."""

    def __call__(self, x):
        """Return 12 NaN actions."""
        return np.full(12, np.nan, dtype=np.float32)

    def reset(self):
        """Keep no state."""


def test_non_finite_network_output_raises(torch_threads):
    """A network output that is not finite raises PolicyError instead of targets."""
    runner = _runner('h12', 'walk')
    runner.runtime = NanRuntime()
    with pytest.raises(PolicyError, match='network output'):
        runner.step(_golden_input('walk', 0))


def _phase(command, form='left_right_sin', stand_below=0.1, phase_time=0.1):
    """Return the gait_phase term's value for a processed command."""
    raw = {'period': 0.8, 'form': form, 'offset': 0.5 if form == 'left_right_sin' else None,
           'stand_below': stand_below}
    term = GaitPhase(GaitPhase.parse('gait_phase', raw), None)
    zeros = np.zeros(12, dtype=np.float32)
    step = StepData(inp=_golden_input('almi', 0), command=np.asarray(command, dtype=np.float64),
                    height=1.0, phase_time=phase_time, last_action=zeros,
                    last_action_clipped=zeros)
    return term(step)


def test_almi_stand_gate_zeroes_the_phase_below_0_1():
    """The phase ALMI observes is (0, 0) while the processed command's norm is below 0.1."""
    np.testing.assert_array_equal(_phase([0.0, 0.0, 0.0]), [0.0, 0.0])
    np.testing.assert_array_equal(_phase([0.06, 0.0, 0.0799]), [0.0, 0.0])
    phase = 0.1 / 0.8
    expected = [np.sin(2 * np.pi * phase), np.sin(2 * np.pi * (phase + 0.5))]
    np.testing.assert_allclose(_phase([0.06, 0.0, 0.08]), expected)
    np.testing.assert_allclose(_phase([0.0, 0.0, 0.0], form='sin_cos', stand_below=None),
                               [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])


def test_almi_runner_observes_a_zero_phase_when_standing(torch_threads):
    """At zero command ALMI's last two inputs, its phase, are zero."""
    runner = _runner('h12', 'almi')
    runner.reset()
    runner.step(_golden_input('almi', 0, cmd=np.zeros(3, dtype=np.float32)))
    np.testing.assert_array_equal(runner.last_input[63:65], [0.0, 0.0])


# (raw request, steps): rise, hold, dip between the thresholds and back, fall, dither
# below the rise threshold, rise again
ONSET_SCRIPT = [
    ([0.0, 0.0, 0.0], 10),
    ([0.6, 0.0, 0.3], 30),
    ([0.05, 0.0, 0.0], 60),  # effective 0.075: under rise, over fall, so still walking
    ([0.6, 0.0, 0.3], 30),
    ([0.0, 0.0, 0.0], 60),  # effective falls under 0.05: re-armed
    ([0.05, 0.0, 0.0], 60),  # effective 0.075 again, under rise: no onset
    ([0.6, 0.0, 0.3], 30),
]


def _script():
    """Return ONSET_SCRIPT as one request per step."""
    return [np.array(cmd) for cmd, steps in ONSET_SCRIPT for _ in range(steps)]


def test_onset_resets_once_per_rise():
    """The command processing of ALMI reports one onset per rise through 0.1 (hysteresis)."""
    commands = CommandProcessor(load_policy_spec('h12', 'almi').command,
                                load_run_config('h12', 'sim').commands)
    onsets = []
    for i, cmd in enumerate(_script()):
        out = commands.process(cmd, DT)
        if commands.onset:
            onsets.append(i)
            # the onset hands on the seed itself: gain, clip, no smoothing
            np.testing.assert_allclose(out, np.clip(cmd * [1.5, 1.0, 2.0], -1.0, 1.0))
    assert len(onsets) == 2
    assert 10 <= onsets[0] < 40
    assert 10 + 30 + 60 + 30 + 60 + 60 <= onsets[1]


def test_onset_resets_the_policy(torch_threads):
    """Each onset resets the network's memory once, and only ALMI's processing has onsets."""
    runner = _runner('h12', 'almi')
    runner.runtime = counting = CountingRuntime(runner.runtime)
    golden = _golden_input('almi', 0)
    onsets = 0
    for cmd in _script():
        runner.step(PolicyInput(t=golden.t, q=golden.q, dq=golden.dq,
                                quat_wxyz=golden.quat_wxyz, gyro=golden.gyro,
                                cmd=cmd.astype(np.float32), height=golden.height))
        onsets += runner.onset
    assert onsets == counting.resets == 2


def test_fame_hand_forces_reach_the_encoder(tmp_path, torch_threads):
    """The latent of FAME encodes the upper-body positions and the spec's hand forces."""
    path = os.path.join(SOURCE_CONFIG, 'h12', 'policies', 'fame.yaml')
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    latent = next(term for group in raw['groups'] for term in group['terms']
                  if term['term'] == 'latent')
    latent['forces'] = FORCES
    spec_path = tmp_path / 'fame_forces.yaml'
    spec_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    runner = _runner('h12', str(spec_path))
    runner.reset()
    inp = _golden_input('fame', 0)
    runner.step(inp)
    (term,) = [term for term in runner.observation.terms if isinstance(term, Latent)]
    np.testing.assert_array_equal(term.last_input,
                                  np.concatenate([inp.q[12:27], *FORCES]).astype(np.float32))
    np.testing.assert_allclose(term.last_latent, OLD_FAME_Z, rtol=0, atol=ATOL)


def test_fame_observes_the_height_command_unscaled(torch_threads):
    """The newest frame of FAME holds the squat command as given; nothing else changes."""
    golden = np.load(os.path.join(GOLDEN_DIR, 'fame.npz'))
    runner = _runner('h12', 'fame')
    runner.reset()
    runner.step(_golden_input('fame', 0, height=0.8))
    expected = golden['net_input'][0].copy()
    expected[2 * 76 + 3] = 0.8  # the newest of three 76-value frames; height is its 4th value
    np.testing.assert_allclose(runner.last_input, expected, rtol=0, atol=ATOL)


def test_onnx_is_not_imported_for_torchscript_policies():
    """Running ALMI, TorchScript only, never imports onnxruntime, which its host may lack."""
    pytest.importorskip('torch', reason='the H1-2 policies run on torch')
    code = textwrap.dedent("""
        import sys
        import numpy as np
        from ament_index_python.packages import get_package_share_directory
        import golem_lowerbody_rl.policy as policy
        from golem_lowerbody_rl.profile import load_profile
        from golem_lowerbody_rl.run_config import load_run_config
        runner = policy.PolicyRunner(
            load_profile('h12'), load_run_config('h12', 'sim'),
            policy.load_policy_spec('h12', 'almi'),
            get_package_share_directory('golem_lowerbody_rl'))
        runner.reset()
        runner.step(policy.PolicyInput(
            t=0.0, q=np.zeros(27, np.float32), dq=np.zeros(27, np.float32),
            quat_wxyz=np.array([1, 0, 0, 0], np.float32), gyro=np.zeros(3, np.float32),
            cmd=np.zeros(3, np.float32), height=1.0))
        assert 'onnxruntime' not in sys.modules, 'onnxruntime was imported'
        print('ok')
    """)
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith('ok')
