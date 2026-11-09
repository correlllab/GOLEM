"""Test the policy specs (config/<robot>/policies/<name>.yaml) and their strict loader."""

import os

from ament_index_python.packages import get_package_share_directory
from golem_lowerbody_rl.policy import load_policy_spec, PolicyRunner
from golem_lowerbody_rl.policy.observation import ENCODER_KEYS, TERMS
from golem_lowerbody_rl.policy.spec import (
    CLIP_TARGET_KEYS, GROUP_KEYS, HEIGHT_KEYS, ONSET_KEYS, RANGE_KEYS, SECTION_KEYS,
    TOP_LEVEL_KEYS)
from golem_lowerbody_rl.profile import known_robots, load_profile
from golem_lowerbody_rl.run_config import load_run_config
import pytest
import yaml

SOURCE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
# the robot whose joints and policies the example carries
EXAMPLE_ROBOT = 'h12'


def _source(robot: str, name: str) -> dict:
    """Return the source tree's spec of robot's policy name."""
    path = os.path.join(SOURCE_CONFIG, robot, 'policies', f'{name}.yaml')
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _write(tmp_path, raw: dict, stem: str = 'spec') -> str:
    """Write raw as tmp_path/<stem>.yaml and return its path."""
    path = tmp_path / f'{stem}.yaml'
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return str(path)


def _runner(robot: str, path: str) -> PolicyRunner:
    """Return the runner of the spec at path on robot's sim run config."""
    return PolicyRunner(load_profile(robot), load_run_config(robot, 'sim'),
                        load_policy_spec(robot, path),
                        get_package_share_directory('golem_lowerbody_rl'))


def _term(raw: dict, name: str) -> dict:
    """Return the first term called name in raw's groups."""
    return next(term for group in raw['groups'] for term in group['terms']
                if term['term'] == name)


def test_every_listed_policy_loads():
    """Each robot's listed policies load, by name, under that name."""
    for robot in known_robots():
        for name in load_profile(robot).policies:
            spec = load_policy_spec(robot, name)
            assert spec.name == name
            assert spec.rate_hz == 50.0
            assert spec.reset_on == ('engage', 'band_release')
            assert spec.runtime.threads == 1


def test_unlisted_policy_is_refused():
    """A policy name the robot does not list is refused, naming the ones it does."""
    with pytest.raises(ValueError, match="unknown policy 'almi' of robot g1: one of homie"):
        load_policy_spec('g1', 'almi')


@pytest.mark.parametrize('where', ['observed_joints', 'default_pose', 'latent'])
def test_unknown_joint_is_refused(tmp_path, where):
    """A joint the robot's table does not have is refused at load, by name."""
    raw = _source('h12', 'fame')
    if where == 'observed_joints':
        raw['observed_joints'][3] = 'left_tail_joint'
    elif where == 'default_pose':
        raw['default_pose']['left_tail_joint'] = 0.1
    else:
        _term(raw, 'latent')['joints'][0] = 'left_tail_joint'
    with pytest.raises(ValueError, match="'left_tail_joint'"):
        load_policy_spec('h12', _write(tmp_path, raw))


def test_default_pose_must_hold_every_leg(tmp_path):
    """Each leg needs a default angle: it is the pre-pose and the action's offset."""
    raw = _source('h12', 'walk')
    del raw['default_pose']['right_knee_joint']
    with pytest.raises(ValueError, match="right_knee_joint"):
        load_policy_spec('h12', _write(tmp_path, raw))


def test_default_pose_of_an_unobserved_joint_is_refused(tmp_path):
    """A default angle is only for an observed joint or a leg."""
    raw = _source('h12', 'walk')
    raw['default_pose']['torso_joint'] = 0.0
    with pytest.raises(ValueError, match="'torso_joint'"):
        load_policy_spec('h12', _write(tmp_path, raw))


def test_unknown_term_is_refused(tmp_path):
    """A term the registry does not have is refused, naming the known ones."""
    raw = _source('h12', 'almi')
    raw['groups'][0]['terms'].append({'term': 'base_lin_vel', 'scale': 2.0})
    with pytest.raises(ValueError, match="unknown term 'base_lin_vel'.*ang_vel"):
        load_policy_spec('h12', _write(tmp_path, raw))


@pytest.mark.parametrize('edit, message', [
    (lambda raw: raw.pop('reset_on'), "missing key 'reset_on'"),
    (lambda raw: raw.update(rate=50), "unknown key 'rate'"),
    (lambda raw: raw['runtime'].pop('threads'), "missing key 'runtime.threads'"),
    (lambda raw: raw['command'].update(deadband=0.1), "unknown key 'command.deadband'"),
    (lambda raw: raw['command']['height'].pop('range'), "missing key 'command.height.range'"),
    (lambda raw: raw['action'].update(kp=[1.0] * 11), 'action.kp must be a list of 12'),
    (lambda raw: _term(raw, 'ang_vel').pop('scale'), "missing key 'groups.0.terms.0.scale'"),
    (lambda raw: _term(raw, 'gait_phase').update(form='square'), 'form must be one of'),
    (lambda raw: raw['runtime'].update(kind='tflite'), 'runtime.kind must be one of'),
    (lambda raw: raw.update(reset_on=['engage', 'onset']), 'reset_on must be'),
    (lambda raw: raw['command'].update(smooth_alpha=1.5), 'smooth_alpha must be in'),
    (lambda raw: raw['command']['onset_reset'].update(fall=0.2), 'fall must not exceed'),
])
def test_bad_spec_is_refused(tmp_path, edit, message):
    """A missing or unknown key, or a value out of range, is refused with its path."""
    raw = _source('h12', 'almi')
    edit(raw)
    with pytest.raises(ValueError, match=message):
        load_policy_spec('h12', _write(tmp_path, raw))


def test_onnx_runtime_has_no_reset_method(tmp_path):
    """An ONNX network keeps no state, so its spec names no reset method."""
    raw = _source('g1', 'homie')
    raw['runtime']['reset'] = 'reset_memory'
    with pytest.raises(ValueError, match='runtime.reset must be null'):
        load_policy_spec('g1', _write(tmp_path, raw))


def test_input_size_mismatch_is_refused(tmp_path, torch_threads):
    """A spec whose input size differs from the network's is refused at load, with both."""
    raw = _source('h12', 'walk')
    raw['groups'][0]['terms'] = [
        term for term in raw['groups'][0]['terms'] if term['term'] != 'gait_phase']
    with pytest.raises(ValueError, match='45 inputs.*47'):
        _runner('h12', _write(tmp_path, raw))


def test_onnx_input_size_mismatch_is_refused(tmp_path):
    """An ONNX network's declared input size is checked against the spec at load."""
    raw = _source('g1', 'homie')
    raw['groups'][0]['history'] = 5
    with pytest.raises(ValueError, match='380 inputs.*456'):
        _runner('g1', _write(tmp_path, raw))


def _key_paths(raw: dict) -> set:
    """
    Return the schema key paths of a spec, terms as 'term:<name>.<key>'.

    default_pose's keys are joint names, not schema keys, so they are left out.
    """
    paths = set()

    def walk(prefix: str, value) -> None:
        """Add the key paths under value."""
        if not isinstance(value, dict):
            return
        for key, inner in value.items():
            paths.add(f'{prefix}{key}')
            walk(f'{prefix}{key}.', inner)

    for key, value in raw.items():
        paths.add(key)
        if key not in ('default_pose', 'groups'):
            walk(f'{key}.', value)
    for group in raw['groups']:
        paths |= {f'groups.{key}' for key in group}
        for term in group['terms']:
            name = term['term']
            paths |= {f'term:{name}.{key}' for key in term if key != 'term'}
            if name == 'latent':
                paths |= {f'term:latent.encoder.{key}' for key in term['encoder']}
    return paths


def _schema_paths() -> set:
    """Return every key path of the spec schema, as _key_paths names them."""
    paths = set(TOP_LEVEL_KEYS)
    for section, keys in SECTION_KEYS.items():
        paths |= {f'{section}.{key}' for key in keys}
    paths |= {f'groups.{key}' for key in GROUP_KEYS}
    paths |= {f'command.trained_range.{key}' for key in RANGE_KEYS}
    paths |= {f'command.onset_reset.{key}' for key in ONSET_KEYS}
    paths |= {f'command.height.{key}' for key in HEIGHT_KEYS}
    paths |= {f'action.clip_targets.{key}' for key in CLIP_TARGET_KEYS}
    for name, term in TERMS.items():
        paths |= {f'term:{name}.{key}' for key in term.KEYS}
    paths |= {f'term:latent.encoder.{key}' for key in ENCODER_KEYS}
    return paths


def test_example_policy_holds_every_key_and_loads():
    """config/example.policy.yaml holds every key of the schema, each term, and loads."""
    path = os.path.join(SOURCE_CONFIG, 'example.policy.yaml')
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    assert _key_paths(raw) == _schema_paths()
    spec = load_policy_spec(EXAMPLE_ROBOT, path)
    assert spec.name == 'example.policy'


def test_real_specs_use_only_schema_keys():
    """Every installed spec's keys are schema keys (the loader would refuse others)."""
    schema = _schema_paths()
    for robot in known_robots():
        for name in load_profile(robot).policies:
            assert _key_paths(_source(robot, name)) <= schema, (robot, name)
