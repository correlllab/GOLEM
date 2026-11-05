"""Test the run configs (config/<robot>/{sim,real}.yaml) and their strict loader."""

import os

import pytest
import yaml

from golem_lowerbody_rl.run_config import load_run_config

SOURCE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')

# (robot, run) -> the values that differ between the four run configs
EXPECTED = {
    ('h12', 'sim'): dict(
        sim=True, wait_for_confirm=False, band_release=True, max_wait=34.0,
        vx=(-1.0, 1.0), vy=(-1.0, 1.0), yaw=(-1.0, 1.0), accel=None, height_rate=None,
        max_state_gap=0.5, max_latency=None, imu_trim_deg=(0.0, 0.0, 0.0)),
    ('h12', 'real'): dict(
        sim=False, wait_for_confirm=True, band_release=False, max_wait=34.0,
        vx=(-1.0, 1.0), vy=(-1.0, 1.0), yaw=(-1.0, 1.0), accel=None, height_rate=None,
        max_state_gap=0.25, max_latency=None, imu_trim_deg=(0.5, -2.5, 0.0)),
    ('g1', 'sim'): dict(
        sim=True, wait_for_confirm=False, band_release=True, max_wait=30.0,
        vx=(-0.8, 1.2), vy=(-0.5, 0.5), yaw=(-0.8, 0.8), accel=(0.5, 0.5, 1.0),
        height_rate=0.1, max_state_gap=0.5, max_latency=None, imu_trim_deg=(0.0, 0.0, 0.0)),
    ('g1', 'real'): dict(
        sim=False, wait_for_confirm=True, band_release=False, max_wait=30.0,
        vx=(-0.3, 0.5), vy=(-0.3, 0.3), yaw=(-0.6, 0.6), accel=(0.5, 0.5, 1.0),
        height_rate=0.1, max_state_gap=0.25, max_latency=None, imu_trim_deg=(0.0, 0.0, 0.0)),
}


@pytest.mark.parametrize('robot, run', sorted(EXPECTED), ids=lambda value: str(value))
def test_values(robot, run):
    """Each run config holds the values of its robot and run kind."""
    expected = EXPECTED[(robot, run)]
    config = load_run_config(robot, run)
    assert config.sim is expected['sim']
    assert config.startup.wait_for_confirm is expected['wait_for_confirm']
    assert config.startup.ramp_seconds == 2.0
    assert config.startup.settle_tolerance == 0.15
    assert config.startup.settle_ticks == 10
    assert config.startup.settle_max_ticks == 400
    assert config.startup.steady_seconds == 1.0
    assert config.startup.tick_rate == (0.9, 1.1)
    assert config.startup.upper_fresh_seconds == 0.5
    assert config.heartbeat.timeout == 0.5
    assert config.band.release is expected['band_release']
    assert config.band.max_wait == expected['max_wait']
    assert config.commands.timeout == 0.5
    for axis in ('vx', 'vy', 'yaw'):
        assert getattr(config.commands, axis) == expected[axis], axis
    assert config.commands.accel == expected['accel']
    assert config.commands.height_rate == expected['height_rate']
    assert config.faults.max_state_gap == expected['max_state_gap']
    assert config.faults.max_latency == expected['max_latency']
    assert config.imu_trim_deg == expected['imu_trim_deg']


def test_a_path_loads_as_its_name():
    """A run config given by path loads as the same config given by name."""
    path = os.path.join(SOURCE_CONFIG, 'g1', 'real.yaml')
    assert load_run_config('g1', path) == load_run_config('g1', 'real')


def _source(robot, run):
    """Return the source tree's config/<robot>/<run>.yaml as a mapping."""
    with open(os.path.join(SOURCE_CONFIG, robot, f'{run}.yaml'), encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _write(tmp_path, raw, name='custom.yaml'):
    """Write raw as YAML to tmp_path/name and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return str(path)


def test_unknown_key_is_refused(tmp_path):
    """An unknown key, at the top or in a section, is refused by name."""
    raw = _source('h12', 'sim')
    raw['startup']['ramp_secs'] = 2.0
    with pytest.raises(ValueError, match="unknown key 'startup.ramp_secs'"):
        load_run_config('h12', _write(tmp_path, raw))
    raw = _source('h12', 'sim')
    raw['tick_seconds'] = 0.001
    with pytest.raises(ValueError, match="unknown key 'tick_seconds'"):
        load_run_config('h12', _write(tmp_path, raw))


def test_missing_key_is_refused(tmp_path):
    """A missing key, at the top or in a section, is refused by name."""
    raw = _source('g1', 'real')
    del raw['faults']['max_latency']
    with pytest.raises(ValueError, match="missing key 'faults.max_latency'"):
        load_run_config('g1', _write(tmp_path, raw))
    raw = _source('g1', 'real')
    del raw['imu_trim_deg']
    with pytest.raises(ValueError, match="missing key 'imu_trim_deg'"):
        load_run_config('g1', _write(tmp_path, raw))


@pytest.mark.parametrize('axis, limits', [('vx', [0.1, 1.0]), ('yaw', [-1.0, -0.1])])
def test_limit_must_contain_zero(tmp_path, axis, limits):
    """A command range without zero is refused: the node falls back to zero velocity."""
    raw = _source('g1', 'sim')
    raw['commands'][axis] = limits
    with pytest.raises(ValueError, match=f'commands.{axis}'):
        load_run_config('g1', _write(tmp_path, raw))


@pytest.mark.parametrize('run, sim', [('sim', False), ('real', True)])
def test_sim_must_match_the_file_role(tmp_path, run, sim):
    """A sim.yaml must say sim: true and a real.yaml sim: false."""
    raw = _source('h12', run)
    raw['sim'] = sim
    with pytest.raises(ValueError, match="'sim'"):
        load_run_config('h12', _write(tmp_path, raw, name=f'{run}.yaml'))


@pytest.mark.parametrize('section, key, value, message', [
    ('startup', 'tick_rate', [1.05, 1.2], 'startup.tick_rate'),
    ('startup', 'settle_ticks', 2.5, 'startup.settle_ticks'),
    ('heartbeat', 'timeout', 0.1, 'heartbeat.timeout'),
    ('commands', 'accel', [0.5, 0.5], 'commands.accel'),
    ('commands', 'height_rate', -0.1, 'commands.height_rate'),
    ('faults', 'max_state_gap', float('nan'), 'faults.max_state_gap'),
    ('band', 'release', 'yes', 'band.release'),
])
def test_bad_value_is_refused(tmp_path, section, key, value, message):
    """A value of the wrong type or out of range is refused, naming its key."""
    raw = _source('g1', 'sim')
    raw[section][key] = value
    with pytest.raises(ValueError, match=message):
        load_run_config('g1', _write(tmp_path, raw))


def test_unknown_run_name_is_refused():
    """A name that is not a run config of the robot is refused."""
    with pytest.raises(ValueError, match="'robot'"):
        load_run_config('h12', 'robot')
