"""The new loader against what the old safety layers derive from the same configs.

The fixtures under test/fixtures/limits/<robot>/<config>.json are written by
tools/snapshot_old_limits.py from h12_safety_layer.core.config and
g1_safety_layer.core.config. This test imports neither package: it runs
golem_safety_dump_config and compares its output to the fixtures.
"""
import json
from pathlib import Path
import shutil
import subprocess

from ament_index_python.packages import get_package_prefix, get_package_share_directory
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / 'fixtures' / 'limits'

CONFIGS = {
    'h12': [
        'default_safety_full', 'default_safety_split', 'locomotion_safety_full',
        'relax_safety_full', 'relax_safety_split', 'sim_safety_split',
        'tight_safety_full', 'tight_safety_split',
    ],
    'g1': ['arm_sdk', 'full', 'sim_arm_sdk', 'sim_split', 'split'],
}
CASES = [f'{robot}/{name}' for robot, names in CONFIGS.items() for name in names]

LIMIT_ARRAYS = (
    'q_clip_low', 'q_clip_high', 'dq_clip', 'tau_clip',
    'q_estop_low', 'q_estop_high', 'dq_estop', 'tau_estop',
)
JOINTS = {'h12': 27, 'g1': 29}
MODE_MACHINE = {'h12': 6, 'g1': 5}
# The five old topic keys and their new names; a DDS topic rt/x is the ROS topic /x.
TOPIC_KEYS = ('low_cmd_in', 'low_cmd_lower_in', 'low_cmd_upper_in', 'low_cmd_out', 'low_state')


def dump_executable():
    prefix = Path(get_package_prefix('golem_safety'))
    installed = prefix / 'lib' / 'golem_safety' / 'golem_safety_dump_config'
    if installed.is_file():
        return str(installed)
    on_path = shutil.which('golem_safety_dump_config')
    assert on_path, f'golem_safety_dump_config is neither on PATH nor at {installed}'
    return on_path


def dump(robot, config):
    result = subprocess.run(
        [dump_executable(), robot, config], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def fixture(case):
    robot, name = case.split('/')
    return json.loads((FIXTURES / robot / f'{name}.json').read_text())


@pytest.mark.parametrize('case', CASES)
def test_limits_match_old_loaders(case):
    robot, name = case.split('/')
    old = fixture(case)['limits']
    new = dump(robot, name)['limits']
    for key in LIMIT_ARRAYS:
        assert len(new[key]) == JOINTS[robot], key
        np.testing.assert_allclose(new[key], old[key], rtol=0, atol=1e-12, err_msg=key)
    assert new['kp_max'] == pytest.approx(old['kp_max'], rel=0, abs=1e-12)
    assert new['kd_max'] == pytest.approx(old['kd_max'], rel=0, abs=1e-12)


@pytest.mark.parametrize('case', CASES)
def test_scalars_match_old_loaders(case):
    robot, name = case.split('/')
    old = fixture(case)['old']
    new = dump(robot, name)

    assert new['robot'] == robot
    assert new['mode_machine'] == MODE_MACHINE[robot]
    assert new['mode'] == old['mode']
    assert new['control']['publish_hz'] == old['publish_hz']
    for key in TOPIC_KEYS:
        assert old['topics'][key].startswith('rt/'), key
        assert new['topics'][key] == '/' + old['topics'][key][len('rt/'):], key

    # Logging keeps each file's own values; only the G1 enabled it.
    assert new['logging'] == old['logging']

    # The stale times. A key the old loader had carries its effective value; the others take
    # 0.2 and 0.1. The old upper watchdog ran in split_mode and arm_sdk_mode only.
    upper = old['upper_stale_seconds'] if old['upper_stale_active'] else 0.0
    assert new['stale']['upper_seconds'] == upper
    assert new['stale']['lower_seconds'] == (
        old['lower_stale_seconds'] if old['lower_stale_seconds'] is not None else 0.2)
    assert new['stale']['state_seconds'] == (
        old['state_stale_seconds'] if old['state_stale_seconds'] is not None else 0.1)

    # estop.state_lag_ms moved to control.state_lag_ms; the G1 had none and takes 100.
    assert new['control']['state_lag_ms'] == (
        old['state_lag_ms'] if old['state_lag_ms'] is not None else 100.0)

    # The H1-2 streams zeros and then goes silent; the G1 waits for the engage gate and keeps
    # writing its estop command.
    if robot == 'h12':
        assert new['startup']['rule'] == 'zero_stream'
        assert new['estop_response']['after'] == 'burst_then_silence'
        assert new['estop_response']['burst_count'] == 50
        assert new['estop_response']['mode'] == 'zero'
        assert new['estop_response']['kd'] == [0.0] * JOINTS[robot]
        assert old['engage'] is None and old['arm_sdk'] is None
    else:
        assert new['startup']['rule'] == 'engage_gate'
        assert new['estop_response']['after'] == 'keep_writing'
        assert new['estop_response']['mode'] == old['estop_response']['mode']
        kd = old['estop_response']['kd']
        assert new['estop_response']['kd'] == (kd if kd is not None else [0.0] * JOINTS[robot])
        # engage.mode_machine moved to the profile, and must be what it was.
        assert old['engage']['mode_machine'] == MODE_MACHINE[robot]
        for key in ('max_state_age_seconds', 'max_position_error'):
            assert new['engage'][key] == old['engage'][key], key
        if old['mode'] == 'arm_sdk_mode':
            for key in ('engage_seconds', 'release_seconds', 'waist_kp_min'):
                assert new['arm_sdk'][key] == old['arm_sdk'][key], key


def test_every_installed_config_has_a_fixture():
    share = Path(get_package_share_directory('golem_safety')) / 'config'
    for robot, names in CONFIGS.items():
        installed = sorted(path.stem for path in (share / robot).glob('*.yaml'))
        assert installed == sorted(names), robot
        fixtures = sorted(path.stem for path in (FIXTURES / robot).glob('*.json'))
        assert fixtures == sorted(names), robot

