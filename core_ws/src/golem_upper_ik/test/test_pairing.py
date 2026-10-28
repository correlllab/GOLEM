'''
Which safety and controller configs may run together, and what the pairs agree on.

The G1's verdicts are compared with the previous G1 arm controller's config_pairing, through
test/fixtures/old_controller/g1_pairing.json (test/fixtures/README.md); this test imports no
old package. The pairs' topics and gain limits are read from golem_safety's installed configs.
'''
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from golem_upper_ik.pairing import default_controller_config, pairing_refusal

FIXTURE = Path(__file__).parent / 'fixtures' / 'old_controller' / 'g1_pairing.json'

H12_PAIRS = {
    'relax_safety_split': 'safety_split',
    'default_safety_split': 'safety_split',
    'tight_safety_split': 'safety_split',
    'sim_safety_split': 'sim_safety_split',
    'relax_safety_full': 'safety_full',
    'default_safety_full': 'safety_full',
    'tight_safety_full': 'safety_full',
    'locomotion_safety_full': 'safety_full',
}
H12_CONTROLLERS = ('safety_split', 'sim_safety_split', 'safety_full')
SIMS = (None, False, True)


def old_cases():
    return json.loads(FIXTURE.read_text())


def test_g1_pairing_matches_old_rules():
    cases = old_cases()['cases']
    assert len(cases) == 294
    for case in cases:
        args = (case['controller'], case['safety'])
        kwargs = {'sim': case['sim'], 'locomotion': case['locomotion']}
        if case['verdict'] == 'raises':
            with pytest.raises(Exception) as raised:
                pairing_refusal('g1', *args, **kwargs)
            assert type(raised.value).__name__ == case['exception'], case
            continue
        refusal = pairing_refusal('g1', *args, **kwargs)
        if case['verdict'] == 'accepted':
            assert refusal is None, case
        else:
            assert refusal == case['message'], case


def test_g1_default_controller_config_matches_old():
    defaults = old_cases()['default_controller_config']
    assert len(defaults) == 7
    for safety, controller in defaults.items():
        assert default_controller_config('g1', safety) == controller, safety


def test_g1_arm_sdk_configs_refused():
    '''golem_safety has no arm_sdk configs: they relay no upper-body channel, pair with none'''
    refusal = pairing_refusal('g1', 'g1', 'arm_sdk', sim=False)
    assert refusal == ('safety config "arm_sdk" must relay the upper-body channel: '
                       'one of split, sim_split')
    refusal = pairing_refusal('g1', 'sim', 'sim_arm_sdk', sim=True)
    assert refusal == ('safety config "sim_arm_sdk" must relay the upper-body channel: '
                       'one of split, sim_split')
    for safety in ('arm_sdk', 'sim_arm_sdk'):
        assert default_controller_config('g1', safety) is None


def test_g1_default_pairs():
    from golem_upper_ik.pairing import (
        G1_CONTROLLER_CONFIGS, G1_SAFETY_CONTROLLER_CONFIGS, G1_UPPER_CHANNEL_SAFETY_CONFIGS)
    assert G1_UPPER_CHANNEL_SAFETY_CONFIGS == ('split', 'sim_split')
    assert G1_CONTROLLER_CONFIGS == ('g1_split', 'sim_split')
    assert G1_SAFETY_CONTROLLER_CONFIGS == {'split': 'g1_split', 'sim_split': 'sim_split'}
    assert default_controller_config('g1', 'split') == 'g1_split'
    assert default_controller_config('g1', 'sim_split') == 'sim_split'
    assert pairing_refusal('g1', 'g1_split', 'split', sim=False) is None
    assert pairing_refusal('g1', 'sim_split', 'sim_split', sim=True) is None


def test_h12_pairs():
    assert pairing_refusal('h12', 'safety_split', 'relax_safety_split') is None
    refusal = pairing_refusal('h12', 'sim_safety_split', 'relax_safety_split')
    assert refusal is not None and 'safety_split' in refusal
    assert pairing_refusal('h12', 'safety_split', 'relax_safety_split', sim=True) is None


def test_h12_known_pairs_run_together_in_sim_and_real():
    for safety, controller in H12_PAIRS.items():
        for sim, locomotion in itertools.product(SIMS, (False, True)):
            for safety_name, controller_name in (
                    (safety, controller),
                    (f'{safety}.yaml', f'{controller}.yaml'),
                    (f'/opt/configs/{safety}.yaml', f'config/{controller}.yaml')):
                assert pairing_refusal('h12', controller_name, safety_name, sim=sim,
                                       locomotion=locomotion) is None


def test_h12_mismatched_pairs_are_refused():
    for safety, paired in H12_PAIRS.items():
        for controller in H12_CONTROLLERS:
            if controller == paired:
                continue
            for sim in SIMS:
                refusal = pairing_refusal('h12', controller, safety, sim=sim)
                assert refusal is not None, (safety, controller, sim)
                assert f'"{safety}"' in refusal and f'"{paired}"' in refusal, refusal


def test_h12_names_outside_the_table_are_not_refused():
    # a custom config is not known by its name; debug is joint_state_publisher's config
    for controller, safety in (('custom', 'relax_safety_split'), ('debug', 'relax_safety_split'),
                               ('safety_split', 'custom'), ('custom', 'custom')):
        for sim in SIMS:
            assert pairing_refusal('h12', controller, safety, sim=sim) is None


def test_h12_default_controller_config():
    for safety, controller in H12_PAIRS.items():
        assert default_controller_config('h12', safety) == controller
        assert default_controller_config('h12', f'/opt/{safety}.yaml') == controller
    assert default_controller_config('h12', 'custom') is None
    assert default_controller_config('h12', 'split') is None


def test_unknown_robot_raises():
    with pytest.raises(ValueError, match='h1'):
        pairing_refusal('h1', 'safety_split', 'relax_safety_split')
    with pytest.raises(ValueError, match='h1'):
        default_controller_config('h1', 'relax_safety_split')


def safety_config(robot, name):
    '''golem_safety's config name of robot, as its YAML'''
    from ament_index_python.packages import get_package_share_directory
    import yaml
    share = Path(get_package_share_directory('golem_safety'))
    return yaml.safe_load((share / 'config' / robot / f'{name}.yaml').read_text())


def paired_configs(robot):
    from golem_upper_ik.pairing import G1_SAFETY_CONTROLLER_CONFIGS
    if robot == 'g1':
        pytest.importorskip('g1_description')
        return G1_SAFETY_CONTROLLER_CONFIGS
    return H12_PAIRS


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_paired_configs_agree_on_the_topics(robot):
    from golem_upper_ik.config import load_controller_config
    for safety, controller in paired_configs(robot).items():
        topics = safety_config(robot, safety)['topics']
        config = load_controller_config(robot, controller)
        # the follow gate reads the topic the safety config writes, by content, not by name
        assert config['topics']['safety_out'] == topics['low_cmd_out'], (safety, controller)
        # and the controller writes an input the safety config reads
        inputs = [topic for key, topic in topics.items() if key.startswith('low_cmd_') and
                  key != 'low_cmd_out']
        assert config['topics']['low_cmd'] in inputs, (safety, controller)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_paired_safety_configs_do_not_clip_the_gains(robot):
    from golem_upper_ik.config import load_controller_config
    for safety, controller in paired_configs(robot).items():
        clip = safety_config(robot, safety)['limits']['clip']
        gains = load_controller_config(robot, controller)['gains']
        assert np.all(gains['kp'] <= clip['kp_max']), (safety, controller)
        assert np.all(gains['kd'] <= clip['kd_max']), (safety, controller)
