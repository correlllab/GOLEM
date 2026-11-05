'''
Which safety and controller configs may run together, and what the pairs agree on.

The pairs' topics and gain limits are read from golem_safety's installed configs.
'''
import inspect
from pathlib import Path

import numpy as np
import pytest

from golem_upper_ik.pairing import default_controller_config, pairing_refusal
from golem_upper_ik.profile import known_robots

SIMS = (None, False, True)


def pairs_of(robot):
    '''The safety config -> controller config pairs of robot, as its installed pairing.yaml'''
    from ament_index_python.packages import get_package_share_directory
    import yaml
    share = Path(get_package_share_directory('golem_upper_ik'))
    return yaml.safe_load((share / 'config' / robot / 'pairing.yaml').read_text())['pairs']


def test_pairing_yaml_is_the_pair_table():
    assert pairs_of('g1') == {'split': 'g1_split', 'sim_split': 'sim_split'}
    assert pairs_of('h12') == {
        'relax_safety_split': 'safety_split',
        'default_safety_split': 'safety_split',
        'tight_safety_split': 'safety_split',
        'sim_safety_split': 'sim_safety_split',
        'relax_safety_full': 'safety_full',
        'default_safety_full': 'safety_full',
        'tight_safety_full': 'safety_full',
        'locomotion_safety_full': 'safety_full',
    }


def test_the_pairing_functions_are_robot_independent():
    from golem_upper_ik import pairing
    assert list(inspect.signature(pairing_refusal).parameters) == [
        'robot', 'controller_config', 'safety_config', 'sim']
    assert list(inspect.signature(default_controller_config).parameters) == [
        'robot', 'safety_config']
    assert [name for name in vars(pairing) if name.startswith(('H12_', 'G1_'))] == []


@pytest.mark.parametrize('robot, safety, controller, sim, expected', [
    ('g1', 'split', 'g1_split', False, None),
    ('g1', 'split', 'g1_split', None, None),
    ('g1', 'sim_split', 'sim_split', True, None),
    ('g1', 'full', 'g1_split', False, 'safety config "full" is not one of the G1 pairs'),
    ('g1', 'sim_split', 'sim_split', False,
     'safety config "sim_split" is a sim config; the real robot never runs one'),
    ('g1', 'split', 'sim_split', None,
     'safety config "split" runs with controller config "g1_split", not "sim_split"'),
    ('g1', 'sim_split', 'g1_split', None,
     'safety config "sim_split" runs with controller config "sim_split", not "g1_split"'),
    ('g1', 'split', 'g1_split', True, 'safety config "split" is for the real robot, not the sim'),
    ('h12', 'relax_safety_split', 'safety_split', False, None),
    ('h12', 'sim_safety_split', 'sim_safety_split', True, None),
    ('h12', 'relax_safety_split', 'sim_safety_split', None,
     'safety config "relax_safety_split" runs with controller config "safety_split", not '
     '"sim_safety_split"'),
    ('h12', 'debug_unknown', 'safety_split', None,
     'safety config "debug_unknown" is not one of the H1-2 pairs'),
    ('h12', 'relax_safety_split', 'safety_split', True,
     'safety config "relax_safety_split" is for the real robot, not the sim'),
    ('h12', 'sim_safety_split', 'sim_safety_split', False,
     'safety config "sim_safety_split" is a sim config; the real robot never runs one'),
])
def test_pairing_one_rule(robot, safety, controller, sim, expected):
    refusal = pairing_refusal(robot, controller, safety, sim=sim)
    if expected is None:
        assert refusal is None
    else:
        assert refusal is not None and refusal.startswith(expected), refusal


@pytest.mark.parametrize('robot', known_robots())
def test_pairs_run_together_by_name_or_path(robot):
    for safety, controller in pairs_of(robot).items():
        sim = safety.startswith('sim')
        for known in (None, sim):
            for safety_name, controller_name in (
                    (safety, controller),
                    (f'{safety}.yaml', f'{controller}.yaml'),
                    (f'/opt/configs/{safety}.yaml', f'config/{controller}.yaml')):
                assert pairing_refusal(robot, controller_name, safety_name,
                                       sim=known) is None
        assert default_controller_config(robot, safety) == controller
        assert default_controller_config(robot, f'/opt/{safety}.yaml') == controller


@pytest.mark.parametrize('robot', known_robots())
def test_mismatched_controllers_are_refused(robot):
    pairs = pairs_of(robot)
    for safety, paired in pairs.items():
        for controller in set(pairs.values()) | {'custom', 'debug'}:
            if controller == paired:
                continue
            for sim in SIMS:
                refusal = pairing_refusal(robot, controller, safety, sim=sim)
                assert refusal is not None, (safety, controller, sim)
                assert f'"{safety}"' in refusal and f'"{paired}"' in refusal, refusal


@pytest.mark.parametrize('robot', known_robots())
def test_default_controller_config_of_an_unpaired_safety_config_raises(robot):
    with pytest.raises(ValueError, match='"custom" is not one of'):
        default_controller_config(robot, 'custom')


def test_unknown_robot_raises():
    with pytest.raises(ValueError, match="unknown robot 'h1'"):
        pairing_refusal('h1', 'safety_split', 'relax_safety_split')
    with pytest.raises(ValueError, match="unknown robot 'h1'"):
        default_controller_config('h1', 'relax_safety_split')


def safety_config(robot, name):
    '''golem_safety's config name of robot, as its YAML'''
    from ament_index_python.packages import get_package_share_directory
    import yaml
    share = Path(get_package_share_directory('golem_safety'))
    return yaml.safe_load((share / 'config' / robot / f'{name}.yaml').read_text())


@pytest.mark.parametrize('robot', known_robots())
def test_paired_configs_agree_on_the_topics(robot):
    from golem_upper_ik.config import SAFETY_OUTPUT_TOPIC, load_controller_config
    for safety, controller in pairs_of(robot).items():
        topics = safety_config(robot, safety)['topics']
        config = load_controller_config(robot, controller)
        # the follow gate reads the topic the safety config writes
        assert SAFETY_OUTPUT_TOPIC == topics['low_cmd_out'], (safety, controller)
        # and the controller writes an input the safety config reads
        inputs = [topic for key, topic in topics.items() if key.startswith('low_cmd_') and
                  key != 'low_cmd_out']
        assert config['topics']['low_cmd'] in inputs, (safety, controller)


@pytest.mark.parametrize('robot', known_robots())
def test_paired_safety_configs_do_not_clip_the_gains(robot):
    from golem_upper_ik.config import load_controller_config
    for safety, controller in pairs_of(robot).items():
        clip = safety_config(robot, safety)['limits']['clip']
        gains = load_controller_config(robot, controller)['gains']
        assert np.all(gains['kp'] <= clip['kp_max']), (safety, controller)
        assert np.all(gains['kd'] <= clip['kd_max']), (safety, controller)
