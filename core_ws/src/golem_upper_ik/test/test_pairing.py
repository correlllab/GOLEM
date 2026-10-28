'''
Which safety and controller configs may run together.

The G1's verdicts are compared with g1_control's config_pairing, through
test/fixtures/old_controller/g1_pairing.json (written by tools/snapshot_old_controller.py);
this test does not import g1_control.
'''
import itertools
import json
from pathlib import Path

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
    assert len(cases) == 600
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
            # the same rule refuses it: the message is the old one, naming golem_safety
            assert refusal == case['message'].replace('g1_safety_layer', 'golem_safety'), case


def test_g1_default_controller_config_matches_old():
    for safety, controller in old_cases()['default_controller_config'].items():
        assert default_controller_config('g1', safety) == controller, safety


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
    assert default_controller_config('h12', 'arm_sdk') is None


def test_unknown_robot_raises():
    with pytest.raises(ValueError, match='h1'):
        pairing_refusal('h1', 'safety_split', 'relax_safety_split')
    with pytest.raises(ValueError, match='h1'):
        default_controller_config('h1', 'relax_safety_split')
