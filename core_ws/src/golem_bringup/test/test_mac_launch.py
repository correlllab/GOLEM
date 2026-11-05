"""
The Mac's H1-2 sim bringup includes the x86 sim's blocks and golem_slam's navigation.

h12_sim_mac.launch.py includes h12_state and h12_control with the arguments
h12_sim.launch.py passes them, h12_lowerbody when GOLEM_LOWERBODY names a policy, and
golem_slam's navigation.launch.py for the H1-2 when GOLEM_SLAM or GOLEM_NAV2 is on. It
reads the environment when its description is generated, so each case sets it first.
Needs the workspace built and sourced (golem_bringup and golem_slam installed).
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
import pytest

from test_bringup_packages import _context, _load, _passed_to, _text
from test_g1_launch import _location

TOGGLES = ('GOLEM_LOWERBODY', 'GOLEM_SLAM', 'GOLEM_NAV2')
NAV_LAUNCH = os.path.join(get_package_share_directory('golem_slam'), 'launch',
                          'navigation.launch.py')


@pytest.fixture
def env(monkeypatch):
    """Clear the Mac's toggles; return a function that sets some of them."""
    for name in TOGGLES:
        monkeypatch.delenv(name, raising=False)

    def set_env(**values):
        for name, value in values.items():
            monkeypatch.setenv(name, value)
    return set_env


def _includes():
    """Return {included file name: {argument: value}} of what h12_sim_mac includes."""
    description = _load('h12_sim_mac.launch.py').generate_launch_description()
    context = _context(description)
    assert all(isinstance(entity, IncludeLaunchDescription)
               for entity in description.entities), 'starts more than its includes'
    includes = {}
    for entity in description.entities:
        path = _location(entity)
        name = 'navigation' if path == NAV_LAUNCH else Path(path).name
        assert name not in includes, f'{name} included twice'
        includes[name] = {_text(context, key): _text(context, value)
                          for key, value in entity.launch_arguments}
    return includes


@pytest.mark.parametrize('lowerbody', [None, ''])
def test_no_toggles_start_state_and_control(env, lowerbody):
    if lowerbody is not None:
        env(GOLEM_LOWERBODY=lowerbody)
    includes = _includes()
    assert set(includes) == {'h12_state.launch.py', 'h12_control.launch.py'}
    for name, arguments in includes.items():
        assert arguments == _passed_to('h12_sim.launch.py', name), name


@pytest.mark.parametrize('policy', ['almi', 'fame', 'walk', ' ALMI '])
def test_lowerbody_includes_the_leg_block(env, policy):
    env(GOLEM_LOWERBODY=policy)
    includes = _includes()
    assert set(includes) == {'h12_state.launch.py', 'h12_control.launch.py',
                             'h12_lowerbody.launch.py'}
    arguments = includes['h12_lowerbody.launch.py']
    assert arguments['lowerbody'] == policy.strip().lower()
    # the x86 sim's arguments, its lowerbody launch argument aside
    assert arguments == dict(_passed_to('h12_sim.launch.py', 'h12_lowerbody.launch.py'),
                             lowerbody=policy.strip().lower())
    assert arguments['start_position_verified'] == 'true'
    assert arguments['use_sim_time'] == 'true'
    assert Path(arguments['lowerbody_params']).name == 'lowerbody_sim.yaml'


@pytest.mark.parametrize('policy', ['bogus', 'mjpc'])
def test_unknown_lowerbody_is_refused(env, policy):
    env(GOLEM_LOWERBODY=policy)
    with pytest.raises(RuntimeError, match=f"GOLEM_LOWERBODY={policy!r}.*fame, walk, almi"):
        _includes()


@pytest.mark.parametrize('toggles', [
    dict(GOLEM_NAV2='1'),
    dict(GOLEM_SLAM='1'),
    dict(GOLEM_SLAM='true', GOLEM_NAV2='on'),
    dict(GOLEM_NAV2='ON'),
])
def test_slam_or_nav2_includes_navigation(env, toggles):
    env(**toggles)
    includes = _includes()
    assert set(includes) == {'h12_state.launch.py', 'h12_control.launch.py', 'navigation'}
    assert includes['navigation'] == {'robot': 'h12', 'use_sim_time': 'true'}
    assert includes['navigation'] == _passed_to('h12_sim.launch.py', 'navigation.launch.py')


@pytest.mark.parametrize('value', ['0', 'off', 'false'])
def test_slam_and_nav2_off(env, value):
    env(GOLEM_SLAM=value, GOLEM_NAV2=value)
    assert 'navigation' not in _includes()
