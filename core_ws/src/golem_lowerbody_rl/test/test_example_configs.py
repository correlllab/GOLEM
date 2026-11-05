"""Test that config/example.*.yaml hold exactly the loaders' keys and are not installed."""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from golem_lowerbody_rl.profile import load_profile, ROBOT_KEYS
from golem_lowerbody_rl.run_config import load_run_config, SECTION_KEYS, TOP_LEVEL_KEYS
import yaml

SOURCE_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
# the robot whose values the examples carry
EXAMPLE_ROBOT = 'h12'


def _load(path):
    """Return a YAML file's content."""
    with open(path, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _run_keys(raw):
    """Return the run-config key paths of raw, sections as 'section.key'."""
    keys = set()
    for key, value in raw.items():
        if key in SECTION_KEYS:
            keys |= {f'{key}.{inner}' for inner in value}
        else:
            keys.add(key)
    return keys


RUN_KEYS = ({key for key in TOP_LEVEL_KEYS if key not in SECTION_KEYS}
            | {f'{section}.{key}' for section, keys in SECTION_KEYS.items() for key in keys})


def test_examples_hold_every_key_and_load():
    """Each example holds exactly its schema's keys, as each real config does, and loads."""
    robot_example = os.path.join(SOURCE_CONFIG, 'example.robot.yaml')
    assert set(_load(robot_example)) == set(ROBOT_KEYS)
    assert load_profile(EXAMPLE_ROBOT, robot_example) == load_profile(EXAMPLE_ROBOT)
    for path in glob.glob(os.path.join(SOURCE_CONFIG, '*', 'robot.yaml')):
        assert set(_load(path)) == set(ROBOT_KEYS), path

    run_example = os.path.join(SOURCE_CONFIG, 'example.run.yaml')
    assert _run_keys(_load(run_example)) == RUN_KEYS
    load_run_config(EXAMPLE_ROBOT, run_example)
    runs = glob.glob(os.path.join(SOURCE_CONFIG, '*', 'sim.yaml')) + glob.glob(
        os.path.join(SOURCE_CONFIG, '*', 'real.yaml'))
    assert len(runs) == 4
    for path in runs:
        assert _run_keys(_load(path)) == RUN_KEYS, path


def test_examples_are_not_installed():
    """The source has the three example files at config/, and none is installed."""
    examples = glob.glob(os.path.join(SOURCE_CONFIG, '**', 'example.*'), recursive=True)
    assert sorted(os.path.relpath(path, SOURCE_CONFIG) for path in examples) == [
        'example.policy.yaml', 'example.robot.yaml', 'example.run.yaml']
    share = os.path.join(get_package_share_directory('golem_lowerbody_rl'), 'config')
    assert os.path.isfile(os.path.join(share, EXAMPLE_ROBOT, 'robot.yaml'))
    assert glob.glob(os.path.join(share, '**', 'example.*'), recursive=True) == []
