"""Test that config/example.*.yaml has exactly the keys of the configs it describes."""
import glob
import os

from golem_slam.navigation_launch import load_pipeline, PIPELINE_KEYS
import pytest
import yaml

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
# example kind -> the configs it describes, relative to config/.
DESCRIBED = {
    'fastlio': ['*/fastlio*.yaml'],
    'nav2': ['*/nav2.yaml'],
    'octomap': ['*/octomap.yaml'],
    'pipeline': ['*/pipeline.yaml'],
    'slam_toolbox': ['slam_toolbox.yaml'],
}


def key_paths(tree, prefix=()):
    """Return the set of key paths to every leaf of a loaded YAML mapping."""
    paths = set()
    for key, value in tree.items():
        if isinstance(value, dict) and value:
            paths |= key_paths(value, prefix + (key,))
        else:
            paths.add(prefix + (key,))
    return paths


def load(path):
    """Return a YAML file's content."""
    with open(path) as source:
        return yaml.safe_load(source)


def described(kind):
    """Return the paths of the configs an example kind describes."""
    return sorted(path for pattern in DESCRIBED[kind]
                  for path in glob.glob(os.path.join(CONFIG, pattern)))


def test_every_example_is_checked():
    """Each example file is one of the kinds checked here."""
    examples = sorted(os.path.basename(path)
                      for path in glob.glob(os.path.join(CONFIG, 'example.*')))
    assert examples == sorted(f'example.{kind}.yaml' for kind in DESCRIBED)


@pytest.mark.parametrize('kind', sorted(DESCRIBED))
def test_example_keys_are_the_configs_keys(kind):
    """The example has every key its configs have, and none they all lack."""
    configs = described(kind)
    assert configs, kind
    union = set().union(*(key_paths(load(path)) for path in configs))
    example = key_paths(load(os.path.join(CONFIG, f'example.{kind}.yaml')))
    assert union - example == set(), f'undocumented keys of {kind}'
    assert example - union == set(), f'example.{kind}.yaml keys no config has'


def test_pipeline_example_loads_with_exactly_the_required_keys():
    """The pipeline example passes load_pipeline and holds only its required keys."""
    path = os.path.join(CONFIG, 'example.pipeline.yaml')
    assert key_paths(load_pipeline(path)) == set(PIPELINE_KEYS)
    for config in described('pipeline'):
        assert key_paths(load(config)) == set(PIPELINE_KEYS), config
