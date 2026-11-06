"""Test that config/h12/example.*.yaml documents every key the configs beside it set."""
from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'h12'
# example kind -> the configs it describes.
DESCRIBED = {
    'mjpc': ('mjpc_real.yaml', 'mjpc_sim.yaml'),
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
    return yaml.safe_load(path.read_text())


def test_every_config_is_described():
    """Each example is one of the kinds checked here, and each config has an example."""
    examples = sorted(path.name for path in CONFIG.glob('example.*'))
    assert examples == sorted(f'example.{kind}.yaml' for kind in DESCRIBED)
    configs = sorted(path.name for path in CONFIG.glob('*.yaml')
                     if not path.name.startswith('example.'))
    assert configs == sorted(name for names in DESCRIBED.values() for name in names)


@pytest.mark.parametrize('kind', sorted(DESCRIBED))
def test_example_has_every_key_of_its_configs(kind):
    """The example has every key its configs set, under the same node names."""
    example = key_paths(load(CONFIG / f'example.{kind}.yaml'))
    for name in DESCRIBED[kind]:
        missing = key_paths(load(CONFIG / name)) - example
        assert missing == set(), f'{name} keys missing from example.{kind}.yaml'
