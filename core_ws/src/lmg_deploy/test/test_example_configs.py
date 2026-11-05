"""Test that the annotated example configs carry exactly the keys of the configs they document.

    python -m pytest test/test_example_configs.py
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")
# export.yaml keys marl_deploy_node reads that a deploy config may also carry.
EXPORT_KEYS_READ = {"gait_phase_cos", "imu", "imu.frame", "imu.gyro_lowpass_hz"}


def _load(*path):
    """Return the YAML file at PKG/path."""
    with open(os.path.join(PKG, *path)) as f:
        return yaml.safe_load(f)


def _keys(cfg, prefix=""):
    """Return cfg's dotted key paths; target_limits' joint names are values, not keys."""
    out = set()
    for key, value in cfg.items():
        out.add(prefix + key)
        if isinstance(value, dict) and key != "target_limits":
            out |= _keys(value, prefix + key + ".")
    return out


def test_example_has_every_deploy_config_key():
    """example.yaml has the template's and every shipped deploy config's keys, plus the export keys the node reads."""
    example = _keys(_load("config", "example.yaml"))
    for path in (("config", "policy_template.yaml"), ("policies", "mid360v2", "mid360v2.yaml"),
                 ("test", "data", "explicit500.yaml")):
        assert _keys(_load(*path)) <= example, path
    assert example == _keys(_load("config", "policy_template.yaml")) | EXPORT_KEYS_READ


def test_example_export_has_the_exports_keys():
    """example.export.yaml has exactly the keys of the shipped policy's export.yaml."""
    assert _keys(_load("config", "example.export.yaml")) == _keys(_load("policies", "mid360v2", "export.yaml"))
