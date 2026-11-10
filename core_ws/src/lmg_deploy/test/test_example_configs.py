"""Test that the annotated example configs carry exactly the keys of the configs they document.

    python -m pytest test/test_example_configs.py
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..")
# The policies the package ships, one per robot (policies/<robot>/<robot>.yaml; policies/README.md).
SHIPPED_POLICIES = ("h12",)
# export.yaml keys marl_deploy_node reads that a deploy config may also carry.
EXPORT_KEYS_READ = {"gait_phase_cos", "imu", "imu.frame", "imu.gyro_lowpass_hz", "leg_action_scales",
                    "arm_residual_scales", "max_ik_step", "command_filter_tau", "legs_obs_terms", "arms_obs_terms",
                    "arm_goals"}


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
    for path in (("config", "policy_template.yaml"), ("test", "data", "explicit500.yaml"),
                 *(("policies", name, f"{name}.yaml") for name in SHIPPED_POLICIES)):
        assert _keys(_load(*path)) <= example, path
    assert example == _keys(_load("config", "policy_template.yaml")) | EXPORT_KEYS_READ


def test_example_export_has_the_exports_keys():
    """example.export.yaml has exactly the keys of the shipped policies' export.yaml files together."""
    shipped = set()
    for name in SHIPPED_POLICIES:
        shipped |= _keys(_load("policies", name, "export.yaml"))
    assert _keys(_load("config", "example.export.yaml")) == shipped
