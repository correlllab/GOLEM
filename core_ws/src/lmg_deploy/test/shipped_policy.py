"""Load the shipped H1-2 policy's deploy config the way marl_deploy_node does, for the tests.

The config is policies/h12/h12.yaml with export.yaml's keys over it and arm_seed_table made absolute; the networks
are not loaded (the tests stand in their own actors).
"""

import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(HERE, "..", "policies", "h12")
URDF = os.environ.get(
    "H12_URDF", os.path.join(HERE, "..", "..", "..", "..", "CL_Assets", "ros_assets", "h1_2_magpie.urdf"))
# A torso-IMU calibration that removes nothing (config/imu_calibration.yaml's h12_sim): the calibrated policy needs one.
ZERO_CALIBRATION = {"roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0, "rotate_gyro": False}


def shipped_cfg() -> dict:
    """Return the shipped policy's deploy config: h12.yaml, export.yaml over it, the seed table's path absolute."""
    with open(os.path.join(POLICY_DIR, "h12.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(POLICY_DIR, "export.yaml")) as f:
        cfg.update(yaml.safe_load(f))
    cfg["arm_seed_table"] = os.path.join(POLICY_DIR, cfg["arm_seed_table"])
    return cfg
