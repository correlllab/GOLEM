"""Test that marl_deploy_node refuses, at start, an export it would misread.

    python -m pytest test/test_node_config.py

The node is built and never spun: it raises before it creates a publisher or a subscription.
"""

import os
import shutil

import pytest
import rclpy
import yaml

from lmg_deploy.marl_deploy_node import MarlDeployNode
from shipped_policy import POLICY_DIR, URDF


def _policy_dir(tmp_path, **export_overrides):
    """Return a copy of the shipped policy's yaml files in tmp_path/h12, export.yaml's keys overridden."""
    out = tmp_path / "h12"
    out.mkdir()
    shutil.copy(os.path.join(POLICY_DIR, "h12.yaml"), out / "h12.yaml")
    with open(os.path.join(POLICY_DIR, "export.yaml")) as f:
        export = yaml.safe_load(f)
    export.update(export_overrides)
    with open(out / "export.yaml", "w") as f:
        yaml.safe_dump(export, f)
    return out


def _start(policy_dir):
    """Build the node on policy_dir's h12.yaml; return the ValueError it raises (the test fails if it starts)."""
    rclpy.init(args=["--ros-args", "-p", f"config:={policy_dir / 'h12.yaml'}", "-p", f"urdf_path:={URDF}"])
    try:
        with pytest.raises(ValueError) as error:
            MarlDeployNode()
        return error.value
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize("imu", [
    {"source": "livox", "frame": "livox_imu_link", "gyro_lowpass_hz": 8.0},   # the /livox/imu attitude path
    {"frame": "imu_link", "gyro_lowpass_hz": 8.0},                            # no source: an export before 2026-11-20
])
def test_an_export_off_the_torso_imu_is_refused(tmp_path, imu):
    """SAFETY: an export trained on another IMU than /lowstate's torso IMU is refused, naming the imu block."""
    error = _start(_policy_dir(tmp_path, imu=imu))
    assert "imu.source lowstate" in str(error)
