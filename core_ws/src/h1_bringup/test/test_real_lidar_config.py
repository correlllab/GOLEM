"""Guards the real-robot MID360 IMU/point frame agreement.

livox_ros_driver2 rotates only the points by MID360_config.json's extrinsic;
livox_imu_upright must rotate the IMU identically, and mid360.yaml (shared
with the sims) must then treat the two as one frame. If any of the three drifts, FAST-LIO builds an
upside-down or diverging map on the real robot while the sim stays fine.
"""

import os
import types

import yaml

from h1_bringup.livox_imu_upright import load_rotation, rotate

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PKG, 'config')


def _vec(x, y, z):
    return types.SimpleNamespace(x=x, y=y, z=z)


def test_raw_inverted_gravity_reads_up_after_rotation():
    r = load_rotation(os.path.join(CONFIG, 'MID360_config.json'))
    # At rest the inverted MID360 IMU reads about (-0.19, -0.01, -0.98) g.
    acc = _vec(-0.19, -0.01, -0.98)
    rotate(r, acc)
    assert acc.z > 0.9
    assert abs(acc.x + 0.19) < 1e-6


def test_fast_lio_config_expects_upright_imu():
    with open(os.path.join(CONFIG, 'mid360.yaml')) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    assert params['common']['imu_topic'] == '/livox/imu'
    assert params['mapping']['extrinsic_R'] == [1., 0., 0., 0., 1., 0., 0., 0., 1.]


def test_fast_lio_keeps_every_lidar_line():
    # line < scan_line is a hard filter: the real MID360 emits lines 0-3 and
    # both sims 0-5, so anything below 6 silently drops sim points.
    with open(os.path.join(CONFIG, 'mid360.yaml')) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    assert params['preprocess']['scan_line'] >= 6


def test_one_slam_config_set():
    names = sorted(os.listdir(CONFIG))
    assert [n for n in names if n.startswith('mid360')] == ['mid360.yaml']
    assert [n for n in names if n.startswith('slam_toolbox')] == ['slam_toolbox_h1.yaml']


def test_real_drivers_route_imu_through_upright_node():
    path = os.path.join(PKG, 'launch', 'h1_real_drivers.launch.py')
    with open(path) as f:
        src = f.read()
    assert "('/livox/imu', '/livox/imu_raw')" in src
    assert "executable='livox_imu_upright'" in src
