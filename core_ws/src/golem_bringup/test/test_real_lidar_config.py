"""
Guards the real-robot MID360 IMU/point frame agreement.

The H1-2's MID-360 hangs upside down and livox_ros_driver2 publishes both the
points and the IMU in the sensor's raw frame, so the two agree only while the
driver config applies no extrinsic rotation and nothing rotates the IMU on its
way to /livox/imu. If either changes, FAST-LIO builds an upside-down or
diverging map on the real robot while the sim stays fine.
"""

import json
import os

import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PKG, 'config', 'h12')
SLAM_CONFIG = os.path.join(os.path.dirname(PKG), 'golem_slam', 'config')
H12_SLAM_CONFIG = os.path.join(SLAM_CONFIG, 'h12')


def test_driver_config_applies_no_rotation():
    with open(os.path.join(H12_SLAM_CONFIG, 'MID360_config.json')) as f:
        extrinsic = json.load(f)['lidar_configs'][0]['extrinsic_parameter']
    assert extrinsic == {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'x': 0, 'y': 0, 'z': 0}


def test_fast_lio_config_reads_the_drivers_imu():
    with open(os.path.join(H12_SLAM_CONFIG, 'fastlio.yaml')) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    assert params['common']['imu_topic'] == '/livox/imu'


def test_fast_lio_keeps_every_lidar_line():
    # line < scan_line is a hard filter: the real MID360 emits lines 0-3 and
    # both sims 0-5, so anything below 6 silently drops sim points.
    with open(os.path.join(H12_SLAM_CONFIG, 'fastlio.yaml')) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    assert params['preprocess']['scan_line'] >= 6


def test_one_slam_config_set():
    assert sorted(os.listdir(H12_SLAM_CONFIG)) == [
        'MID360_config.json', 'fastlio.yaml', 'nav2.yaml', 'octomap.yaml', 'pipeline.yaml']
    names = sorted(os.listdir(SLAM_CONFIG))
    assert [n for n in names if n.startswith('slam_toolbox')] == ['slam_toolbox.yaml']
    # Bringup loads lidar/SLAM/nav configs from golem_slam; a copy here would silently drift.
    leftovers = [n for n in os.listdir(CONFIG)
                 if n.lower().startswith(('mid360', 'slam_toolbox', 'nav2'))]
    assert leftovers == []


def test_real_drivers_publish_the_raw_imu_directly():
    path = os.path.join(PKG, 'launch', 'h12', 'h12_real_drivers.launch.py')
    with open(path) as f:
        src = f.read()
    assert '/livox/imu' not in src
    assert 'remappings' not in src
    assert "'frame_id': 'livox_frame'" in src
