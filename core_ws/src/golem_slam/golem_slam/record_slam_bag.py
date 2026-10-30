"""
Record a rosbag2 of a SLAM/navigation run: every topic except the heavy ones.

    ros2 run golem_slam record_slam_bag --name <name> [--dir <dir>]

The bag goes to ``<dir>/<name>`` (default ``<dir>`` is ``core_ws/bags``,
found from this package's install prefix). It keeps what is needed to replay
or rerun SLAM offline (``/livox/lidar``, ``/livox/imu``, ``/lowstate``, TF,
FAST-LIO odometry and registered clouds, the slam_toolbox map, Nav2
costmaps and plans, compressed camera images) and leaves out:

- ``/Laser_map``: FAST-LIO appends every scan to one cloud that is never
  cleared and republishes the whole cloud each tick, so the bag grows
  quadratically with run length. FAST-LIO's ``map_save`` service writes the
  same cloud to a PCD once instead.
- ``/realsense/accumulated_point_cloud``: cl_realsense's ``pc_acc``, the
  same problem.
- RealSense ``.../points`` clouds and uncompressed or theora images; their
  ``compressed``/``compressedDepth`` transports are kept.
- Unitree firmware topics rosbag2 can't record: ``/SymState`` and
  ``/config_change_status`` (their unitree_go types aren't in unitree_ros2,
  and an unloadable type aborts the whole recording) and ``/lf/lowstate``
  (published with both a unitree_go and a unitree_hg type; a low-rate copy of
  ``/lowstate``).
"""

import argparse
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix

# ros2 bag -x drops any topic whose name contains a match, so each pattern is
# anchored to stop it from also catching the topics' compressed transports.
EXCLUDE = [
    r'^/Laser_map$',
    r'^/realsense/accumulated_point_cloud$',
    r'/points$',
    r'/image_raw$',
    r'/image_rect_raw$',
    r'/theora$',
    r'^/SymState$',
    r'^/config_change_status$',
    r'^/lf/lowstate$',
]


def default_bag_dir():
    """Return ``core_ws/bags``, from the ``core_ws/install/golem_slam`` prefix."""
    return Path(get_package_prefix('golem_slam')).parents[1] / 'bags'


def main():
    """Parse --name/--dir and exec ``ros2 bag record`` in place of this process."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--name', required=True,
                        help='bag name; the bag is written to <dir>/<name>')
    parser.add_argument('--dir', type=Path, default=None,
                        help='parent directory (default: core_ws/bags)')
    args = parser.parse_args()

    bag_dir = args.dir if args.dir is not None else default_bag_dir()
    out = bag_dir / args.name
    if out.exists():
        parser.error(f'{out} already exists')
    bag_dir.mkdir(parents=True, exist_ok=True)

    cmd = ['ros2', 'bag', 'record', '-a', '-o', str(out), '-x', '|'.join(EXCLUDE)]
    print(' '.join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
