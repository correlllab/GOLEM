"""Execute the native RViz C++ binary, without starting any robot nodes."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'core_ws/src/h12_bringup/rviz/pegboard.rviz',
                        help='RViz recipe (default: main.rviz-derived pegboard.rviz in the source checkout)')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    domain = os.environ.get('ROS_DOMAIN_ID', '1')
    if not domain.isascii() or not domain.isdecimal() or not 1 <= int(domain) <= 232:
        parser.error('ROS_DOMAIN_ID must be 1..232, never domain 0')
    sys.path.insert(0, str(ROOT / 'core_ws/scripts'))
    import build_ros_workspace as workspace
    binary = Path('/opt/ros/humble/lib/rviz2/rviz2')
    overlay = ROOT / 'core_ws/install/local_setup.bash'
    for path in (binary, overlay, args.config):
        if not path.is_file():
            parser.error(f'missing prerequisite: {path}')
    if args.check:
        print(f'RViz ready: {binary}; config={args.config}; domain={domain}; fixed_frame=pelvis; use_sim_time=true')
        return
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('DISPLAY or WAYLAND_DISPLAY is required for RViz')
    env = workspace.environment()
    # RViz is C++, not a Python console script. Do not inject controller/Conda Qt.
    env['QT_QPA_PLATFORM'] = 'xcb' if env.get('DISPLAY') else 'wayland'
    os.execve('/bin/bash', ['/bin/bash', '--noprofile', '--norc', '-c',
              'set -e; source "$1"; shift; exec "$@"', 'golem-rviz', str(overlay),
              str(binary), '-d', str(args.config.resolve()), '--ros-args', '-p', 'use_sim_time:=true'], env)


if __name__ == '__main__':
    main()
