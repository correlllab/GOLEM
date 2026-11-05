"""Root dependency setup/check dispatch. No command launches nodes or DDS."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog='''From repository root (also works by absolute path from zsh/bash):
  scripts/setup_environment.sh ros --local-ros-debs
  scripts/setup_environment.sh simulator
  core_ws/scripts/build_ros_workspace.sh build --profile manipulation --compile-jobs 2
  scripts/setup_environment.sh simulator --build-interfaces
  scripts/setup_environment.sh ros --check --built --profile manipulation
  scripts/setup_environment.sh simulator --check --bridges
  core_ws/scripts/build_ros_workspace.sh build --profile experiment --compile-jobs 2
  scripts/setup_environment.sh controller
  scripts/setup_environment.sh controller --check
  scripts/run_experiment.sh --help
  scripts/run_simulation.sh --task PegInHole

Setup recreates missing .venv with /usr/bin/python3 (3.10), system-site-packages,
no user site, uv sync --locked --inexact, and the SDK from this checkout --no-deps.
ROS uses NumPy 1 and Torch 2.6 CPU; simulator and isolated controller use NumPy 2.
Controller setup uses only baseline Pin/Pink/OMPL/QP libraries, not Crocoddyl or ML.
Its checks import actual arm entrypoints and run offline math, never constructors.
Build experiment before controller setup/check; it needs native generated interfaces.
No global pip, sudo, conda, or upgrades.
--local-ros-debs only downloads/extracts the ROS helper's two allowlisted debs.
--build-interfaces explicitly generates pinned Livox messages after core messages
are installed, then checks actual bridge imports and native CDR serialization.
It does not resync dependencies. Launch never installs or compiles anything.
Locks: environments/{ros,simulator,controller}/uv.lock. System prerequisites:
environments/ros/apt-packages.txt; ROS Humble, GL libraries, and checked-out assets.
No command launches nodes, controllers, simulator loops, or DDS participants.''')
    parser.add_argument('environment', choices=('ros', 'simulator', 'controller'))
    parser.add_argument('--local-ros-debs', action='store_true')
    parser.add_argument('--build-interfaces', action='store_true')
    parser.add_argument('--compile-jobs', type=int, choices=range(1, 65), default=2, metavar='1..64')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--built', action='store_true')
    parser.add_argument('--profile', choices=('core', 'interfaces', 'manipulation', 'experiment'), default='manipulation')
    parser.add_argument('--bridges', action='store_true')
    args = parser.parse_args()
    if args.local_ros_debs and (args.environment != 'ros' or args.check):
        parser.error('--local-ros-debs requires ros setup')
    if args.build_interfaces and (args.environment != 'simulator' or args.check):
        parser.error('--build-interfaces requires simulator, without --check')
    if args.built and (args.environment != 'ros' or not args.check):
        parser.error('--built requires ros --check')
    if args.bridges and (args.environment != 'simulator' or not args.check):
        parser.error('--bridges requires simulator --check')
    if args.environment == 'controller':
        import setup_controller
        setup_controller.main(args.check)
    elif args.environment == 'ros':
        command = [ROOT / 'core_ws/scripts/build_ros_workspace.py', 'check' if args.check else 'setup']
        if args.local_ros_debs:
            command.append('--local-ros-debs')
        if args.check:
            command += ['--profile', args.profile]
            if args.built:
                command.append('--built')
        os.execv('/usr/bin/python3', ['/usr/bin/python3', *map(str, command)])
    else:
        import setup_simulator
        if args.build_interfaces:
            setup_simulator.require_venv()
            setup_simulator.build_livox(args.compile_jobs)
            setup_simulator.check(True)
        else:
            sys.argv = ['setup_simulator.py', 'check' if args.check else 'setup']
            if args.bridges:
                sys.argv.append('--bridges')
            setup_simulator.main()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(1)
