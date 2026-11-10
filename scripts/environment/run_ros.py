"""Run a ROS CLI command in the clean native overlay, or gated experiment bringup."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def acquire_experiment_lock(domain):
    """Keep one experiment launch per checkout/domain, including its child nodes."""
    path = ROOT / '.cache/runtime' / f'experiment-domain-{int(domain)}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(descriptor, True)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError('An experiment is already running on this domain. '
                           'Stop its launch with Ctrl+C before switching between '
                           'observation and actuation modes.') from None
    except BaseException:
        os.close(descriptor)
        raise
    # The descriptor deliberately survives exec and releases on process exit.
    return descriptor


def parser():
    result = argparse.ArgumentParser(description=__doc__, epilog='''Preparation:
scripts/setup_environment.sh ros --local-ros-debs
core_ws/scripts/build_ros_workspace.sh build --profile experiment --compile-jobs 2
scripts/setup_environment.sh controller
scripts/setup_environment.sh controller --check
Observe only: scripts/run_experiment.sh
Actuation only after separate approval and safe simulation start verification:
scripts/run_experiment.sh start_position_verified:=true lowerbody:=fame_finetune
Alternative standing policy: lowerbody:=almi_manip_stand
Upper controller: upper_controller:=frame|counter (default frame).
Same-run comparison requires upper_controller:=counter, including frame trials.
comparison_order:=frame,counter (CSV; duplicates allowed, e.g. frame,counter,counter,frame).
Change order between batches: scripts/run_ros.sh param set /frame_task_server comparison_order "counter,frame"
The lower-body engage_wait_for_confirm interlock remains enabled.
Start the simulator separately, preferably GOLEM_CAMERA_SCALE=1 scripts/run_simulation.sh.
Generic CLI: scripts/run_ros.sh topic list
Generic parameter CLI: scripts/run_ros.sh param set NODE PARAM VALUE
No launch command installs/builds. Nonzero simulation domain required.
Offline checks do not qualify live control, balance, or insertion accuracy.''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--experiment', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('arguments', nargs=argparse.REMAINDER,
                        help='ROS CLI arguments, or experiment key:=value launch arguments')
    return result


def main():
    args = parser().parse_args()
    domain = os.environ.get('ROS_DOMAIN_ID') or '1'
    if not domain.isascii() or not domain.isdecimal() or not 1 <= int(domain) <= 232:
        raise RuntimeError('ROS_DOMAIN_ID must be 1..232; domain 0 is the real robot bus')
    os.environ['ROS_DOMAIN_ID'] = str(int(domain))
    sys.path.insert(0, str(ROOT / 'core_ws/scripts'))
    import build_ros_workspace as workspace
    workspace.require_venv()
    overlay = ROOT / 'core_ws/install/local_setup.bash'
    if not overlay.is_file():
        raise RuntimeError('build the native workspace first')
    env = workspace.environment()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    arguments = args.arguments
    if args.experiment:
        allowed = {'start_position_verified', 'lowerbody', 'marker_profile_path',
                    'publish_annotated', 'image_transport', 'upper_controller', 'comparison_order'}
        if any(':=' not in arg or arg.split(':=', 1)[0] not in allowed for arg in arguments):
            raise RuntimeError('experiment arguments: start_position_verified:=true|false, '
                               'lowerbody:=fame_finetune|almi_manip_stand, marker_profile_path:=PATH, '
                                'publish_annotated:=true|false, image_transport:=raw|compressed, '
                                'upper_controller:=frame|counter, comparison_order:=frame,counter')
        for arg in arguments:
            if arg.startswith('comparison_order:='):
                value = arg.split(':=', 1)[1]
                if not value or any(mode.strip() not in ('frame', 'counter')
                                    for mode in value.split(',')):
                    raise RuntimeError('comparison_order must be CSV containing only frame or counter')
        if any(arg.startswith('upper_controller:=') and arg.split(':=', 1)[1] not in ('frame', 'counter')
               for arg in arguments):
            raise RuntimeError('upper_controller must be frame or counter')
        if any(arg.startswith('image_transport:=') and arg.split(':=', 1)[1] not in ('raw', 'compressed')
               for arg in arguments):
            raise RuntimeError('image_transport must be raw or compressed')
        python = ROOT / 'environments/controller/.venv/bin/python'
        if not python.is_file():
            raise RuntimeError('run scripts/setup_environment.sh controller first')
        env['GOLEM_CONTROLLER_PYTHON'] = str(python)
        catalog = ROOT / 'CL_Assets/mujoco_assets/manipulation_net/manipulation_catalog.json'
        if not catalog.is_file():
            raise RuntimeError(f'missing catalog: {catalog}')
        env['GOLEM_MARKER_CATALOG'] = str(catalog)
        upper = next((arg.split(':=', 1)[1] for arg in reversed(arguments)
                      if arg.startswith('upper_controller:=')), 'frame')
        print(f'Upper controller: {upper} (actuation requires start_position_verified:=true)', flush=True)
        arguments = ['launch', 'h12_bringup', 'h12_native_experiment.launch.py', *arguments]
        acquire_experiment_lock(domain)
    elif not arguments:
        raise RuntimeError('supply a ROS CLI command')
    os.execve('/bin/bash', ['/bin/bash', '--noprofile', '--norc', '-c',
              'set -e; source "$1"; shift; exec "$@"', 'golem-ros', str(overlay),
              str(workspace.PYTHON), '/opt/ros/humble/bin/ros2', *arguments], env)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(1)
