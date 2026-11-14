"""Run ROS commands or the simulation controller in the clean native overlay."""

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
        raise RuntimeError('Controller bringup is already running on this domain. '
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
Start the simulator once, then the controller:
scripts/run_simulation.sh --task PegInHole --lowerbody almi_manip_stand
scripts/run_controller.sh start_position_verified:=true
Task defaults: ALMI manipulation standing, Frame control mode, compressed RGB,
and the simulator gripper calibration. lowerbody:=fame_finetune is optional.
Engage the standing policy after initialization:
scripts/run_ros.sh service call /lowerbody/confirm_engage std_srvs/srv/Trigger '{}'
Choose Frame or Counter while idle:
scripts/run_ros.sh param set /frame_task_server control_mode counter
scripts/run_ros.sh param set /frame_task_server control_mode frame
For repeated grasps, return to approach_start, then select the saved approach:
scripts/run_ros.sh param set /h12_skills approach_mode reuse
Restore the source peg before repeating an extraction.
The existing run_experiment.sh remains available for explicit backend settings.
Generic CLI: scripts/run_ros.sh topic list
Generic parameter CLI: scripts/run_ros.sh param set NODE PARAM VALUE
No launch command installs/builds. Nonzero simulation domain required.
Actuator launch remains gated by start_position_verified:=true.''',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    launch = result.add_mutually_exclusive_group()
    launch.add_argument('--experiment', action='store_true', help=argparse.SUPPRESS)
    launch.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('arguments', nargs=argparse.REMAINDER,
                         help='ROS CLI arguments, or controller key:=value launch arguments')
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
    if args.experiment or args.controller:
        allowed = {'start_position_verified', 'lowerbody', 'marker_profile_path',
                    'publish_annotated', 'image_transport', 'upper_controller', 'comparison_order',
                    'control_mode', 'approach_mode'}
        if any(':=' not in arg or arg.split(':=', 1)[0] not in allowed for arg in arguments):
            raise RuntimeError('experiment arguments: start_position_verified:=true|false, '
                               'lowerbody:=fame_finetune|almi_manip_stand, marker_profile_path:=PATH, '
                                'publish_annotated:=true|false, image_transport:=raw|compressed, '
                                 'control_mode:=frame|counter, approach_mode:=plan|reuse, '
                                 'upper_controller:=frame|counter, comparison_order:=frame,counter')
        if args.controller:
            provided = {arg.split(':=', 1)[0]: arg.split(':=', 1)[1] for arg in arguments}
            if provided.get('upper_controller', 'counter') != 'counter':
                raise RuntimeError('run_controller supports both modes in one backend; '
                                   'select control_mode:=frame or control_mode:=counter')
            defaults = {
                'upper_controller': 'counter', 'control_mode': 'frame',
                'lowerbody': 'almi_manip_stand', 'image_transport': 'compressed',
                'marker_profile_path': str(ROOT / 'core_ws/src/h12_skills/h12_skills/'
                                          'marker_profile_sim_extraction.json'),
            }
            arguments += [f'{key}:={value}' for key, value in defaults.items() if key not in provided]
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
        for key, choices in (('control_mode', ('frame', 'counter')), ('approach_mode', ('plan', 'reuse'))):
            if any(arg.startswith(f'{key}:=') and arg.split(':=', 1)[1] not in choices
                   for arg in arguments):
                raise RuntimeError(f'{key} must be {" or ".join(choices)}')
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
        if args.controller:
            mode = next(arg.split(':=', 1)[1] for arg in arguments if arg.startswith('control_mode:='))
            print(f'Task control mode: {mode}; standing engagement is a separate command', flush=True)
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
