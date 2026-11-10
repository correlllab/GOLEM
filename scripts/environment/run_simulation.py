"""Run the existing MuJoCo simulator, not bringup or robot controllers."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

PEGBOARD_CAMERA_DEFAULTS = {
    'GOLEM_CAMERA_SCALE': '1',
    'GOLEM_CAMERA_DEPTH': '0',
    'GOLEM_CAMERA_POINTCLOUD': '0',
}


def parser():
    result = argparse.ArgumentParser(description=__doc__, epilog=(
        'Works directly from zsh/bash, from any directory. Default task: PegInHole; '
        'RoboCasa kitchens are not installed. GUI display credentials are preserved, '
        'not repaired: use --headless when no working display is available. '
        'ROS_DOMAIN_ID defaults to 1; 0 is rejected before SDK imports. '
        'PegInHole camera defaults: full-resolution RGB, standalone depth off, '
        'camera point clouds off. Aligned depth and Livox remain enabled. '
        'Explicit GOLEM_CAMERA_* environment settings override these defaults; '
        'other tasks retain their existing defaults. '
        'Ground-truth odometry stays off unless GOLEM_SIM_ODOM is explicitly enabled. '
        'Run scripts/setup_environment.sh --help for the full preparation sequence. '
        'Launch checks prerequisites but never installs or compiles.'))
    result.add_argument('--task', default='PegInHole')
    result.add_argument('--headless', action='store_true')
    result.add_argument('--layout', type=int)
    result.add_argument('--style', type=int)
    result.add_argument('--seed', type=int)
    result.add_argument('--timing', action='store_true')
    result.add_argument('--manipulation-hoist', action='store_true',
                        help='Supported manipulation fixture: existing pelvis rotational hold '
                        'at 2000 N*m/rad while the startup tether is enabled. '
                        'Not free-standing or a balance-performance test.')
    result.add_argument('--record', metavar='PATH', help='Opt-in WebM + evaluation-only JSONL recording')
    result.add_argument('--record-fps', type=float, default=None,
                        help='Frames per simulation second (default 20); not wall real-time')
    result.add_argument('--peg', help='PegInHole catalog peg ID (default: cylinder_1)')
    result.add_argument('--source-hole', help='PegInHole seated source hole (default: cylinder_1)')
    result.add_argument('--lowerbody', choices=('fame_finetune', 'almi_manip_stand'), default=None,
                        help='Opt-in simulator joint/pelvis spawn fit from source configuration: '
                        'FAME lower_start_pos (not its training default), or ALMI first 12 '
                        'default_angles. Pelvis heading is independent of toe yaw; inspect '
                        'actual fit logs. Omit for the exact legacy baked spawn. Does not load '
                        'checkpoints or start/confirm a controller; independently match ROS '
                        'run_experiment.sh lowerbody:=POLICY.')
    return result


def main():
    args = parser().parse_args()
    if args.task != 'PegInHole' and (args.peg is not None or args.source_hole is not None):
        raise RuntimeError('--peg/--source-hole require --task PegInHole')
    domain = os.environ.get('ROS_DOMAIN_ID') or '1'
    if not domain.isascii() or not domain.isdecimal() or not 1 <= int(domain) <= 232:
        raise RuntimeError('ROS_DOMAIN_ID must be 1..232; domain 0 is the real robot bus')
    os.environ['ROS_DOMAIN_ID'] = str(int(domain))
    from setup_simulator import PYTHON, VENV, check
    required = {
        PYTHON: 'scripts/setup_environment.sh simulator',
        ROOT / 'core_ws/install/local_setup.bash': 'core_ws/scripts/build_ros_workspace.sh build --profile interfaces',
        ROOT / 'core_ws/install/magpie_msgs/share/magpie_msgs/local_setup.bash':
            'core_ws/scripts/build_ros_workspace.sh build --profile interfaces',
        VENV / 'livox-interfaces/install/share/livox_ros_driver2/local_setup.bash':
            'scripts/setup_environment.sh simulator --build-interfaces',
        ROOT / 'unitree_sdk2_python/unitree_sdk2py/__init__.py': 'initialize the checked-out Unitree SDK submodule',
    }
    for path, remedy in required.items():
        if not path.is_file():
            raise RuntimeError(f'missing {path}; run: {remedy}')
    check(True)
    command = [str(PYTHON), str(ROOT / 'h1_robocasa/h12_mujoco.py'), '--task', args.task]
    for name in ('layout', 'style', 'seed', 'peg', 'source_hole', 'lowerbody', 'record', 'record_fps'):
        value = getattr(args, name)
        if value is not None:
            command += ['--' + name.replace('_', '-'), str(value)]
    for name in ('headless', 'timing'):
        if getattr(args, name):
            command.append('--' + name)
    # Only local_setup overlays: never chain saved Docker or obsolete venv underlays.
    from setup_simulator import environment
    overlays = [ROOT / 'core_ws/install/local_setup.bash',
                VENV / 'livox-interfaces/install/share/livox_ros_driver2/local_setup.bash']
    env = environment()
    env.setdefault('GOLEM_SIM_ODOM', '0')
    if args.manipulation_hoist:
        env['GOLEM_SIM_TETHER_ROT_STIFFNESS'] = '2000'
        env['GOLEM_SIM_TETHER_ROT_BODY'] = 'pelvis'
        print('Manipulation hoist: pelvis rotational support enabled at 2000 N*m/rad; '
              'NOT free-standing balance. Lower-body confirmation releases the tether.', flush=True)
    if args.task == 'PegInHole':
        # Marker localization uses RGB/CameraInfo; preserve explicit sensor choices.
        for name, value in PEGBOARD_CAMERA_DEFAULTS.items():
            env.setdefault(name, value)
    print(f'SDK source: {ROOT / "unitree_sdk2_python"}; simulation DDS domain: {domain}', flush=True)
    if args.lowerbody is None:
        print('Spawn pose: legacy baked spawn (--lowerbody omitted); no policy selected', flush=True)
    else:
        print(f'Spawn pose: {args.lowerbody} source configuration; independently match '
              f'./scripts/run_experiment.sh lowerbody:={args.lowerbody}; '
              'simulator does not start or confirm controllers', flush=True)
    print('Camera settings: '
          f'scale={env.get("GOLEM_CAMERA_SCALE", "0.5")}, '
          f'standalone depth={env.get("GOLEM_CAMERA_DEPTH", "1")}, '
          f'camera point clouds={env.get("GOLEM_CAMERA_POINTCLOUD", "1")}; '
          'aligned depth and Livox unchanged', flush=True)
    os.execve('/bin/bash', ['/bin/bash', '--noprofile', '--norc', '-c',
              'set -e; source "$1"; source "$2"; shift 2; exec "$@"',
              'golem-simulator', *map(str, overlays), *command], env)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(1)
