"""Prepare/check the native simulator environment without starting ROS or DDS."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SIM = ROOT / 'h1_robocasa'
PROJECT = ROOT / 'environments/simulator'
VENV = PROJECT / '.venv'
PYTHON = VENV / 'bin/python'
LIVOX = ROOT / 'core_ws/src/livox_ros_driver2'
LIVOX_SHA = '13eb05e4e6dd7a765b934d0c5fd6236676a57b49'


def run(command, **kwargs):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def environment():
    env = os.environ.copy()
    env.update(PATH=str(VENV / 'bin') + ':/usr/bin:/bin', VIRTUAL_ENV=str(VENV),
               UV_NO_CONFIG='1', UV_PYTHON_DOWNLOADS='never',
               UV_CACHE_DIR=str(PROJECT / '.cache/uv'), UV_PROJECT_ENVIRONMENT=str(VENV))
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT / 'unitree_sdk2_python'), str(SIM),
                                       env.get('PYTHONPATH', '')))
    env['CMAKE_PREFIX_PATH'] = '/opt/ros/humble' + os.pathsep + env.get('CMAKE_PREFIX_PATH', '')
    return env


def require_venv():
    if not PYTHON.is_file():
        raise RuntimeError('simulator venv missing; run scripts/setup_environment.sh simulator')
    run([PYTHON, '-c', 'import sys, site; from pathlib import Path; '
         f'assert Path(sys.prefix) == Path({str(VENV)!r}); '
         'assert sys.version_info[:2] == (3, 10); assert sys.base_prefix == "/usr"; '
         'assert not site.ENABLE_USER_SITE'], env=environment())


def build_livox(jobs):
    if not (ROOT / 'core_ws/install/magpie_msgs/share/magpie_msgs/local_setup.bash').is_file():
        raise RuntimeError('build core interfaces first: core_ws/scripts/build_ros_workspace.sh build --profile interfaces')
    revision = subprocess.check_output(['git', '-C', str(LIVOX), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LIVOX_SHA:
        raise RuntimeError(f'Livox must remain at {LIVOX_SHA}; found {revision}')
    for name in ('CustomMsg.msg', 'CustomPoint.msg'):
        pinned = subprocess.check_output(['git', '-C', str(LIVOX), 'show', f'{LIVOX_SHA}:msg/{name}'])
        if (LIVOX / 'msg' / name).read_bytes() != pinned:
            raise RuntimeError(f'Livox {name} differs from pinned source; refusing local generation')
    # Only generated interfaces are required by the simulator, not the hardware
    # executable/PCL/Livox SDK. Copy the pinned wire definitions without edits.
    import shutil
    source = VENV / 'livox-interfaces/src'
    source.mkdir(parents=True, exist_ok=True)
    for name in ('CMakeLists.txt', 'package.xml'):
        shutil.copyfile(HERE / 'livox_interfaces' / name, source / name)
    (source / 'msg').mkdir(exist_ok=True)
    for name in ('CustomMsg.msg', 'CustomPoint.msg'):
        shutil.copyfile(LIVOX / 'msg' / name, source / 'msg' / name)
    build = VENV / 'livox-interfaces/build'
    run(['cmake', '-S', source, '-B', build,
         f'-DCMAKE_INSTALL_PREFIX={VENV / "livox-interfaces/install"}',
         '-DPython3_EXECUTABLE=/usr/bin/python3', '-DPYTHON_EXECUTABLE=/usr/bin/python3',
         '-DBUILD_TESTING=OFF'], env=environment())
    run(['cmake', '--build', build, '--parallel', jobs], env=environment())
    run(['cmake', '--install', build], env=environment())
    provenance = {'revision': revision, 'source': str(LIVOX), 'sha256': {
        name: hashlib.sha256((LIVOX / 'msg' / name).read_bytes()).hexdigest()
        for name in ('CustomMsg.msg', 'CustomPoint.msg')}}
    (VENV / 'livox-interfaces/provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def check(bridges=False):
    require_venv()
    command = [PYTHON, HERE / 'check_simulator.py']
    if bridges:
        overlays = [ROOT / 'core_ws/install/local_setup.bash',
                    VENV / 'livox-interfaces/install/share/livox_ros_driver2/local_setup.bash']
        for overlay in overlays:
            if not overlay.is_file():
                raise RuntimeError(f'missing {overlay}; build core interfaces, then run scripts/setup_environment.sh simulator --build-interfaces')
        command += ['--bridges']
        command = ['/bin/bash', '--noprofile', '--norc', '-c',
                   'set -e; source "$1"; source "$2"; shift 2; exec "$@"',
                   'simulator-check', *overlays, *command]
    run(command, env=environment(), cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup', help='create isolated venv, sync locked dependencies, install checkout SDK')
    checking = sub.add_parser('check', help='non-actuating dependency/import checks')
    checking.add_argument('--bridges', action='store_true')
    building = sub.add_parser('build-livox-interfaces', help='generate native Livox messages only')
    building.add_argument('--compile-jobs', type=int, choices=range(1, 65), default=2, metavar='1..64')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or sys.base_prefix != '/usr' or os.environ.get('ROS_DISTRO') != 'humble':
        raise RuntimeError('use scripts/setup_environment.sh simulator with system Python 3.10 and clean Humble')
    if args.command == 'setup':
        uv = os.environ.get('GOLEM_NATIVE_UV')
        if not uv or not Path(uv).is_file():
            raise RuntimeError('uv is required; install it in your user account')
        if VENV.exists():
            require_venv()
        else:
            run([uv, 'venv', '--python', '/usr/bin/python3', '--system-site-packages', VENV], env=environment())
        run([uv, 'sync', '--locked', '--inexact', '--project', PROJECT, '--python', '/usr/bin/python3'],
            env=environment())
        run([uv, 'pip', 'install', '--python', PYTHON, '--no-deps', '--no-build-isolation',
             '--config-settings=--global-option=build',
             f'--config-settings=--global-option=--build-base={VENV / "sdk-build"}',
             ROOT / 'unitree_sdk2_python'], env=environment())
        check()
    elif args.command == 'check':
        check(args.bridges)
    else:
        require_venv()
        build_livox(args.compile_jobs)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(1)
