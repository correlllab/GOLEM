"""Prepare, build, and check selected GOLEM ROS 2 packages without starting nodes."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
ROOT = WORKSPACE.parent
ROS_ENV = WORKSPACE / 'environments'
VENV = ROS_ENV / '.venv'
PYTHON = VENV / 'bin/python'
INSTALL = WORKSPACE / 'install'
LOCAL_ROS = VENV / 'ros-debs/opt/ros/humble'
LOCAL_DEBS = ('ros-humble-rosidl-generator-dds-idl', 'ros-humble-nav2-msgs')
CORE = ('estop', 'h12_safety_layer')
INTERFACES = CORE + ('custom_ros_messages', 'magpie_msgs')
MANIPULATION = INTERFACES + (
    'unitree_hg', 'h12_ros2_model',
    'h12_ros2_controller', 'magpie_control', 'h12_lowerbody_rl', 'h12_skills',
)


def package_path(name):
    if name == 'unitree_hg':
        return WORKSPACE / 'src/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg'
    return WORKSPACE / 'src' / name


def packages(profile):
    return {'core': CORE, 'interfaces': INTERFACES, 'manipulation': MANIPULATION}[profile]


def run(command, **kwargs):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def environment():
    env = os.environ.copy()
    env['PATH'] = str(VENV / 'bin') + ':/usr/bin:/bin'
    # System dist-packages can contain a stale editable SDK .pth hook. Explicit
    # checkout precedence and an import provenance check protect this dependency.
    env['PYTHONPATH'] = str(ROOT / 'unitree_sdk2_python') + os.pathsep + env.get('PYTHONPATH', '')
    env['VIRTUAL_ENV'] = str(VENV)
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['UV_NO_CONFIG'] = '1'
    env['UV_PYTHON_DOWNLOADS'] = 'never'
    env['UV_CACHE_DIR'] = str(WORKSPACE / '.cache/uv')
    env['UV_PROJECT_ENVIRONMENT'] = str(VENV)
    env['COLCON_HOME'] = str(VENV / 'colcon-home')
    env['COLCON_DEFAULTS_FILE'] = '/dev/null'
    if LOCAL_ROS.is_dir():
        for key, suffix in (
            ('AMENT_PREFIX_PATH', ''), ('CMAKE_PREFIX_PATH', ''),
            ('LD_LIBRARY_PATH', 'lib'), ('PYTHONPATH', 'local/lib/python3.10/dist-packages'),
        ):
            env[key] = str(LOCAL_ROS / suffix) + os.pathsep + env.get(key, '')
    return env


def prerequisites():
    if sys.version_info[:2] != (3, 10) or Path(sys.executable) != Path('/usr/bin/python3'):
        raise RuntimeError('use ./scripts/build_ros_workspace.sh with system /usr/bin/python3 (3.10)')
    if os.environ.get('ROS_DISTRO') != 'humble' or os.environ.get('PYTHONNOUSERSITE') != '1':
        raise RuntimeError('use ./scripts/build_ros_workspace.sh to obtain the clean Humble environment')
    required = [line.strip() for line in (ROS_ENV / 'apt-packages.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
    missing = []
    for name in required:
        result = subprocess.run(
            ['/usr/bin/dpkg-query', '-W', '-f=${Status}', name],
            capture_output=True, text=True,
        )
        if result.returncode or result.stdout != 'install ok installed':
            missing.append(name)
    if missing:
        raise RuntimeError('missing apt prerequisites (not installed automatically): ' + ' '.join(missing))
    if not (ROOT / 'unitree_sdk2_python/setup.py').is_file():
        raise RuntimeError('missing checkout SDK; submodule changes are outside this bootstrap')
    print('PASS: system Python 3.10, clean Humble, apt build prerequisites', flush=True)


def require_venv():
    if not PYTHON.is_file():
        raise RuntimeError('ROS venv missing; run: ./scripts/build_ros_workspace.sh setup')
    run([PYTHON, '-c',
         'import sys; from pathlib import Path; '
         f'assert Path(sys.prefix) == Path({str(VENV)!r}); '
         'assert sys.version_info[:2] == (3, 10); '
         'assert sys.base_prefix == "/usr"'], env=environment())


def check(built=False, profile='manipulation'):
    require_venv()
    command = [PYTHON, HERE / 'check_ros_workspace.py']
    if built:
        overlay = INSTALL / 'local_setup.bash'
        if not overlay.is_file():
            raise RuntimeError('native install overlay missing; build the requested profile first')
        command += ['--built', '--profile', profile]
        # local_setup deliberately avoids the setup.bash chain of saved underlays.
        command = ['/bin/bash', '--noprofile', '--norc', '-c',
                   'set -e; source "$1"; shift; exec "$@"', 'ros-workspace', overlay, *command]
    run(command, env=environment(), cwd=ROOT)


def build_command(profile):
    selected = packages(profile)
    return [
        PYTHON, '-m', 'colcon', '--log-base', WORKSPACE / 'log', 'build',
        '--build-base', WORKSPACE / 'build', '--install-base', INSTALL,
        '--executor', 'sequential', '--symlink-install',
        '--paths', *(package_path(name) for name in selected),
        '--packages-select', *selected,
        '--event-handlers', 'console_cohesion-', 'console_direct-', 'console_start_end+', 'console_stderr+',
        '--cmake-args', '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF', '-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF',
        f'-DPython3_EXECUTABLE={PYTHON}', f'-DPYTHON_EXECUTABLE={PYTHON}',
    ]


def local_ros_debs():
    # apt-get download is unprivileged; dpkg-deb only extracts into this venv.
    # No package installation, maintainer scripts, system writes, or apt update.
    downloads = VENV / 'ros-debs/downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    for name in LOCAL_DEBS:
        run(['/usr/bin/apt-get', 'download', name], cwd=downloads, env=environment())
        archives = list(downloads.glob(name + '_*.deb'))
        if len(archives) != 1:
            raise RuntimeError(f'expected one {name} archive in {downloads}; remove stale downloads manually')
        run(['/usr/bin/dpkg-deb', '--extract', archives[0], VENV / 'ros-debs'], env=environment())


def manipulation_prerequisites():
    run([PYTHON, '-c',
         'from ament_index_python.packages import get_package_prefix; '
         'get_package_prefix("rosidl_generator_dds_idl")'], env=environment())


def parser():
    result = argparse.ArgumentParser(
        prog='build_ros_workspace.sh',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples (from core_ws):
  ./scripts/build_ros_workspace.sh setup --local-ros-debs
  ./scripts/build_ros_workspace.sh build --profile core
  ./scripts/build_ros_workspace.sh build --profile manipulation --compile-jobs 2
  ./scripts/build_ros_workspace.sh check --built --profile manipulation

The executable also works by absolute path from any directory.

setup: local uv venv, system Python 3.10 + system-site-packages, locked ROS
       NumPy 1 dependencies, and --no-deps SDK install from THIS checkout.
       uv sync --locked --inexact --no-build rejects stale locks, uses wheels
       only, and retains SDK/ROS extras already installed in the venv.
       --local-ros-debs optionally downloads/extracts ONLY the DDS IDL generator
       and nav2_msgs .deb files under the venv (no sudo/system installation).
check: apt prerequisites and dependency imports only; --built checks native
       installed core imports and generated interfaces, without node creation.
build: compile only the selected package profile, using the venv Python.
       core: estop and h12_safety_layer (2 packages).
       interfaces: core plus custom_ros_messages and magpie_msgs (4 packages).
       manipulation: interfaces plus unitree_hg, h12_ros2_model,
         h12_ros2_controller, magpie_control, h12_lowerbody_rl and h12_skills
         (10 packages; arm control, grippers, standing and scripted skills).

Build parallelism: packages build ONE AT A TIME. --compile-jobs limits
concurrent compiler/build-tool jobs inside the current package (default: 2).
It does not mean two packages or two ROS nodes, and has little effect on
pure Python packages. Increase it for faster C/C++ builds at higher CPU/RAM
usage, or use 1 for the lowest build concurrency.

Outputs (under core_ws): environments/.venv/, .cache/uv/, and
{build,install,log}/ (plus existing package setup.py staging).
No Docker overlays are sourced.
No arbitrary colcon arguments, packages-up-to, bringup, AI/model_server,
navigation/SLAM, MJPC, simulator, assets updates, or system installation.

Dependency maintenance (from core_ws; uv 0.13.4):
  export UV_NO_CONFIG=1 UV_PYTHON_DOWNLOADS=never
  export UV_CACHE_DIR="$PWD/.cache/uv"
  uv add --project environments --python /usr/bin/python3 --no-sync 'name==version'
  uv lock --project environments --python /usr/bin/python3
  uv lock --project environments --python /usr/bin/python3 --check --offline
  ./scripts/build_ros_workspace.sh setup
Edit environments/pyproject.toml for pin changes/removals, then run uv lock.
Use uv lock --upgrade-package NAME with the same project/Python flags to refresh
a transitive dependency within the direct constraints. Never hand-edit uv.lock.
The lock covers only Python libraries, not apt/ROS packages, C/C++ libraries,
controller ABI requirements, or the SDK revision. Apt prerequisites live in
environments/apt-packages.txt. The SDK comes from the existing git-pinned
submodule, outside the uv graph: apt supplies OpenCV, not opencv-python.

Compilation is NOT full runtime qualification: controller math requires a
separate NumPy 2 environment (Crocoddyl >=3.2.1, pin >=3.9, Pink >=4.1).
Torch/policies, physical drivers, skill service availability and controller
runtime are not qualified here. No command launches nodes or creates a DDS
participant, controller, simulator, or serial connection.

Manipulation compilation needs ros-humble-rosidl-generator-dds-idl. Skills runtime
also needs ros-humble-nav2-msgs. If missing, request a system administrator
to install them or explicitly use setup --local-ros-debs. Download failures
(including stale apt indexes) are errors, not successful setup.''',
    )
    sub = result.add_subparsers(dest='command', required=True)
    setup_parser = sub.add_parser('setup', help='check apt prerequisites and sync locked local ROS dependencies',
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  epilog='Dependency and lock maintenance commands: build_ros_workspace.sh --help')
    setup_parser.add_argument('--local-ros-debs', action='store_true')
    check_parser = sub.add_parser('check', help='non-actuating dependency/import checks')
    check_parser.add_argument('--built', action='store_true',
                              help='also verify compiled packages and generated interfaces')
    check_parser.add_argument('--profile', choices=('core', 'interfaces', 'manipulation'),
                              default='manipulation', help='built package profile to verify (default: manipulation)')
    build_parser = sub.add_parser('build', help='compile only a fixed package profile',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    build_parser.add_argument('--profile', choices=('core', 'interfaces', 'manipulation'),
                              default='manipulation',
                              help='core: safety/estop; interfaces: core + messages; '
                                   'manipulation: arm/gripper/standing/skills stack')
    build_parser.add_argument('--compile-jobs', type=int, default=2, choices=range(1, 65),
                              metavar='1..64', help='maximum compiler jobs inside each package; '
                                                   'packages always build one at a time')
    return result


def main():
    args = parser().parse_args()
    prerequisites()
    if args.command == 'setup':
        uv = os.environ.get('GOLEM_NATIVE_UV')
        if not uv or not Path(uv).is_file():
            raise RuntimeError('uv executable not found; install uv in your user account')
        if VENV.exists():
            require_venv()
        else:
            run([uv, 'venv', '--python', '/usr/bin/python3', '--system-site-packages', VENV],
                env=environment(), cwd=ROOT)
        run([uv, 'sync', '--locked', '--inexact', '--no-build',
             '--project', ROS_ENV, '--python', '/usr/bin/python3'], env=environment(), cwd=ROOT)
        # SDK OpenCV metadata would pull another cv2/NumPy stack over apt cv_bridge.
        run([uv, 'pip', 'install', '--python', PYTHON, '--no-deps', '--no-build-isolation',
             '--config-settings=--global-option=build',
             f'--config-settings=--global-option=--build-base={VENV / "sdk-build"}',
             ROOT / 'unitree_sdk2_python'], env=environment(), cwd=ROOT)
        if args.local_ros_debs:
            local_ros_debs()
        check()
    elif args.command == 'check':
        check(args.built, args.profile)
    else:
        check()
        if args.profile == 'manipulation':
            try:
                manipulation_prerequisites()
            except subprocess.CalledProcessError as exc:
                raise RuntimeError('manipulation requires ros-humble-rosidl-generator-dds-idl; '
                                   'install externally or run setup --local-ros-debs') from exc
        for name in packages(args.profile):
            if not (package_path(name) / 'package.xml').is_file():
                raise RuntimeError(f'missing selected package: {name}')
        env = environment()
        env['CMAKE_BUILD_PARALLEL_LEVEL'] = str(args.compile_jobs)
        env['MAKEFLAGS'] = f'-j{args.compile_jobs}'
        print(f'Building {args.profile}: {len(packages(args.profile))} packages, '
              f'one package at a time; compiler job limit {args.compile_jobs}.', flush=True)
        run(build_command(args.profile), env=env, cwd=WORKSPACE)
        check(True, args.profile)
        print(f'PASS: {args.profile} compilation and limited built imports; runtime NOT qualified')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        sys.exit(1)
