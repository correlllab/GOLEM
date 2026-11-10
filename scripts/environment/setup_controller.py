"""Sync isolated Frame/Counter libraries and perform offline math/import checks."""

import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'environments/controller'
VENV = PROJECT / '.venv'
PYTHON = VENV / 'bin/python'


def environment():
    env = os.environ.copy()
    env.update(PATH=f'{VENV}/bin:/usr/bin:/bin', VIRTUAL_ENV=str(VENV),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', UV_NO_CONFIG='1',
               UV_PYTHON_DOWNLOADS='never', UV_PROJECT_ENVIRONMENT=str(VENV),
               UV_CACHE_DIR=str(PROJECT / '.cache/uv'))
    env['PYTHONPATH'] = str(ROOT / 'unitree_sdk2_python') + ':' + env.get('PYTHONPATH', '')
    return env


def check():
    if not PYTHON.is_file():
        raise RuntimeError('missing controller environment; run scripts/setup_environment.sh controller')
    overlay = ROOT / 'core_ws/install/local_setup.bash'
    if not overlay.is_file():
        raise RuntimeError('build the experiment workspace profile before controller checks')
    subprocess.run(['/bin/bash', '--noprofile', '--norc', '-c',
                    'set -e; source "$1"; shift; exec "$@"', 'golem-controller',
                    str(overlay), str(PYTHON), str(Path(__file__).with_name('check_controller.py'))],
                   env=environment(), cwd=ROOT, check=True)


def main(check_only=False):
    if not check_only:
        uv = os.environ.get('GOLEM_NATIVE_UV')
        if not uv or not Path(uv).is_file():
            raise RuntimeError('uv executable not found')
        if not VENV.exists():
            subprocess.run([uv, 'venv', '--python', '/usr/bin/python3',
                            '--system-site-packages', str(VENV)], env=environment(), check=True)
        subprocess.run([uv, 'sync', '--locked', '--inexact', '--no-build', '--project',
                        str(PROJECT), '--python', '/usr/bin/python3'], env=environment(), check=True)
    check()
