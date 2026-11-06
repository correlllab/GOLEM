"""
Check that golem_lowerbody_rl's code talks ROS only: no Unitree SDK, no raw CycloneDDS.

The package code, launch files, build files and the Python files beside the policies'
weights are scanned; test/ is not.
"""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ('golem_lowerbody_rl', 'launch')
SCANNED_FILES = ('setup.py', 'setup.cfg', 'package.xml')
FORBIDDEN = ('unitree_sdk2py', 'cyclonedds')


def scanned_files() -> list[Path]:
    """Return every file the check reads, failing if a scanned directory is empty."""
    files = [PACKAGE / name for name in SCANNED_FILES]
    for directory in SCANNED_DIRS:
        found = sorted(p for p in (PACKAGE / directory).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts)
        assert found, f'nothing to scan in {directory}/'
        files += found
    policy_scripts = sorted((PACKAGE / 'policies').rglob('*.py'))
    assert policy_scripts, 'no Python file under policies/'
    return files + policy_scripts


def test_no_sdk_import() -> None:
    """No scanned file mentions unitree_sdk2py or cyclonedds."""
    hits = [f'{p.relative_to(PACKAGE)}: {name}' for p in scanned_files()
            for name in FORBIDDEN if name in p.read_text(errors='ignore')]
    assert hits == []
