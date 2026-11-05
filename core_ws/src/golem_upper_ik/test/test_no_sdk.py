"""Check that golem_upper_ik's code talks ROS only: nothing in it imports the Unitree SDK.

The package code, launch files and build files are scanned; tools/ and test/ are not.
"""

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ('golem_upper_ik', 'launch')
SCANNED_FILES = ('setup.py', 'setup.cfg', 'package.xml')


def test_no_sdk_import():
    """No scanned file mentions unitree_sdk2py."""
    files = [PACKAGE / name for name in SCANNED_FILES]
    for directory in SCANNED_DIRS:
        found = sorted(p for p in (PACKAGE / directory).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts)
        assert found, f'nothing to scan in {directory}/'
        files += found
    hits = [str(p.relative_to(PACKAGE)) for p in files
            if 'unitree_sdk2py' in p.read_text(errors='ignore')]
    assert hits == []
