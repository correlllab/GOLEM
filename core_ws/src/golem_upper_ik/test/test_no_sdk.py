'''
golem_upper_ik's code talks ROS only: nothing in it imports the Unitree SDK. tools/ and
test/ are not scanned: the offline fixture tools read the old packages on purpose.
'''

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ('golem_upper_ik', 'launch')
OPTIONAL_DIRS = ('scripts',)
SCANNED_FILES = ('setup.py', 'setup.cfg', 'package.xml')


def test_no_sdk_import():
    files = [PACKAGE / name for name in SCANNED_FILES]
    for directory in SCANNED_DIRS + OPTIONAL_DIRS:
        found = sorted(p for p in (PACKAGE / directory).rglob('*')
                       if p.is_file() and '__pycache__' not in p.parts)
        if directory in SCANNED_DIRS:
            assert found, f'nothing to scan in {directory}/'
        files += found
    hits = [str(p.relative_to(PACKAGE)) for p in files
            if 'unitree_sdk2py' in p.read_text(errors='ignore')]
    assert hits == []
