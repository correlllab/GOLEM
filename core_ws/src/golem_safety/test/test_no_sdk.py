'''
golem_safety's code talks ROS only: nothing in it imports the Unitree SDK. tools/ and
test/fixtures are not scanned: the offline fixture tools read the SDK on purpose.
'''

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ('src', 'include', 'launch', 'scripts')
SCANNED_FILES = ('CMakeLists.txt', 'package.xml')


def test_no_sdk_import():
    files = [PACKAGE / name for name in SCANNED_FILES]
    for directory in SCANNED_DIRS:
        found = sorted(p for p in (PACKAGE / directory).rglob('*') if p.is_file())
        assert found, f'nothing to scan in {directory}/'
        files += found
    hits = [str(p.relative_to(PACKAGE)) for p in files
            if 'unitree_sdk2py' in p.read_text(errors='ignore')]
    assert hits == []
