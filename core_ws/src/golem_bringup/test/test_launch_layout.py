"""
Where golem_bringup's launch, config and rviz files live, and how they install.

`ros2 launch golem_bringup <file>` finds a file by name anywhere under the
package's share directory and refuses a name it finds twice, so a launch file
name must be unique across launch/h12/ and launch/g1/, and setup.py must install
each file once, at its own relative directory under share/golem_bringup.
"""

from collections import Counter
from pathlib import Path
import runpy

import setuptools

PACKAGE = Path(__file__).resolve().parents[1]

TOP_LEVEL = {
    'h12': ('h12_sim', 'h12_sim_mac', 'h12_real_robot', 'h12_real_desktop',
            'h12_safety_grippers', 'lmg_sim', 'lmg_real'),
    'g1': ('g1_sim', 'g1_real_robot'),
}

INSTALLED = ('launch/**/*.launch.py', 'config/**/*.yaml', 'rviz/**/*.rviz')


def _files(pattern):
    return sorted(path.relative_to(PACKAGE) for path in PACKAGE.glob(pattern))


def _data_files(monkeypatch):
    """Return setup.py's data_files as [(destination, [source relative to PACKAGE])]."""
    captured = {}
    monkeypatch.setattr(setuptools, 'setup', lambda **kwargs: captured.update(kwargs))
    monkeypatch.chdir(PACKAGE)
    runpy.run_path(str(PACKAGE / 'setup.py'))
    return [(Path(destination), [Path(source) for source in sources])
            for destination, sources in captured['data_files']]


def test_launch_names_unique():
    launch_files = _files('launch/**/*.launch.py')
    assert launch_files
    names = Counter(path.name for path in launch_files)
    assert [name for name, count in names.items() if count > 1] == []


def test_top_level_files_exist():
    for robot, names in TOP_LEVEL.items():
        for name in names:
            path = PACKAGE / 'launch' / robot / f'{name}.launch.py'
            assert path.is_file(), f'missing {path.relative_to(PACKAGE)}'


def test_setup_installs_every_launch_file(monkeypatch):
    data_files = _data_files(monkeypatch)
    for pattern in INSTALLED:
        files = _files(pattern)
        assert files, pattern
        for path in files:
            destinations = [destination for destination, sources in data_files
                            if path in sources]
            assert destinations == [Path('share', 'golem_bringup', path.parent)], path
