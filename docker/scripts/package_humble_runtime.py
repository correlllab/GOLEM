"""Package a private ROS 2 Humble / Python 3.10 runtime into /native-python.

Runs in IsaacDockerfile's magpie_humble stage, with ROS Humble and the built
magpie_msgs sourced. It copies the Python 3.10 interpreter, its standard
library and dist-packages, and every shared library that the interpreter,
/opt/ros/humble, /opt/golem-magpie-humble and those Python trees load
(followed through ldd), into /native-python/{bin,lib}. glibc's loader family
is left out: the Isaac Sim stage runs it on its own, newer glibc. Raises
RuntimeError when ldd reports a library as not found.
"""
from pathlib import Path
import re
import shutil
import subprocess

root = Path('/native-python')
(root / 'bin').mkdir(parents=True)
(root / 'lib').mkdir()
shutil.copy2('/usr/bin/python3.10', root / 'bin/python3.10')
(root / 'bin/python3').symlink_to('python3.10')
shutil.copytree('/usr/lib/python3.10', root / 'lib/python3.10')
shutil.copytree('/usr/lib/python3/dist-packages', root / 'lib/python3.10/site-packages', dirs_exist_ok=True)
# glibc's loader and core libraries stay the host image's, as one family;
# every other native dependency (libddsc and the Python extension modules'
# libraries among them) is copied privately.
glibc = re.compile(r'^(ld-linux.*|lib(c|m|pthread|dl|rt|util|resolv|anl)\.so(?:\..*)?)$')
inputs = [Path('/usr/bin/python3.10')]
for prefix in ('/opt/ros/humble', '/opt/golem-magpie-humble', '/usr/lib/python3.10', '/usr/lib/python3/dist-packages'):
    inputs.extend(p for p in Path(prefix).rglob('*.so*') if p.is_file())
# Walk the dependency graph: each copied library is itself scanned with ldd.
# Libraries under /opt/ are copied whole by IsaacDockerfile, so they are
# skipped here.
seen = set()
while inputs:
    binary = inputs.pop()
    if binary in seen:
        continue
    seen.add(binary)
    output = subprocess.run(['ldd', str(binary)], text=True, capture_output=True).stdout
    for line in output.splitlines():
        if '=> not found' in line:
            raise RuntimeError(f'{binary}: {line.strip()}')
        paths = re.findall(r'(/[^\s()]+)', line)
        for value in paths:
            path = Path(value)
            if glibc.match(path.name) or str(path).startswith('/opt/'):
                continue
            destination = root / 'lib' / path.name
            if not destination.exists():
                shutil.copy2(path, destination)
                inputs.append(path)
