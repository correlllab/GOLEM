"""Package a private Humble/Python runtime without replacing the host's libc."""
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
# The destination has a newer glibc. Keep its loader/libc family together;
# copy other native dependencies privately, including libddsc and Python modules.
glibc = re.compile(r'^(ld-linux.*|lib(c|m|pthread|dl|rt|util|resolv|anl)\.so(?:\..*)?)$')
inputs = [Path('/usr/bin/python3.10')]
for prefix in ('/opt/ros/humble', '/opt/golem-magpie-humble', '/usr/lib/python3.10', '/usr/lib/python3/dist-packages'):
    inputs.extend(p for p in Path(prefix).rglob('*.so*') if p.is_file())
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
