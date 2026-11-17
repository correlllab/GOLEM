"""Compare h12_safety_layer's derived limits with golem_safety's, config by config.

Usage (in the golem_ros container, with core_ws sourced):
  python3 compare_limits.py <h12_safety_layer checkout> > limits.json

Loads every config of the old checkout with its own load_config, dumps the
golem_safety config of the same name with golem_safety_dump_config, and prints,
per config and per array, the joints whose values differ (old, new). It also
compares the two joint tables. Imports no DDS: config.py's channel factory import
is stubbed.
"""
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

OLD = Path(sys.argv[1])
DUMP = '/tmp/gs/install/golem_safety/lib/golem_safety/golem_safety_dump_config'
NEW_TOOLS = Path('/home/code/core_ws/src/golem_safety/tools')

# Stub the SDK's channel module: load_config never calls it.
channel = types.ModuleType('unitree_sdk2py.core.channel')
channel.ChannelFactoryInitialize = None
sys.modules.setdefault('unitree_sdk2py', types.ModuleType('unitree_sdk2py'))
sys.modules.setdefault('unitree_sdk2py.core', types.ModuleType('unitree_sdk2py.core'))
sys.modules['unitree_sdk2py.core.channel'] = channel
sys.path.insert(0, str(OLD))
from h12_safety_layer.core import joint_limits as old_jl  # noqa: E402
from h12_safety_layer.core.config import load_config  # noqa: E402

PAIRS = {  # old key -> new key(s); q limits are (n, 2) in the old loader
    'q_clip_limits': ('q_clip_low', 'q_clip_high'),
    'dq_clip_limits': ('dq_clip',),
    'tau_clip_limits': ('tau_clip',),
    'kp_clip_max': ('kp_max',),
    'kd_clip_max': ('kd_max',),
    'q_estop_limits': ('q_estop_low', 'q_estop_high'),
    'dq_estop_limits': ('dq_estop',),
    'tau_estop_limits': ('tau_estop',),
}


def diff(name, old, new, names):
    """Return {joint: [old, new]} where the arrays differ beyond 1e-9."""
    old = np.broadcast_to(np.asarray(old, float), (len(names),))
    new = np.broadcast_to(np.asarray(new, float), (len(names),))
    return {names[i]: [round(float(old[i]), 6), round(float(new[i]), 6)]
            for i in np.nonzero(np.abs(old - new) > 1e-9)[0]}


def joint_tables():
    """Return the joints whose URDF-table limits differ between the two packages."""
    sys.path.insert(0, str(NEW_TOOLS))
    import h12_joint_limits as new_jl
    out = {}
    if list(new_jl.JOINT_NAMES) != list(old_jl.JOINT_NAMES):
        out['names'] = [list(old_jl.JOINT_NAMES), list(new_jl.JOINT_NAMES)]
    for table in ('URDF_POSITION_LIMITS', 'URDF_VELOCITY_LIMITS', 'URDF_TORQUE_LIMITS'):
        old_t, new_t = getattr(old_jl, table), getattr(new_jl, table)
        rows = {old_jl.JOINT_NAMES[i]: [o, n] for i, (o, n) in enumerate(zip(old_t, new_t))
                if o != n}
        if rows:
            out[table] = rows
    return out


def main():
    """Print the per-config differences as JSON."""
    names = old_jl.JOINT_NAMES
    report = {}
    for path in sorted((OLD / 'config').glob('*.yaml')):
        old = load_config(path)['limits']
        dumped = subprocess.run([DUMP, 'h12', path.stem], capture_output=True, text=True)
        if dumped.returncode != 0:
            report[path.stem] = {'missing_in_golem_safety': dumped.stderr.strip()[-200:]}
            continue
        new = json.loads(dumped.stdout)['limits']
        entry = {}
        for okey, nkeys in PAIRS.items():
            if len(nkeys) == 2:
                for col, nkey in enumerate(nkeys):
                    d = diff(nkey, old[okey][:, col], new[nkey], names)
                    if d:
                        entry[nkey] = d
            else:
                d = diff(nkeys[0], old[okey], new[nkeys[0]], names)
                if d:
                    entry[nkeys[0]] = d
        report[path.stem] = entry or 'identical'
    report['_joint_tables'] = joint_tables()
    print(json.dumps(report, indent=1))


main()
