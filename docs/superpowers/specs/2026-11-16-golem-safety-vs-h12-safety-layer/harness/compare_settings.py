"""Print each config's non-limit settings in h12_safety_layer and golem_safety, side by side.

Usage (golem_ros container, core_ws sourced): python3 compare_settings.py <old checkout>
"""
import json, subprocess, sys, types
from pathlib import Path
OLD = Path(sys.argv[1])
ch = types.ModuleType('unitree_sdk2py.core.channel'); ch.ChannelFactoryInitialize = None
for m in ('unitree_sdk2py', 'unitree_sdk2py.core'):
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules['unitree_sdk2py.core.channel'] = ch
sys.path.insert(0, str(OLD))
from h12_safety_layer.core.config import load_config  # noqa: E402
DUMP = '/tmp/gs/install/golem_safety/lib/golem_safety/golem_safety_dump_config'
for p in sorted((OLD / 'config').glob('*.yaml')):
    o = load_config(p)
    r = subprocess.run([DUMP, 'h12', p.stem], capture_output=True, text=True)
    n = json.loads(r.stdout) if r.returncode == 0 else {}
    print(p.stem, json.dumps({
        'old': {'mode': o['mode'], 'hz': o['control']['publish_hz'], 'estop': o['estop'],
                'domain': o['network']['domain_id'], 'log': o['logging']['enabled'],
                'topics': o['topics']},
        'new': {k: n.get(k) for k in ('mode', 'control', 'stale', 'single_writer', 'estop',
                                      'estop_response', 'topics')} | {'log': n.get('logging', {}).get('enabled')},
    }))
