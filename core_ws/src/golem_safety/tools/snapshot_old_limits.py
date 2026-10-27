#!/usr/bin/env python3
"""Writes what the old safety layers derive from their configs, for test_limit_parity.py.

For every config of h12_safety_layer and g1_safety_layer it runs that package's own
load_config and writes test/fixtures/limits/<robot>/<config>.json, relative to this package:

  limits  the derived clip and estop arrays, indexed by body slot, and kp_max / kd_max
  old     the other values the old loader returns, under their old meaning: mode, topics
          (rt/ names), publish_hz, the stale times, state_lag_ms, the engage gate, the
          arm_sdk ramps, the estop response and logging

The new tests import neither old package: they compare golem_safety_dump_config against
these files. Both old packages must import, so run it in the G1-mode ros container after the
workspace is built and sourced:

  python3 /home/code/core_ws/src/golem_safety/tools/snapshot_old_limits.py

kp_max and kd_max are per-joint vectors in the old loaders; golem_safety keeps one number
each, so this fails for a config whose entries differ.
"""
import json
import os
from pathlib import Path
import sys

# The G1 loader replaces logging.base_dir with a per-run directory when this is set.
os.environ.pop('G1_SAFETY_LOG_DIR', None)

PACKAGE = Path(__file__).resolve().parents[1]
SOURCES = PACKAGE.parent
OUTPUT = PACKAGE / 'test' / 'fixtures' / 'limits'

LIMIT_KEYS = {
    'q_clip_low': ('q_clip_limits', 0),
    'q_clip_high': ('q_clip_limits', 1),
    'dq_clip': ('dq_clip_limits', None),
    'tau_clip': ('tau_clip_limits', None),
    'q_estop_low': ('q_estop_limits', 0),
    'q_estop_high': ('q_estop_limits', 1),
    'dq_estop': ('dq_estop_limits', None),
    'tau_estop': ('tau_estop_limits', None),
}


def derived_limits(config, where):
    """The old derived arrays as lists, with kp_max and kd_max as single numbers."""
    limits = config['limits']
    out = {}
    for key, (old_key, column) in LIMIT_KEYS.items():
        array = limits[old_key]
        out[key] = [float(x) for x in (array if column is None else array[:, column])]
    for key, old_key in (('kp_max', 'kp_clip_max'), ('kd_max', 'kd_clip_max')):
        values = {float(x) for x in limits[old_key]}
        if len(values) != 1:
            sys.exit(f'{where}: {old_key} differs between joints; golem_safety keeps one number')
        out[key] = values.pop()
    return out


def old_h12(config):
    mode = config['mode']
    estop = config['estop']
    return {
        'mode': mode,
        'topics': dict(config['topics']),
        'publish_hz': config['control']['publish_hz'],
        'upper_stale_seconds': estop['upper_stale_seconds'],
        'upper_stale_active': mode == 'split_mode',
        'lower_stale_seconds': None,
        'state_stale_seconds': None,
        'state_lag_ms': estop['state_lag_ms'],
        'engage': None,
        'arm_sdk': None,
        'estop_response': None,
        'logging': dict(config['logging']),
    }


def old_g1(config):
    from g1_safety_layer.core.safety_layer import UPPER_STALE_ESTOP_SECONDS

    mode = config['mode']
    response = config['estop_response']
    return {
        'mode': mode,
        'topics': dict(config['topics']),
        'publish_hz': config['control']['publish_hz'],
        'upper_stale_seconds': UPPER_STALE_ESTOP_SECONDS,
        'upper_stale_active': mode in ('split_mode', 'arm_sdk_mode'),
        'lower_stale_seconds': config['control']['lower_stale_estop_seconds'],
        'state_stale_seconds': config['control']['state_stale_estop_seconds'],
        'state_lag_ms': None,
        'engage': dict(config['engage']),
        'arm_sdk': dict(config['arm_sdk']),
        'estop_response': {
            'mode': response['mode'],
            'kd': None if response['kd'] is None else [float(x) for x in response['kd']],
        },
        'logging': dict(config['logging']),
    }


def snapshot(robot, load_config, describe, config_dir):
    for path in sorted(config_dir.glob('*.yaml')):
        where = f'{robot}/{path.stem}'
        config = load_config(path)
        target = OUTPUT / robot / f'{path.stem}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'robot': robot,
            'config': path.stem,
            'limits': derived_limits(config, where),
            'old': describe(config),
        }
        target.write_text(json.dumps(data, indent=1) + '\n')
        print(f'wrote {target.relative_to(PACKAGE)}')


def main():
    from g1_safety_layer.core.config import load_config as load_g1
    from h12_safety_layer.core.config import load_config as load_h12

    snapshot('h12', load_h12, old_h12, SOURCES / 'h12_safety_layer' / 'config')
    snapshot('g1', load_g1, old_g1, SOURCES / 'g1' / 'g1_safety_layer' / 'config')


if __name__ == '__main__':
    main()
