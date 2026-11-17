"""Summarise and compare the two layers' recordings, scenario by scenario.

Usage: python3 analyze.py <out dir>   (prints a plain-text report)

For each scenario and layer: frames and rate from t = 0 to the scenario's end, the
headers written, snapshots of chosen slots at chosen times, and the estop: the first
frame after which every body slot is mode 0 with zero gains once a gain had been
relayed, how many frames followed it, and the layer's own reason line.
"""
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
# scenario -> (times to snapshot, slots to show)
SNAP = {
    'startup_idle': ([-1.0, 0.5], [0, 13, 27]),
    'relay_clip_split': ([0.4, 0.9], [0, 1, 3, 13, 14, 27, 34]),
    'relay_clip_full': ([0.4, 0.9], [0, 1, 3, 13, 14, 27, 34]),
    'nan_unowned_slot': ([0.4, 0.9], [0, 13, 20]),
    'upper_silent': ([0.4, 1.0, 2.4, 3.4], [0, 13]),
    'lower_silent': ([0.4, 1.0, 1.4], [0, 13]),
    'estop_q_edge': ([0.4, 0.9, 1.4], [0, 13]),
    'estop_q_below': ([0.4, 1.4], [0, 13]),
    'estop_dq': ([0.4, 0.9], [0, 13]),
    'state_nan': ([0.4, 0.9], [0, 13]),
    'engage_mm0': ([0.5, 1.5], [0, 13]),
    'foreign_mm': ([0.5, 1.5], [0, 13]),
    'second_writer': ([0.5, 1.2], [0, 13]),
    'hw_estop_silent': ([0.2, 0.8, 1.8], [0, 13]),
    'hw_estop_trigger': ([0.2, 0.8, 1.8], [0, 13]),
}
FIELDS = ('mode', 'q', 'dq', 'tau', 'kp', 'kd')


def latch(t, motor):
    """Return the time of the first all-zero frame after a nonzero gain, or None."""
    gains = np.abs(motor[:, :27, 4]).sum(1) + np.abs(motor[:, :27, 5]).sum(1)
    on = np.nonzero(gains > 0)[0]
    if not len(on):
        return None
    zero = np.nonzero((gains == 0) & (np.arange(len(t)) > on[0])
                      & (np.abs(motor[:, :27, :]).sum((1, 2)) == 0))[0]
    return float(t[zero[0]]) if len(zero) else None


def reason(log):
    """Return the layer's estop and engage lines, shortened."""
    lines = []
    for line in log.splitlines():
        if re.search(r'estop|second writer|engage_refused|input_engaged|input_quiet|'
                     r'input_resumed|stale|Command validation|state_quiet', line, re.I):
            line = re.sub(r'^\[\w+\] \[[\d.]+\] \[safety_node\]: ', '', line)
            lines.append(line[:230])
    # keep first of each kind
    seen, out = set(), []
    for line in lines:
        key = line[:40]
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out[:8]


def main():
    """Print the report."""
    bits = {}
    for scen, (times, slots) in SNAP.items():
        print(f'\n=== {scen}')
        for layer in ('old', 'new'):
            f = OUT / f'{layer}_{scen}.npz'
            if not f.exists():
                print(f'  {layer}: missing')
                continue
            d = np.load(f)
            dur = float(d['duration'])
            keep = d['t'] <= dur
            t, motor = d['t'][keep], d['motor'][keep]
            mm, pr, res = d['mode_machine'][keep], d['mode_pr'][keep], d['reserve'][keep]
            run = (t >= 0) & (t <= dur)
            first = t[0] if len(t) else float('nan')
            print(f'  {layer}: frames in [0,{dur}]: {run.sum()} ({run.sum() / dur:.0f}/s); '
                  f'first frame t={first:.2f}; headers (mm,pr) {sorted(set(zip(mm.tolist(), pr.tolist())))}; '
                  f'reserve {sorted(set(map(tuple, res.tolist())))}')
            lt = latch(t, motor)
            if lt is not None:
                after = (t >= lt).sum()
                last = t[-1]
                print(f'    estop at t={lt:.3f}: {after} frames from then on, last t={last:.3f}')
            elif len(t) and t[-1] < dur - 0.05:
                print(f'    output stops at t={t[-1]:.3f}')
            for ts in times:
                i = np.searchsorted(t, ts) - 1
                if i < 0:
                    print(f'    t={ts}: no frame yet')
                    continue
                row = '; '.join(f'{s}:' + ','.join(f'{v:.4g}' for v in motor[i, s]) for s in slots)
                print(f'    t={ts} [{"/".join(FIELDS)}] {row}')
            if scen.startswith('relay_clip'):
                for lo, hi in ((0.1, 0.45), (0.6, 0.95)):
                    w = (t > lo) & (t < hi)
                    rows = {r.tobytes() for r in motor[w, :27, :].astype(np.float32)}
                    print(f'    distinct body rows in ({lo},{hi}): {len(rows)}')
                    bits.setdefault(scen, {}).setdefault((lo, hi), {})[layer] = rows
            log = (OUT / f'{layer}_{scen}.log').read_text()
            for line in reason(log):
                print(f'    log: {line}')
    print('\n=== bitwise: clipped body slots, old vs new')
    for scen, wins in bits.items():
        for win, by in wins.items():
            print(f'  {scen} {win}: identical={by.get("old") == by.get("new")}')


main()
