"""Score one loco-manipulation game run (tests/locomanipulation_game/run.sh) from its artifacts. Host python + numpy.

Reads <result>/ros/commander.jsonl (events, goal results, e-stops, 10 Hz status) and <result>/record/telemetry.jsonl
(the recorder's 50 Hz pelvis pose and joint state). Writes <result>/metrics.json and timeline.png:
  engaged       sim time from the elastic band's release to the end of the run
  falls         pelvis below 0.55 m or tilted past 1 rad after the release
  e-stops       safety heartbeat False (the safety layer's estop latched), deploy faults
  walking       per walk event (its first 1.5 s excluded: settle and acceleration): commanded vs measured pelvis
                velocity in the pelvis frame (xy error, yaw-rate error) and the share of the commanded path covered
  arm goals     reached (both wrists within 5 cm for 1 s), best wrist error, by standing / low / needs-crouch;
                pelvis drop below the run's standing height during low goals
  joints        closest approach to each URDF position limit, peak speed / URDF velocity limit, share of samples
                with the applied torque at the limit, against h12_safety_layer's relax_safety_split table

    python3 analyze_run.py <result dir>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'core_ws/src/h12_safety_layer'))
from h12_safety_layer.core.joint_limits import (JOINT_NAMES, URDF_POSITION_LIMITS, URDF_TORQUE_LIMITS,  # noqa: E402
                                                URDF_VELOCITY_LIMITS)

result = Path(sys.argv[1])
records = [json.loads(line) for line in open(result / 'ros' / 'commander.jsonl')] if (result / 'ros' / 'commander.jsonl').exists() else []
tele = [json.loads(line) for line in open(result / 'record' / 'telemetry.jsonl')]
t = np.array([r['sim_time'] for r in tele])
pos = np.array([r['pelvis_pos'] for r in tele])
quat = np.array([r['pelvis_quat'] for r in tele])
q = np.array([r['q'] for r in tele])
qd = np.array([r['qd'] for r in tele])
tau = np.array([r['tau'] for r in tele])
band = np.array([r['band'] for r in tele])


def rotate_inverse(qw, v):
    """v (world) into the frame of unit quaternions qw (wxyz), row-wise."""
    w, xyz = qw[:, :1], qw[:, 1:]
    xyz = -xyz
    t2 = 2.0 * np.cross(xyz, v)
    return v + w * t2 + np.cross(xyz, t2)


up = rotate_inverse(quat, np.tile([0.0, 0.0, 1.0], (len(quat), 1)))
tilt = np.arccos(np.clip(up[:, 2], -1.0, 1.0))
yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
released = np.flatnonzero(~band)
t0 = t[released[0]] if len(released) else t[-1]
live = t >= t0
fallen = live & ((pos[:, 2] < 0.55) | (tilt > 1.0))
first_fall = float(t[np.flatnonzero(fallen)[0]]) if fallen.any() else None
# sim time is the commander's clock too (use_sim_time)
estops = [r for r in records if r['type'] == 'estop']
faults = [r for r in records if r['type'] == 'deploy_state' and str(r.get('state', '')).startswith('fault')]
end = min([first_fall or math.inf] + [r['t'] for r in estops] + [t[-1]])
standing_z = float(np.median(pos[live & (t < t0 + 3.0), 2])) if (live & (t < t0 + 3.0)).any() else float('nan')

# measured pelvis velocity in its own frame, 0.5 s central differences
k = 12
vel_w = np.zeros_like(pos)
vel_w[k:-k] = (pos[2 * k:] - pos[:-2 * k]) / (t[2 * k:] - t[:-2 * k])[:, None]
vel_b = rotate_inverse(quat, vel_w)
yaw_rate = np.zeros_like(yaw)
yaw_rate[k:-k] = np.unwrap(yaw)[2 * k:] - np.unwrap(yaw)[:-2 * k]
yaw_rate[k:-k] /= (t[2 * k:] - t[:-2 * k])

events = [r for r in records if r['type'] == 'event']
walks, goals = [], []
for i, ev in enumerate(events):
    start = ev['t']
    stop = events[i + 1]['t'] if i + 1 < len(events) else end
    stop = min(stop, end)
    if stop <= start:
        continue
    if ev['kind'] == 'walk':
        m = (t >= start + 1.5) & (t < stop) & live
        if m.sum() < 25:
            continue
        cmd = np.array(ev['cmd'])
        err_xy = np.linalg.norm(vel_b[m, :2] - cmd[:2], axis=1)
        commanded = np.linalg.norm(cmd[:2]) * (t[m][-1] - t[m][0])
        moved = np.dot(np.linalg.norm(vel_b[m, :2], axis=1)[:-1], np.diff(t[m]))
        along = np.dot(vel_b[m, :2][:-1] @ (cmd[:2] / max(np.linalg.norm(cmd[:2]), 1e-9)), np.diff(t[m])) if commanded > 0 else 0.0
        walks.append({'t': start, 'cmd': ev['cmd'], 'xy_error': float(err_xy.mean()),
                      'yaw_rate_error': float(np.abs(yaw_rate[m] - cmd[2]).mean()),
                      'path_ratio': float(along / commanded) if commanded > 0.2 else None, 'moved_m': float(moved)})
    else:
        res = next((r for r in records if r['type'] == 'goal_result' and r['index'] == ev['index']), None)
        m = (t >= start + ev.get('settle', 0.0)) & (t < stop) & live
        goals.append({'t': start, 'low': ev['low'], 'needs_crouch': ev['needs_crouch'],
                      'reached': None if res is None else res['reached'], 'best_error': None if res is None else res['best_error'],
                      'pelvis_drop_max': float(standing_z - pos[m, 2].min()) if m.any() else None,
                      'lowest_target_z': float(min(p[2] for p in ev['poses'])), 'completed': bool(res is not None and stop >= start + ev['duration'])})


def share(values):
    values = [v for v in values if v is not None]
    return None if not values else float(np.mean(values))


lo = np.array([lim['low'] for lim in URDF_POSITION_LIMITS])
hi = np.array([lim['high'] for lim in URDF_POSITION_LIMITS])
lm = live & (t <= end)
joints = {}
for j, name in enumerate(JOINT_NAMES):
    if not lm.any():
        break
    joints[name] = {'closest_to_limit_rad': round(float(np.minimum(q[lm, j] - lo[j], hi[j] - q[lm, j]).min()), 4),
                    'peak_speed_of_limit': round(float(np.abs(qd[lm, j]).max() / URDF_VELOCITY_LIMITS[j]), 3),
                    'torque_at_limit_share': round(float((np.abs(tau[lm, j]) >= 0.999 * URDF_TORQUE_LIMITS[j]).mean()), 4)}
done_goals = [g for g in goals if g['reached'] is not None]
metrics = {
    'result': str(result),
    'engaged_s': round(float(end - t0), 1),
    'band_released_at_s': round(float(t0), 2),
    'fell': first_fall is not None, 'first_fall_s': first_fall,
    'estops': [{k: r[k] for k in ('t', 'detail') if k in r} for r in estops],
    'deploy_faults': [r['state'] for r in faults],
    'standing_pelvis_z': round(standing_z, 3),
    'walking': {'segments': len(walks), 'xy_error_m_s': share([w['xy_error'] for w in walks]),
                'yaw_rate_error': share([w['yaw_rate_error'] for w in walks]),
                'path_ratio': share([w['path_ratio'] for w in walks]), 'distance_m': float(sum(w['moved_m'] for w in walks))},
    'arm_goals': {'judged': len(done_goals), 'reached': share([float(g['reached']) for g in done_goals]),
                  'reached_standing': share([float(g['reached']) for g in done_goals if not g['low']]),
                  'reached_low': share([float(g['reached']) for g in done_goals if g['low']]),
                  'reached_needs_crouch': share([float(g['reached']) for g in done_goals if g['needs_crouch']]),
                  'best_error_m_median': None if not done_goals else float(np.median([g['best_error'] for g in done_goals if g['best_error'] is not None] or [np.nan])),
                  'pelvis_drop_low_goals_m': share([g['pelvis_drop_max'] for g in goals if g['low']])},
    'obstacle_stops': sum(1 for r in records if r['type'] == 'obstacle_stop'),
    'joints_worst': {
        'closest_to_limit': sorted(((v['closest_to_limit_rad'], n) for n, v in joints.items()))[:5],
        'peak_speed_of_limit': sorted(((v['peak_speed_of_limit'], n) for n, v in joints.items()), reverse=True)[:5],
        'torque_at_limit_share': sorted(((v['torque_at_limit_share'], n) for n, v in joints.items()), reverse=True)[:5],
    },
    'goals': goals, 'walks': walks, 'joints': joints,
}
json.dump(metrics, open(result / 'metrics.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
summary = {k: v for k, v in metrics.items() if k not in ('goals', 'walks', 'joints')}
print(json.dumps(summary, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    status = [r for r in records if r['type'] == 'status']
    st = np.array([r['t'] for r in status])
    werr = np.array([[np.nan if e is None else e for e in (r['wrist_error'] or [None, None])] for r in status], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax in axes:
        for ev in events:
            stop = next((e['t'] for e in events if e['t'] > ev['t']), end)
            if ev['kind'] == 'arm_goal':
                ax.axvspan(ev['t'], stop, color='#ffd9c2' if ev['needs_crouch'] else '#dbe8fa', lw=0)
        for r in estops:
            ax.axvline(r['t'], color='#d03030', lw=1.5)
        if first_fall:
            ax.axvline(first_fall, color='#d03030', lw=1.5, ls='--')
    axes[0].plot(t, pos[:, 2], color='#2a78d6', lw=1.4)
    axes[0].set_ylabel('pelvis height (m)')
    if len(st):
        axes[1].plot(st, np.nanmax(werr, axis=1) * 100, color='#2a78d6', lw=1.2)
    axes[1].axhline(5, color='#999', lw=1)
    axes[1].set_ylabel('worse wrist error (cm)')
    axes[1].set_ylim(0, 40)
    cmd_t = [r['t'] for r in status]
    cmd_v = [np.hypot(*r['cmd'][:2]) for r in status]
    axes[2].plot(cmd_t, cmd_v, color='#999', lw=1.2, label='commanded')
    axes[2].plot(t, np.linalg.norm(vel_b[:, :2], axis=1), color='#2a78d6', lw=1.2, label='measured')
    axes[2].set_ylabel('xy speed (m/s)')
    axes[2].set_xlabel('sim time (s)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[0].set_title('Arm goals shaded (orange: needs a crouch), e-stops red, fall dashed', loc='left', fontsize=10)
    fig.tight_layout()
    fig.savefig(result / 'timeline.png', dpi=110)
except ImportError:
    pass
