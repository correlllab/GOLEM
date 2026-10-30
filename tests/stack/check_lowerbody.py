#!/usr/bin/env python3
"""Exercise ROS lower-body control; score private simulator telemetry offline.

Telemetry is JSONL with sim_time, position[3], quaternion[WXYZ],
joint_positions[27] (Unitree wire order), and support_active (band OR fixed base).
It is never published to ROS. Run on an isolated simulation domain only.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time


LIMITS = dict(max_tilt_deg=25., minimum_height=.65, max_height_drop=.20,
              balance_drift=.15, minimum_forward=.15, maximum_forward=1.,
              maximum_lateral=.20, minimum_knee_excursion=.08,
              stopped_drift=.08, maximum_sample_gap=.25)


def validate_row(row):
    for key, size in (('position', 3), ('quaternion', 4), ('joint_positions', 27)):
        if len(row[key]) != size or not all(math.isfinite(x) for x in row[key]):
            raise AssertionError(f'Invalid telemetry {key}')
    if not math.isfinite(row['sim_time']) or row['sim_time'] < 0:
        raise AssertionError('Invalid telemetry sim_time')
    if type(row.get('support_active')) is not bool:
        raise AssertionError('Telemetry must report actual support_active (band OR fixed base)')
    norm = math.sqrt(sum(x*x for x in row['quaternion']))
    if not .99 <= norm <= 1.01:
        raise AssertionError('Telemetry quaternion must be normalized WXYZ')


def tilt_deg(quaternion):
    w, x, y, z = quaternion
    return math.degrees(math.acos(max(-1., min(1., 1. - 2.*(x*x+y*y)))))


def distance(a, b):
    return math.hypot(a['position'][0]-b['position'][0], a['position'][1]-b['position'][1])


def score_run(rows, phases, policies, expected_walk_policy='walk'):
    """Score a completed 5 s balance, 3 s walk, 3 s stop experiment."""
    selected = {}
    for name, duration in (('balance', 5.), ('walk', 3.), ('stop', 3.)):
        start, end = phases[name]
        assert end - start >= duration - .001, f'{name} phase too short'
        samples = [r for r in rows if start <= r['sim_time'] <= end]
        assert samples, f'No {name} telemetry'
        for row in samples:
            validate_row(row)
            assert not row['support_active'], f'{name}: external support still active'
        stamps = [r['sim_time'] for r in samples]
        gaps = [b-a for a, b in zip(stamps, stamps[1:])]
        assert all(0 < g <= LIMITS['maximum_sample_gap'] for g in gaps), f'{name}: stale/nonmonotonic telemetry'
        assert stamps[0]-start <= .25 and end-stamps[-1] <= .25, f'{name}: incomplete telemetry coverage'
        assert max(tilt_deg(r['quaternion']) for r in samples) <= LIMITS['max_tilt_deg'], f'{name}: excessive tilt'
        selected[name] = samples
    initial = selected['balance'][0]
    baseline = statistics.median(r['position'][2] for r in selected['balance'] if r['sim_time'] <= phases['balance'][0]+1.)
    all_rows = sum(selected.values(), [])
    minimum_height = min(r['position'][2] for r in all_rows)
    assert minimum_height >= max(LIMITS['minimum_height'], baseline-LIMITS['max_height_drop']), 'Base collapsed or lost height'
    balance_drift = max(distance(initial, r) for r in selected['balance'])
    assert balance_drift <= LIMITS['balance_drift'], 'Standing drift exceeds limit'

    def policy_at(stamp):
        eligible = [(t, p) for t, p in policies if t <= stamp]
        assert eligible, 'Missing active_policy state'
        return max(eligible, key=lambda pair: pair[0])[1]
    for row in selected['balance']:
        assert policy_at(row['sim_time']) == 'almi', 'ALMI not continuously active during balance'
    # Handover is allowed for the first second; the last two seconds must run
    # the requested locomotion policy. Stop must return to ALMI for its last second.
    for phase, policy, settle in (('walk', expected_walk_policy, 1.), ('stop', 'almi', 2.)):
        for row in selected[phase]:
            if row['sim_time'] >= phases[phase][0]+settle:
                assert policy_at(row['sim_time']) == policy, f'{phase}: expected active policy {policy}'
    first, last = selected['walk'][0], selected['walk'][-1]
    w, x, y, z = first['quaternion']
    yaw = math.atan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z))
    dx, dy = [last['position'][i]-first['position'][i] for i in (0, 1)]
    forward = dx*math.cos(yaw)+dy*math.sin(yaw)
    lateral = -dx*math.sin(yaw)+dy*math.cos(yaw)
    assert LIMITS['minimum_forward'] <= forward <= LIMITS['maximum_forward'], f'Forward displacement {forward:.3f} m outside limits'
    assert abs(lateral) <= LIMITS['maximum_lateral'], f'Lateral deviation {lateral:.3f} m exceeds limit'
    excursions = [max(r['joint_positions'][i] for r in selected['walk'])-min(r['joint_positions'][i] for r in selected['walk']) for i in (3, 9)]
    assert min(excursions) >= LIMITS['minimum_knee_excursion'], 'Both knees must articulate during walking'
    stopped = [r for r in selected['stop'] if r['sim_time'] >= phases['stop'][1]-1.]
    stop_drift = max(distance(stopped[0], r) for r in stopped)
    assert stop_drift <= LIMITS['stopped_drift'], 'Robot did not settle after zero cmd_vel'
    return dict(baseline_height=baseline, minimum_height=minimum_height,
                balance_drift=balance_drift, forward_displacement=forward,
                lateral_displacement=lateral, knee_excursions=excursions,
                stopped_drift=stop_drift, limits=LIMITS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-walk-policy', choices=('walk', 'almi'), default='walk')
    parser.add_argument('--confirm-engage', action='store_true', help='Call confirm_engage on the isolated simulation domain')
    parser.add_argument('--wall-timeout', type=float, default=600.)
    args = parser.parse_args()
    domain = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    if not 1 <= domain <= 232:
        parser.error('Set explicit isolated ROS_DOMAIN_ID in 1..232 (never hardware domain 0)')
    if not math.isfinite(args.wall_timeout) or args.wall_timeout <= 0:
        parser.error('wall timeout must be finite and positive')
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from geometry_msgs.msg import Twist
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from unitree_hg.msg import LowState
    rclpy.init()
    node = rclpy.create_node('lowerbody_stack_check')
    publisher = node.create_publisher(Twist, '/cmd_vel', 10)
    state = dict(clock=None, policy=None, lowstate_at=None, error=None)
    policies, rows, phases = [], [], {}
    def on_clock(msg):
        stamp = msg.clock.sec+msg.clock.nanosec*1e-9
        if state['clock'] is not None and stamp < state['clock']:
            state['error'] = 'Simulation clock reset during test'
        state['clock'] = stamp
    def on_policy(msg):
        state['policy'] = msg.data
        policies.append((state['clock'] if state['clock'] is not None else 0., msg.data))
    def on_state(msg):
        state['lowstate_at'] = time.monotonic()
    node.create_subscription(Clock, '/clock', on_clock, 10)
    node.create_subscription(String, '/lowerbody/active_policy', on_policy,
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
    node.create_subscription(LowState, '/lowstate', on_state, qos_profile_sensor_data)
    confirm = node.create_client(Trigger, '/lowerbody/confirm_engage')
    future = None
    file_position, pending = 0, ''
    deadline = time.monotonic()+args.wall_timeout
    def pump(speed=0.):
        nonlocal file_position, pending
        assert time.monotonic() < deadline, 'Stack test wall timeout'
        cmd = Twist(); cmd.linear.x = speed; publisher.publish(cmd)
        rclpy.spin_once(node, timeout_sec=.02)
        assert state['error'] is None, state['error']
        if args.telemetry.exists():
            with args.telemetry.open() as stream:
                stream.seek(file_position)
                pending += stream.read()
                file_position = stream.tell()
            while '\n' in pending:
                line, pending = pending.split('\n', 1)
                row = json.loads(line)
                validate_row(row)
                assert not rows or row['sim_time'] > rows[-1]['sim_time'], 'Telemetry reset/nonmonotonic timestamp'
                rows.append(row)
                if not row['support_active']:
                    assert row['position'][2] >= LIMITS['minimum_height'], 'Unsupported base collapsed before completing acceptance'
                    assert tilt_deg(row['quaternion']) <= LIMITS['max_tilt_deg'], 'Unsupported base exceeded tilt limit before completing acceptance'
    report = dict(success=False, expected_walk_policy=args.expected_walk_policy, phases=phases)
    try:
        while True:
            pump()
            if args.confirm_engage and future is None and confirm.service_is_ready():
                future = confirm.call_async(Trigger.Request())
            if future is not None and future.done():
                response = future.result()
                assert response is not None and response.success, f'Engage rejected: {response}'
            if (state['clock'] is not None and state['policy'] == 'almi' and
                state['lowstate_at'] is not None and time.monotonic()-state['lowstate_at'] < 2. and
                rows and abs(rows[-1]['sim_time']-state['clock']) < .25 and not rows[-1]['support_active']):
                break
        for name, duration, speed in (('balance', 5., 0.), ('walk', 3., .15), ('stop', 3., 0.)):
            start = state['clock']
            print(f'PHASE {name} sim_time={start:.3f} cmd_vx={speed}', flush=True)
            while state['clock']-start < duration:
                pump(speed)
                assert time.monotonic()-state['lowstate_at'] < 2., 'LowState stopped arriving'
            phases[name] = (start, state['clock'])
        # Allow the final private telemetry record to flush without advancing a scored phase.
        while not rows or rows[-1]['sim_time'] < phases['stop'][1]-.05:
            pump()
        report.update(score_run(rows, phases, policies, args.expected_walk_policy))
        report['success'] = True
        print('LOWERBODY_STACK_PASS', flush=True)
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        # /cmd_vel is latched by the controller: explicit zero is mandatory.
        for _ in range(5):
            publisher.publish(Twist())
            rclpy.spin_once(node, timeout_sec=.02)
        report['policies'] = policies
        report['telemetry_samples'] = len(rows)
        if rows:
            report['minimum_height_observed'] = min(r['position'][2] for r in rows)
            report['maximum_tilt_observed_deg'] = max(tilt_deg(r['quaternion']) for r in rows)
            report['final_sim_time'] = rows[-1]['sim_time']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
