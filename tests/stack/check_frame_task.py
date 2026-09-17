#!/usr/bin/env python3
"""Exercise both arms through the real /frame_task server and save measured metrics.

Contract: FrameTaskServer publishes /{left,right}_ee_pose in pelvis coordinates
(metres, quaternion XYZW). UpperController binds those poses to the wrist yaw
links. Unitree LowState slots 13:20 / 20:27 are the left / right arm motors.
The action server interprets goal.duration as WALL seconds; acceptance below
uses /clock simulation seconds with an independent wall timeout.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time


def arm_metrics(start_pose, target_pose, pose, start_joints, joints):
    """Evaluate measured motion; pose = XYZ followed by quaternion XYZW."""
    values = start_pose + target_pose + pose + start_joints + joints
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite arm pose or motor state')
    delta = [target_pose[i] - start_pose[i] for i in range(3)]
    distance = math.sqrt(sum(v*v for v in delta))
    if distance < 1e-6:
        raise ValueError('Motion goal must differ from its measured start')
    moved = [pose[i] - start_pose[i] for i in range(3)]
    q1, q2 = target_pose[3:], pose[3:]
    norm = math.sqrt(sum(v*v for v in q1) * sum(v*v for v in q2))
    if norm < 1e-9:
        raise ValueError('Invalid zero quaternion')
    dot = min(1., abs(sum(a*b for a, b in zip(q1, q2))) / norm)
    return {
        'position_error_m': math.dist(target_pose[:3], pose[:3]),
        'displacement_m': math.sqrt(sum(v*v for v in moved)),
        'directed_progress_m': sum(a*b for a, b in zip(moved, delta)) / distance,
        'orientation_error_rad': 2 * math.acos(dot),
        'max_joint_change_rad': max(abs(a-b) for a, b in zip(joints, start_joints)),
        'target_distance_m': distance,
    }


def motion_passes(metrics, position_tolerance=.01):
    return (metrics['position_error_m'] <= position_tolerance
            and metrics['directed_progress_m'] >= .4 * metrics['target_distance_m']
            and metrics['displacement_m'] >= .4 * metrics['target_distance_m']
            and metrics['orientation_error_rad'] <= .15
            and metrics['max_joint_change_rad'] >= .005)


def pose_values(pose):
    return [pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--simulator', choices=('isaac', 'robocasa'), required=True)
    parser.add_argument('--start-delay', type=float, default=float(os.environ.get('GOLEM_ARM_START_DELAY', '2')),
                        help='Simulation seconds after all endpoints become ready')
    parser.add_argument('--sim-timeout', type=float, default=15., help='Maximum simulation seconds per action phase')
    parser.add_argument('--wall-timeout', type=float, default=float(os.environ.get('GOLEM_TEST_WALL_TIMEOUT', '300')))
    parser.add_argument('--offset', type=float, default=.02, help='Pelvis-frame forward displacement in metres')
    parser.add_argument('--position-tolerance', type=float, default=.01)
    args = parser.parse_args()
    if not 1 <= int(os.environ.get('ROS_DOMAIN_ID', '0')) <= 232:
        parser.error('Set ROS_DOMAIN_ID to a nonzero simulation domain')
    if not .01 <= args.offset <= .03:
        parser.error('--offset must be 0.01..0.03 metres')
    if not 0 < args.position_tolerance < args.offset:
        parser.error('--position-tolerance must be positive and smaller than --offset')
    if args.start_delay < 0 or min(args.sim_timeout, args.wall_timeout) <= 0:
        parser.error('Timeouts must be positive and start delay nonnegative')

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Clock
    from unitree_hg.msg import LowState
    from custom_ros_messages.action import FrameTask

    started = time.monotonic()
    deadline = started + args.wall_timeout
    report = {'test': 'frame_task_both_arms', 'simulator': args.simulator,
              'domain': int(os.environ['ROS_DOMAIN_ID']), 'passed': False,
              'coordinate_frame': 'pelvis', 'frame_names': ['left_wrist_yaw_link', 'right_wrist_yaw_link'],
              'offset_m': args.offset, 'position_tolerance_m': args.position_tolerance, 'phases': []}
    latest, counts = {}, {'left': 0, 'right': 0, 'joints': 0}
    active_handle = None
    rclpy.init()
    node = rclpy.create_node('golem_frame_task_physics_check')
    client = ActionClient(node, FrameTask, '/frame_task')
    subscriptions = []

    def clock(msg):
        now = msg.clock.sec + msg.clock.nanosec * 1e-9
        if 'clock' in latest and now < latest['clock']:
            raise AssertionError('Simulation clock reversed during arm test')
        latest['clock'] = now

    def ee(side, msg):
        if msg.header.frame_id != 'pelvis':
            raise AssertionError(f'{side} EE frame must be pelvis, got {msg.header.frame_id}')
        if not all(math.isfinite(v) for v in pose_values(msg.pose)):
            raise AssertionError(f'{side} EE pose contains nonfinite values')
        latest[side] = msg.pose
        counts[side] += 1

    def state(msg):
        joints = [float(m.q) for m in msg.motor_state[:27]]
        if len(joints) != 27 or not all(math.isfinite(v) for v in joints):
            raise AssertionError('LowState must contain 27 finite motor positions')
        if latest.get('tick') != int(msg.tick):
            latest['tick_changed_at'] = time.monotonic()
        latest['joints'] = joints
        latest['tick'] = int(msg.tick)
        counts['joints'] += 1

    subscriptions.append(node.create_subscription(Clock, '/clock', clock, qos_profile_sensor_data))
    subscriptions.append(node.create_subscription(LowState, '/lowstate', state, qos_profile_sensor_data))
    for side in ('left', 'right'):
        subscriptions.append(node.create_subscription(PoseStamped, f'/{side}_ee_pose', lambda msg, side=side: ee(side, msg), qos_profile_sensor_data))

    def spin():
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Wall timeout after {args.wall_timeout}s; received {counts}')
        rclpy.spin_once(node, timeout_sec=.02)

    try:
        print('FRAME_TASK_CHECK_READY', flush=True)
        while not (client.server_is_ready() and all(k in latest for k in ('left', 'right', 'joints', 'clock'))):
            spin()
        ready_time = latest['clock']
        while latest['clock'] - ready_time < args.start_delay:
            spin()
        original = {side: copy.deepcopy(latest[side]) for side in ('left', 'right')}
        for phase_name in ('forward', 'return'):
            start_poses = {side: pose_values(latest[side]) for side in ('left', 'right')}
            start_joints = latest['joints'].copy()
            targets = {side: copy.deepcopy(original[side]) for side in ('left', 'right')}
            if phase_name == 'forward':
                for pose in targets.values():
                    pose.position.x += args.offset
            goal = FrameTask.Goal()
            goal.plan = False
            goal.slow_mode = True
            goal.frame_names = report['frame_names']
            goal.frame_targets = [targets['left'], targets['right']]
            remaining = max(1., deadline - time.monotonic())
            goal.duration.sec = math.ceil(remaining)  # Server timeout is wall time.
            phase = {'name': phase_name, 'sim_start': latest['clock'], 'action_success': False,
                     'starts': start_poses, 'targets': {s: pose_values(p) for s, p in targets.items()}}
            report['phases'].append(phase)
            feedback_samples = []
            def feedback(message):
                feedback_samples.append({'linear_m': list(message.feedback.errors_linear),
                                         'angular_rad': list(message.feedback.errors_angular)})
            pending = client.send_goal_async(goal, feedback_callback=feedback)
            result = None
            result_counts = None
            result_tick = None
            stable_since = None
            while True:
                spin()
                elapsed = latest['clock'] - phase['sim_start']
                if elapsed > args.sim_timeout:
                    raise TimeoutError(f'{phase_name} exceeded {args.sim_timeout} simulation seconds')
                if active_handle is None and pending.done():
                    active_handle = pending.result()
                    if not active_handle.accepted:
                        raise AssertionError(f'{phase_name} goal rejected')
                    result = active_handle.get_result_async()
                if result is not None and result.done() and result_counts is None:
                    completed = result.result()
                    phase['action_status'] = completed.status
                    phase['action_success'] = bool(completed.result.success)
                    if completed.status != 4 or not completed.result.success:
                        raise AssertionError(f'{phase_name} action failed: status={completed.status}, success={completed.result.success}')
                    result_counts = counts.copy()
                    result_tick = latest['tick']
                metrics = {}
                for side, joint_slice in (('left', slice(13, 20)), ('right', slice(20, 27))):
                    metrics[side] = arm_metrics(start_poses[side], pose_values(targets[side]), pose_values(latest[side]),
                                               start_joints[joint_slice], latest['joints'][joint_slice])
                phase['metrics'] = metrics
                phase['sim_elapsed_s'] = elapsed
                phase['feedback_count'] = len(feedback_samples)
                phase['last_feedback'] = feedback_samples[-1] if feedback_samples else None
                fresh = (result_counts is not None
                         and latest['tick'] != result_tick
                         and time.monotonic() - latest.get('tick_changed_at', 0.) < 1.
                         and all(counts[k] > result_counts[k] for k in counts))
                passed = fresh and all(motion_passes(m, args.position_tolerance) for m in metrics.values())
                stable_since = latest['clock'] if passed and stable_since is None else stable_since if passed else None
                if stable_since is not None and latest['clock'] - stable_since >= .2:
                    phase['passed'] = True
                    print(json.dumps({'phase': phase_name, 'metrics': metrics}), flush=True)
                    active_handle = None
                    break
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if active_handle is not None and active_handle.accepted:
            future = active_handle.cancel_goal_async()
            until = time.monotonic() + 2.
            while not future.done() and time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=.05)
        report['wall_elapsed_s'] = time.monotonic() - started
        report['sample_counts'] = counts
        report['final_sim_time'] = latest.get('clock')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'passed': report['passed'], 'output': str(args.output), 'error': report.get('error')}), flush=True)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
