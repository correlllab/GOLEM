import os
import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from g1_control.core.controller.frame_controller import FrameController
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.utility.controller_config import (
    load_controller_config,
    init_channel_factory_for_commands,
    maybe_start_controller_logging,
)
from g1_control.utility.path_definition import (
    PROJECT_ROOT,
    URDF_PATH,
)


TRAJ_DIR = PROJECT_ROOT / 'data' / 'trajectory'
NUMBER_RE = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
FRAME_RE = re.compile(
    rf'"?([A-Za-z_][\w./-]*)"?\s*[:,]?\s*'
    rf'({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})\s+'
    rf'({NUMBER_RE})\s+({NUMBER_RE})\s+({NUMBER_RE})'
)


@dataclass
class Waypoint:
    config_name: str = ''
    frame_names: list = None
    frame_poses: list = None

    @property
    def is_config(self):
        return self.config_name != ''


def resolve_trajectory_path(traj):
    '''Resolve trajectory path from absolute, relative, or data/trajectory input'''
    path = Path(traj).expanduser()
    if path.is_absolute() or path.exists():
        return path

    data_path = TRAJ_DIR / path
    if data_path.exists() or path.suffix:
        return data_path

    txt_path = data_path.with_suffix('.txt')
    return txt_path if txt_path.exists() else data_path


def strip_comment(line):
    return line.split('#', 1)[0].strip()


def strip_quotes(text):
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def waypoint_body(line):
    '''Extract optional brace body from lines like: way point 1 { ... }'''
    if '{' not in line and '}' not in line:
        return line
    start = line.find('{')
    end = line.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('mismatched waypoint braces')
    return line[start + 1:end].strip()


def parse_waypoint_line(line, line_num):
    '''Parse one named config or one frame waypoint line'''
    line = strip_comment(line)
    if not line:
        return None

    config_name = strip_quotes(line)
    if config_name in NAMED_CONFIGS:
        return Waypoint(config_name=config_name)

    # parse one or more frame pose blocks from the line
    body = waypoint_body(line)
    matches = list(FRAME_RE.finditer(body))
    if not matches:
        raise ValueError(f'line {line_num}: expected named config or frame pose')

    leftover = FRAME_RE.sub('', body).replace(',', '').replace(';', '').strip()
    if leftover:
        raise ValueError(f'line {line_num}: could not parse "{leftover}"')

    frame_names = []
    frame_poses = []
    for match in matches:
        values = [float(match.group(i)) for i in range(2, 8)]
        values[3:] = np.deg2rad(values[3:]).tolist()
        frame_names.append(match.group(1))
        frame_poses.append(np.asarray(values, dtype=float))

    return Waypoint(frame_names=frame_names, frame_poses=frame_poses)


def load_trajectory(path):
    '''Load a line-based trajectory text file'''
    waypoints = []
    with open(path, 'r', encoding='utf-8') as traj_file:
        for line_num, line in enumerate(traj_file, start=1):
            waypoint = parse_waypoint_line(line, line_num)
            if waypoint is not None:
                waypoints.append(waypoint)
    if not waypoints:
        raise ValueError(f'no waypoints found in {path}')
    return waypoints


def init_frame_controller(config_name, visualize=True, real_robot=False):
    '''Initialize controller and DDS channel from config'''
    config = load_controller_config(config_name)
    # domain 0 needs --real-robot
    init_channel_factory_for_commands(
        config, real_robot,
        'waist and arm commands (an arm controller that moves the arms to home)',
    )
    controller = FrameController(
        URDF_PATH,
        init=True,
        visualize=visualize,
        config=config,
    )
    maybe_start_controller_logging(controller)
    return controller, config['controller']


def add_frame_waypoint_tasks(controller, waypoint):
    '''Replace active frame tasks with the waypoint tasks'''
    controller.clear_frame_tasks()
    for frame_name, pose in zip(waypoint.frame_names, waypoint.frame_poses):
        task_name = f'{frame_name}_task'
        controller.add_frame_task(task_name, frame_name, pose)


def max_frame_error(controller, frame_names):
    '''Return max linear and angular error across active frame tasks'''
    linear_errors = []
    angular_errors = []
    for frame_name in frame_names:
        error = controller.get_frame_task_error(f'{frame_name}_task')
        linear_errors.append(np.linalg.norm(error[:3]))
        angular_errors.append(np.linalg.norm(error[3:]))
    return max(linear_errors), max(angular_errors)


def apply_scaled_velocity_step(controller, vel, speed_scale):
    '''Scale the IK command before sending it to the robot'''
    # keep the motion slower without changing the underlying solver
    vel = controller._limit_joint_vel(speed_scale * vel)
    controller.ik_solver.integrate(vel)
    controller._apply_joint_position(controller.ik_solver.q)
    controller.update_robot_model()
    return vel


def run_control_loop(controller, controller_cfg, step_fn, error_fn, timeout):
    '''Run IK convergence followed by steady-state hold'''
    threshold_ik = controller_cfg['threshold_ik']
    threshold_joint = controller_cfg['threshold_joint']
    start_time = time.time()
    ik_converged = False
    steady_state_converged = False

    while time.time() - start_time < timeout:
        frame_start_time = time.time()

        # run IK until the command velocity becomes small
        if not ik_converged:
            vel = step_fn()
            vel_error = np.max(np.abs(vel))
            if vel_error < threshold_ik:
                ik_converged = True
                controller.init_steady_state()
                print('IK converged; entering steady-state hold')
        else:
            steady_state_converged = controller.steady_state_step(threshold_joint)
            steady_state_converged = steady_state_converged or error_fn()
            if steady_state_converged:
                print('Steady state reached')
                break

        time.sleep(max(0.0, controller.dt - (time.time() - frame_start_time)))

    return ik_converged and steady_state_converged


def execute_config_waypoint(controller, controller_cfg, waypoint, timeout,
                            speed_scale=1.0):
    '''Move controller to one named config waypoint'''
    q_reduced = NAMED_CONFIGS[waypoint.config_name]
    controller.update_ik_solver()

    def step_fn():
        vel = controller.ik_solver.goto_reduced_configuration(q_reduced)
        return apply_scaled_velocity_step(controller, vel, speed_scale)

    def error_fn():
        error = np.max(np.abs(controller.reduced_configuration_error))
        print(f'Configuration Error: {error:.4f}')
        return error < controller_cfg['threshold_joint']

    return run_control_loop(controller, controller_cfg, step_fn, error_fn, timeout)


def execute_frame_waypoint(controller, controller_cfg, waypoint, timeout,
                           com=False, speed_scale=1.0):
    '''Move controller to one multi-frame waypoint'''
    add_frame_waypoint_tasks(controller, waypoint)
    controller.update_ik_solver()

    def step_fn():
        vel = controller.ik_solver.ik_step_reduced(com=com)
        return apply_scaled_velocity_step(controller, vel, speed_scale)

    def error_fn():
        linear_error, angular_error = max_frame_error(controller, waypoint.frame_names)
        print(f'Linear Error: {linear_error:.4f}, Angular Error: {angular_error:.4f}')
        return (
            linear_error < controller_cfg['threshold_linear'] and
            angular_error < controller_cfg['threshold_angular']
        )

    return run_control_loop(controller, controller_cfg, step_fn, error_fn, timeout)


def run_trajectory(controller, controller_cfg, waypoints, duration=0.0,
                   pause=0.0, com=False, speed_scale=1.0):
    '''Execute every waypoint in order'''
    timeout = duration if duration > 0.0 else controller_cfg['timeout']
    for idx, waypoint in enumerate(waypoints, start=1):
        if waypoint.is_config:
            print(f'Waypoint {idx}: named config {waypoint.config_name}')
            success = execute_config_waypoint(
                controller,
                controller_cfg,
                waypoint,
                timeout,
                speed_scale=speed_scale,
            )
        else:
            print(f'Waypoint {idx}: frames {waypoint.frame_names}')
            success = execute_frame_waypoint(
                controller,
                controller_cfg,
                waypoint,
                timeout,
                com=com,
                speed_scale=speed_scale,
            )

        if not success:
            print(f'Waypoint {idx} failed; stopping trajectory')
            return False
        if pause > 0.0 and idx < len(waypoints):
            time.sleep(pause)
    return True


def main(traj, config_name='g1.yaml', duration=0.0,
         pause=0.0, com=False, speed_scale=0.4, visualize=True, real_robot=False):
    # load trajectory before commanding the robot
    traj_path = resolve_trajectory_path(traj)
    waypoints = load_trajectory(traj_path)
    print(f'Loaded {len(waypoints)} waypoints from {traj_path}')

    # initialize the same direct controller used by other examples
    controller, controller_cfg = init_frame_controller(
        config_name, visualize=visualize, real_robot=real_robot
    )
    try:
        # execute parsed waypoints one at a time
        run_trajectory(
            controller,
            controller_cfg,
            waypoints,
            duration=duration,
            pause=pause,
            com=com,
            speed_scale=speed_scale,
        )
    finally:
        print('Shutting down...')
        controller.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame task trajectory runner')
    parser.add_argument(
        '--traj',
        required=True,
        help='Trajectory file path or data/trajectory filename',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='g1.yaml',
        help='YAML file name under config/',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=0.0,
        help='Timeout per waypoint; 0 uses config',
    )
    parser.add_argument(
        '--pause',
        type=float,
        default=0.0,
        help='Pause between waypoints in seconds',
    )
    parser.add_argument(
        '--speed-scale',
        type=float,
        default=0.4,
        help='Scale IK motion commands for slower tracking',
    )
    parser.add_argument('--com', action='store_true', help='Use center of mass control')
    parser.add_argument('--no-viz', action='store_true', help='Disable controller visualizer')
    parser.add_argument('--real-robot', action='store_true',
                        help='Allow DDS domain 0 (the real robot); refused without it')
    args = parser.parse_args()

    main(
        args.traj,
        config_name=args.config,
        duration=args.duration,
        pause=args.pause,
        com=args.com,
        speed_scale=args.speed_scale,
        visualize=not args.no_viz,
        real_robot=args.real_robot,
    )
