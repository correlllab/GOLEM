import time
import curses
import argparse

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from g1_control.core.robot_model import RobotModel
from g1_control.core.channel_interface import LowCmdPublisher
from g1_control.utility.controller_config import (
    load_controller_config,
    init_channel_factory_guard,
    init_channel_factory_for_commands,
)
from g1_control.utility.path_definition import URDF_PATH
from g1_control.utility.joint_definition import NUM_MOTOR
from g1_control.utility.joint_definition import BODY_JOINTS, LOWER_BODY_JOINTS, UPPER_BODY_JOINTS
from g1_control.utility.joint_limits import JOINT_POSITION_LIMITS


DEFAULT_WARN_THRESHOLD = 0.8
DEFAULT_CRIT_THRESHOLD = 0.9


def setup_low_damping(command_publisher, damping_value=0.5):
    '''Setup very low damping for all joints'''
    print(f'Setting damping to {damping_value} for all joints...')

    for i in range(len(BODY_JOINTS)):
        command_publisher.kp[i] = 0.0
        command_publisher.kd[i] = damping_value
        command_publisher.q[i] = 0.0
        command_publisher.dq[i] = 0.0
        command_publisher.tau[i] = 0.0


def build_joint_limits():
    '''Build a list of joint limits aligned to BODY_JOINTS'''
    limits = []
    for idx in range(len(BODY_JOINTS)):
        if idx < len(JOINT_POSITION_LIMITS):
            limit = JOINT_POSITION_LIMITS[idx]
            limits.append((limit['low'], limit['high']))
        else:
            limits.append((None, None))
    return limits


def compute_limit_metrics(position, low, high):
    '''Compute position ratio and limit proximity metric'''
    if low is None or high is None:
        return 0.5, 0.0, False

    span = high - low
    if span <= 0.0:
        return 0.5, 0.0, False

    ratio = (position - low) / span
    ratio = min(max(ratio, 0.0), 1.0)

    distance_low = position - low
    distance_high = high - position
    nearest = min(distance_low, distance_high)
    if nearest < 0.0:
        return ratio, 1.0, True

    half_range = span / 2.0
    closeness = 1.0 - min(nearest / half_range, 1.0)
    return ratio, closeness, False


def format_pos_bar(ratio, closeness, is_outside, width, warn_threshold, crit_threshold):
    ratio_idx = int(round(ratio * (width - 1)))
    ratio_idx = min(max(ratio_idx, 0), width - 1)
    chars = ['-'] * width

    for i in range(width):
        if i == ratio_idx:
            chars[i] = '|'

    if is_outside:
        level_char = '!'
    elif closeness >= crit_threshold:
        level_char = '#'
    elif closeness >= warn_threshold:
        level_char = '='
    else:
        level_char = '-'

    edge_width = max(1, width // 8)
    for i in range(edge_width):
        if chars[i] == '-':
            chars[i] = level_char
        j = width - 1 - i
        if chars[j] == '-':
            chars[j] = level_char

    return '[' + ''.join(chars) + ']'


def pick_bar_width(columns):
    if columns < 100:
        return 16
    if columns < 120:
        return 24
    if columns < 150:
        return 36
    if columns < 180:
        return 52
    if columns < 220:
        return 68
    return 88


def format_limit_status(closeness, is_outside, warn_threshold, crit_threshold):
    if is_outside:
        return 'OUT'
    if closeness >= crit_threshold:
        return 'CRIT'
    if closeness >= warn_threshold:
        return 'WARN'
    return 'OK'


def _safe_addstr(screen, row, col, text, attr=0):
    max_y, max_x = screen.getmaxyx()
    if row < 0 or row >= max_y or col >= max_x:
        return

    if col < 0:
        text = text[-col:]
        col = 0

    max_len = max_x - col - 1
    if max_len <= 0:
        return

    if len(text) > max_len:
        text = text[:max_len]

    try:
        screen.addstr(row, col, text, attr)
    except curses.error:
        pass


def render_tui(screen, robot_model, joint_groups, limits, update_rate, damping_value, warn_threshold, crit_threshold):
    state = robot_model.state
    rows, columns = screen.getmaxyx()
    bar_width = pick_bar_width(columns)

    title = f'MOTOR VIEWER TUI | rate {update_rate:.1f}Hz'
    if damping_value is None:
        title += ' | listener only'
    else:
        title += f' | damping {damping_value:.3f}'

    screen.erase()
    line = '=' * min(columns - 1, 80)
    dash = '-' * min(columns - 1, 40)
    row = 0

    _safe_addstr(screen, row, 0, line)
    row += 1
    _safe_addstr(screen, row, 0, f'{title:^{len(line)}}')
    row += 1
    _safe_addstr(screen, row, 0, line)
    row += 2

    for group_name, joint_names in joint_groups.items():
        _safe_addstr(screen, row, 0, f'{group_name.upper()}:')
        row += 1
        _safe_addstr(screen, row, 0, dash)
        row += 1
        header = f'  {"Joint Name":26} | {"Pos":<{bar_width + 11}} | {"Vel":>8} | {"Tau":>8} | Stat'
        _safe_addstr(screen, row, 0, header)
        row += 1

        for joint_name in joint_names:
            if row >= rows - 3:
                _safe_addstr(screen, rows - 2, 0, 'Terminal too small to render all joints.')
                _safe_addstr(screen, rows - 1, 0, 'Resize terminal or lower font size. Press Ctrl+C to exit.')
                screen.refresh()
                return

            try:
                joint_idx = BODY_JOINTS.index(joint_name)
                position = state['q'][joint_idx]
                velocity = state['dq'][joint_idx]
                torque = state['tau'][joint_idx]
                low, high = limits[joint_idx]
                ratio, closeness, is_outside = compute_limit_metrics(position, low, high)
                status = format_limit_status(closeness, is_outside, warn_threshold, crit_threshold)
                bar = format_pos_bar(ratio, closeness, is_outside, bar_width, warn_threshold, crit_threshold)
                line_text = (
                    f'  {joint_name:26} | {bar} {position:8.4f} | '
                    f'{velocity:8.4f} | {torque:8.4f} | {status:4}'
                )
                _safe_addstr(screen, row, 0, line_text)
                row += 1
            except ValueError:
                _safe_addstr(screen, row, 0, f'  {joint_name:26} | Joint not found in BODY_JOINTS')
                row += 1

        row += 1

    _safe_addstr(screen, rows - 1, 0, 'Ctrl+C to stop.')
    screen.refresh()


def run_tui(robot_model, joint_groups, limits, update_rate, damping_value, warn_threshold, crit_threshold):
    def curses_main(screen):
        curses.curs_set(0)
        screen.nodelay(True)

        while True:
            robot_model.update_kinematics()
            render_tui(
                screen,
                robot_model,
                joint_groups,
                limits,
                update_rate,
                damping_value,
                warn_threshold,
                crit_threshold
            )
            time.sleep(1.0 / update_rate)

    curses.wrapper(curses_main)


def main(damping_value, update_rate, warn_threshold, crit_threshold, real_robot=False):
    # --damping publishes, so domain 0 needs --real-robot
    config = load_controller_config()
    if damping_value is None:
        init_channel_factory_guard(config)
    else:
        init_channel_factory_for_commands(
            config, real_robot,
            'kp 0 on every motor slot (through g1_safety_layer, so in arm_sdk mode the '
            'waist goes limp under the upper body while Unitree balances the legs)',
        )

    robot_model = RobotModel(URDF_PATH)
    robot_model.init_subscriber()

    print('Waiting for robot connection...')
    time.sleep(1.0)
    robot_model.update_kinematics()

    command_publisher = None
    if damping_value is not None:
        command_publisher = LowCmdPublisher()
        setup_low_damping(command_publisher, damping_value)
        motor_ids = list(range(NUM_MOTOR))
        init_q = robot_model.state['q']
        command_publisher.enable_motors(motor_ids, init_q)
        command_publisher.start()
        print(f'Starting TUI with damping={damping_value}, update_rate={update_rate}Hz')
    else:
        print(f'Starting TUI (listener only), update_rate={update_rate}Hz')

    joint_groups = {
        'lower_body': LOWER_BODY_JOINTS,
        'upper_body': UPPER_BODY_JOINTS
    }
    limits = build_joint_limits()

    print('Launching TUI... Press Ctrl+C to stop.')
    time.sleep(0.3)

    try:
        run_tui(
            robot_model,
            joint_groups,
            limits,
            update_rate,
            damping_value,
            warn_threshold,
            crit_threshold
        )
    except KeyboardInterrupt:
        print('Stopping motor viewer...')
    finally:
        if command_publisher is not None:
            command_publisher.shutdown()
        robot_model.shutdown()
        print('Motor viewer stopped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motor viewer TUI with joint limit proximity')
    parser.add_argument('--damping', type=float, default=None,
                        help='Damping value to set for all joints (optional, listener only if not provided)')
    parser.add_argument('--rate', type=float, default=30.0,
                        help='Update rate in Hz (default: 30.0)')
    parser.add_argument('--warn-threshold', type=float, default=DEFAULT_WARN_THRESHOLD,
                        help='Warn threshold for joint limit proximity (default: 0.7)')
    parser.add_argument('--crit-threshold', type=float, default=DEFAULT_CRIT_THRESHOLD,
                        help='Critical threshold for joint limit proximity (default: 0.9)')
    parser.add_argument('--real-robot', action='store_true',
                        help='Allow DDS domain 0 (the real robot); refused without it')
    args = parser.parse_args()

    main(args.damping, args.rate, args.warn_threshold, args.crit_threshold,
         real_robot=args.real_robot)
