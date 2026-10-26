import time
import argparse
import numpy as np

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
from g1_control.utility.joint_definition import BODY_JOINTS, UPPER_BODY_JOINTS, LOWER_BODY_JOINTS
from g1_control.utility.path_definition import URDF_PATH
from g1_control.utility.joint_definition import NUM_MOTOR

def print_joint_positions(robot_model, joint_groups):
    '''Print joint positions in an organized format by groups'''
    state = robot_model.state

    print('='*80)
    print(f'{"JOINT POSITIONS":^80}')
    print('='*80)

    for group_name, joint_names in joint_groups.items():
        print(f'\n{group_name.upper()}:')
        print('-' * 40)

        for joint_name in joint_names:
            try:
                joint_idx = BODY_JOINTS.index(joint_name)
                position = state['q'][joint_idx]
                velocity = state['dq'][joint_idx]
                torque = state['tau'][joint_idx]
                print(f'  {joint_name:26} | Pos: {position:8.4f} | Vel: {velocity:8.4f} | Tau: {torque:8.4f}')
            except ValueError:
                print(f'  {joint_name:26} | Joint not found in BODY_JOINTS')

    print('='*80)

def setup_low_damping(command_publisher, damping_value=0.5):
    '''Setup very low damping for all joints'''
    print(f'Setting damping to {damping_value} for all joints...')

    # set low damping for all joints
    for i in range(len(BODY_JOINTS)):
        command_publisher.kp[i] = 0.0  # no position control
        command_publisher.kd[i] = damping_value  # low damping
        command_publisher.q[i] = 0.0
        command_publisher.dq[i] = 0.0
        command_publisher.tau[i] = 0.0

def main(damping_value, update_rate, real_robot=False):
    # initialize channel; --damping publishes, so domain 0 needs --real-robot
    config = load_controller_config()
    if damping_value is None:
        init_channel_factory_guard(config)
    else:
        init_channel_factory_for_commands(
            config, real_robot,
            'kp 0 on every motor slot (through g1_safety_layer, so in arm_sdk mode the '
            'waist goes limp under the upper body while Unitree balances the legs)',
        )

    # initialize robot model
    robot_model = RobotModel(URDF_PATH)
    robot_model.init_subscriber()

    # wait for initial state
    print('Waiting for robot connection...')
    time.sleep(1.0)
    robot_model.update_kinematics()

    command_publisher = None
    if damping_value is not None:
        # initialize command publisher only if damping is provided
        command_publisher = LowCmdPublisher()
        # setup low damping
        setup_low_damping(command_publisher, damping_value)
        # enable all motors with low damping
        motor_ids = list(range(NUM_MOTOR))
        init_q = robot_model.state['q']
        command_publisher.enable_motors(motor_ids, init_q)
        command_publisher.start()
        print(f'Starting position monitoring with damping={damping_value}, update_rate={update_rate}Hz')
    else:
        print(f'Starting position monitoring (listener only), update_rate={update_rate}Hz')

    # define joint groups using predefined definitions from joint_definition
    joint_groups = {
        'lower_body': LOWER_BODY_JOINTS,
        'upper_body': UPPER_BODY_JOINTS
    }

    print('Press Ctrl+C to stop...')

    try:
        while True:
            robot_model.update_kinematics()
            print_joint_positions(robot_model, joint_groups)
            time.sleep(1.0 / update_rate)

    except KeyboardInterrupt:
        print('Stopping motor position checker...')

    finally:
        # shutdown
        if command_publisher is not None:
            command_publisher.shutdown()
        robot_model.shutdown()
        print('Motor position checker stopped.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motor position checker with optional damping')
    parser.add_argument('--damping', type=float, default=None,
                        help='Damping value to set for all joints (optional, listener only if not provided)')
    parser.add_argument('--rate', type=float, default=30.0,
                        help='Update rate in Hz (default: 30.0)')
    parser.add_argument('--real-robot', action='store_true',
                        help='Allow DDS domain 0 (the real robot); refused without it')
    args = parser.parse_args()

    main(args.damping, args.rate, real_robot=args.real_robot)
