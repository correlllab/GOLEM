import time
import argparse
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from g1_control.core.controller.arm_controller import ArmController
from g1_control.utility.controller_config import (
    load_controller_config,
    init_channel_factory_for_commands,
    maybe_start_controller_logging,
)
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.utility.path_definition import (
    URDF_PATH,
)

def input_keyword_or_poses():
    '''
    Ask user for keyword command or manual pose input
    Returns (keyword, left_pose, right_pose) tuple
    If keyword is provided, poses will be None
    If no keyword, poses will be provided
    '''
    print(f'Available keywords: {list(NAMED_CONFIGS.keys())}')

    choice = input('Enter keyword (or press Enter for manual poses): ').strip()

    if choice:
        # user entered a keyword
        return choice, None, None
    else:
        # user wants manual pose input
        left_pose = input_pose('left')
        right_pose = input_pose('right')
        return '', left_pose, right_pose

def input_pose(side):
    print(f'{side.capitalize()} end-effector pose...')

    while True:
        input_pose = input('Enter x y z roll pitch yaw (separated by space): ')
        parts = input_pose.strip().split()

        if len(parts) != 6:
            print('Invalid input. Please enter exactly 6 values.')
            continue
        try:
            values = [float(val) for val in parts]
            # degrees to radian
            values[3:] = np.deg2rad(values[3:])
            return values
        except ValueError:
            print('Invalid input. Make sure all 6 values are numeric.')
            continue


def main(config_name='g1.yaml', real_robot=False):
    # parse threshold and timeout from config
    config= load_controller_config(config_name)
    controller_cfg = config['controller']
    timeout = controller_cfg['timeout']
    threshold_ik = controller_cfg['threshold_ik']
    threshold_joint = controller_cfg['threshold_joint']
    threshold_linear = controller_cfg['threshold_linear']
    threshold_angular = controller_cfg['threshold_angular']

    # initialize channel factory; domain 0 needs --real-robot
    init_channel_factory_for_commands(
        config, real_robot,
        'waist and arm commands (an arm controller that moves the arms to home)',
    )
    # initialize arm controller
    arm_controller = ArmController(URDF_PATH,
                                   init=True,
                                   visualize=True,
                                   config=config)
    maybe_start_controller_logging(arm_controller)

    try:
        while True:
            # get keyword or target poses
            keyword, left_pose, right_pose = input_keyword_or_poses()

            # handle keyword configuration
            if keyword in NAMED_CONFIGS:
                print(f'Going to named configuration: {keyword}')
                q_reduced = NAMED_CONFIGS[keyword]
                # use goto as step function
                step_function = lambda: arm_controller.goto_reduced_configuration(q_reduced)
                # step_function = lambda: arm_controller.sim_goto_reduced_configuration(q_reduced)
            elif keyword != '':
                print(f'Unknown keyword: {keyword}')
                continue
            else:
                # manual pose input
                print('Going to target end-effector poses')
                # set target poses
                arm_controller.left_ee_target_pose = left_pose
                arm_controller.right_ee_target_pose = right_pose
                step_function = lambda: arm_controller.control_step_reduced()
                # step_function = lambda: arm_controller.sim_step_reduced()

            # update ik solver with current state
            arm_controller.update_ik_solver()

            # main loop
            start_time = time.time()
            ik_converged = False
            steady_state_converged = False
            while time.time() - start_time < timeout:
                frame_start_time = time.time()
                if not ik_converged:
                    # run IK until the command stops changing
                    vel = step_function()
                    vel_error = np.max(np.abs(vel))

                    # print error for config task
                    if keyword in NAMED_CONFIGS:
                        error = np.max(
                            np.abs(arm_controller.reduced_configuration_error)
                        )
                        print(
                            f'Configuration Error: {error:.4f}, '
                            f'IK Velocity: {vel_error:.4f}'
                        )
                    # print errors for end-effector pose task
                    else:
                        left_error_linear = np.linalg.norm(
                            arm_controller.left_ee_error[:3]
                        )
                        left_error_angular = np.linalg.norm(
                            arm_controller.left_ee_error[3:]
                        )
                        right_error_linear = np.linalg.norm(
                            arm_controller.right_ee_error[:3]
                        )
                        right_error_angular = np.linalg.norm(
                            arm_controller.right_ee_error[3:]
                        )

                        print(f'Left Error Linear: {left_error_linear:.4f}, '
                              f'Left Error Angular: {left_error_angular:.4f}, '
                              f'Right Error Linear: {right_error_linear:.4f}, '
                              f'Right Error Angular: {right_error_angular:.4f}, '
                              f'IK Velocity: {vel_error:.4f}')

                    if vel_error < threshold_ik:
                        ik_converged = True
                        arm_controller.init_steady_state()
                        print('IK converged; entering steady-state hold')
                else:
                    steady_state_converged = arm_controller.steady_state_step(
                        threshold_joint
                    )
                    if keyword in NAMED_CONFIGS:
                        error = np.max(np.abs(arm_controller.reduced_configuration_error))
                        print(f'Configuration Error: {error:.4f}')
                        steady_state_converged = steady_state_converged or error < threshold_joint
                    else:
                        left_error_linear = np.linalg.norm(arm_controller.left_ee_error[:3])
                        left_error_angular = np.linalg.norm(arm_controller.left_ee_error[3:])
                        right_error_linear = np.linalg.norm(arm_controller.right_ee_error[:3])
                        right_error_angular = np.linalg.norm(arm_controller.right_ee_error[3:])
                        print(f'Left Error Linear: {left_error_linear:.4f}, '
                              f'Left Error Angular: {left_error_angular:.4f}, '
                              f'Right Error Linear: {right_error_linear:.4f}, '
                              f'Right Error Angular: {right_error_angular:.4f}')
                        steady_state_converged = steady_state_converged or (
                            left_error_linear < threshold_linear and
                            left_error_angular < threshold_angular and
                            right_error_linear < threshold_linear and
                            right_error_angular < threshold_angular
                        )

                    if steady_state_converged:
                        print('Steady state reached!')
                        break

                time.sleep(max(0.0, arm_controller.dt - (time.time() - frame_start_time)))

            if not ik_converged:
                print('Timed out before IK convergence')
            elif not steady_state_converged:
                print('Timed out during steady-state hold')
            else:
                print('Target reached!')

            if keyword in NAMED_CONFIGS:
                error = arm_controller.reduced_configuration_error
                print(f'Final Configuration Error: {np.max(np.abs(error)):.4f}')
            else:
                left_error_linear = np.linalg.norm(arm_controller.left_ee_error[:3])
                left_error_angular = np.linalg.norm(arm_controller.left_ee_error[3:])
                right_error_linear = np.linalg.norm(arm_controller.right_ee_error[:3])
                right_error_angular = np.linalg.norm(arm_controller.right_ee_error[3:])
                print(f'Final Left Error Linear: {left_error_linear:.4f}, '
                    f'Final Left Error Angular: {left_error_angular:.4f}, '
                    f'Final Right Error Linear: {right_error_linear:.4f}, '
                    f'Final Right Error Angular: {right_error_angular:.4f}')

            input('Press any key to continue...') # flush the input buffer
            cont = input('Do you want to send another goal? (y/n): ').lower()
            if cont != 'y':
                break
    except Exception as e:
        print(f'Exception occurred: {e}')
    finally:
        print('Shutting down...')
        arm_controller.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arm Controller Goto')
    parser.add_argument('--config', type=str, default='g1.yaml', help='YAML file name under config/')
    parser.add_argument('--real-robot', action='store_true',
                        help='Allow DDS domain 0 (the real robot); refused without it')
    args = parser.parse_args()
    main(config_name=args.config, real_robot=args.real_robot)
