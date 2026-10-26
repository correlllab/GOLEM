import time
import argparse
import numpy as np
import pinocchio as pin

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from g1_control.core.controller.frame_controller import FrameController
from g1_control.utility.controller_config import (
    load_controller_config,
    init_channel_factory_for_commands,
    maybe_start_controller_logging,
)
from g1_control.utility.named_config import NAMED_CONFIGS
from g1_control.utility.path_definition import (
    URDF_PATH,
)

def input_keyword_or_frame_task():
    '''
    Ask user for keyword command or manual frame task input
    Returns (keyword, frame_name, pose) tuple
    If keyword is provided, frame_name and pose will be None
    If no keyword, frame_name and pose will be provided
    '''
    print(f'Available keywords: {list(NAMED_CONFIGS.keys())}')

    choice = input('Enter keyword (or press Enter for manual frame task): ').strip()

    if choice:
        # user entered a keyword
        return choice, None, None
    else:
        # user wants manual frame task input
        frame_name, pose = input_frame_task()
        return '', frame_name, pose

def input_frame_task():
    '''Get frame name and pose from user'''
    frame_name = input('Enter frame name: ')
    while True:
        input_pose = input('Enter x y z roll pitch yaw (separated by space): ')
        parts = input_pose.strip().split()

        if len(parts) != 6:
            print('Invalid input. Please enter exactly 6 values.')
            continue
        try:
            values = [float(val) for val in parts]
            # degrees to radians for roll, pitch, yaw
            values[3:] = np.deg2rad(values[3:])
            return frame_name, np.array(values)
        except ValueError:
            print('Invalid input. Make sure all 6 values are numeric.')
            continue

def main(com=False,
         config_name='g1.yaml',
         real_robot=False):
    config= load_controller_config(config_name)
    controller_cfg = config['controller']
    timeout = controller_cfg['timeout']
    threshold_ik = controller_cfg['threshold_ik']
    threshold_joint = controller_cfg['threshold_joint']
    threshold_linear = controller_cfg['threshold_linear']
    threshold_angular = controller_cfg['threshold_angular']
    # domain 0 needs --real-robot
    init_channel_factory_for_commands(
        config, real_robot,
        'waist and arm commands (an arm controller that moves the arms to home)',
    )
    # initialize upper task controller
    frame_controller = FrameController(URDF_PATH,
                                       init=True,
                                       visualize=True,
                                       config=config)
    maybe_start_controller_logging(frame_controller)

    try:
        while True:
            # get keyword or frame task input
            keyword, frame_name, pose = input_keyword_or_frame_task()

            # handle keyword configuration
            if keyword in NAMED_CONFIGS:
                print(f'Going to named configuration: {keyword}')
                q_reduced = NAMED_CONFIGS[keyword]
                # use goto as step function
                step_function = lambda: frame_controller.goto_reduced_configuration(q_reduced)
                # step_function = lambda: frame_controller.sim_goto_reduced_configuration(q_reduced)
            elif keyword != '':
                print(f'Unknown keyword: {keyword}')
                continue
            else:
                # manual frame task input
                print('Going to manual frame task')
                task_name = f'{frame_name}_task'
                # add frame task
                frame_controller.clear_frame_tasks()
                frame_controller.add_frame_task(task_name, frame_name, pose)
                step_function = lambda: frame_controller.control_step_reduced(com=com)
                # step_function = lambda: frame_controller.sim_step_reduced(com=com)

            # update ik solver with current state
            frame_controller.update_ik_solver()

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
                            np.abs(frame_controller.reduced_configuration_error)
                        )
                        print(
                            f'Configuration Error: {error:.4f}, '
                            f'IK Velocity: {vel_error:.4f}'
                        )
                    # print error for manual frame task
                    else:
                        error = frame_controller.get_frame_task_error(task_name)
                        linear_error = np.linalg.norm(error[:3])
                        angular_error = np.linalg.norm(error[3:])

                        print(
                            f'Linear Error: {linear_error:.4f}, '
                            f'Angular Error: {angular_error:.4f}, '
                            f'IK Velocity: {vel_error:.4f}'
                        )

                    if vel_error < threshold_ik:
                        ik_converged = True
                        frame_controller.init_steady_state()
                        print('IK converged; entering steady-state hold')
                else:
                    steady_state_converged = frame_controller.steady_state_step(
                        threshold_joint
                    )
                    if keyword in NAMED_CONFIGS:
                        error = np.max(
                            np.abs(frame_controller.reduced_configuration_error)
                        )
                        print(f'Configuration Error: {error:.4f}')
                        steady_state_converged = (
                            steady_state_converged or
                            error < threshold_joint
                        )
                    else:
                        error = frame_controller.get_frame_task_error(task_name)
                        linear_error = np.linalg.norm(error[:3])
                        angular_error = np.linalg.norm(error[3:])
                        print(
                            f'Linear Error: {linear_error:.4f}, '
                            f'Angular Error: {angular_error:.4f}'
                        )
                        steady_state_converged = steady_state_converged or (
                            linear_error < threshold_linear and
                            angular_error < threshold_angular
                        )

                    if steady_state_converged:
                        print('Steady state reached!')
                        break

                time.sleep(max(
                    0.0,
                    frame_controller.dt - (time.time() - frame_start_time)
                ))

            if not ik_converged:
                print('Timed out before IK convergence')
            elif not steady_state_converged:
                print('Timed out during steady-state hold')
            else:
                print('Target reached!')

            if keyword in NAMED_CONFIGS:
                error = frame_controller.reduced_configuration_error
                print(f'Final Configuration Error: {np.max(np.abs(error)):.4f}')
            else:
                error = frame_controller.get_frame_task_error(task_name)
                linear_error = np.linalg.norm(error[:3])
                angular_error = np.linalg.norm(error[3:])
                print(f'Final Linear Error: {linear_error:.4f}, '
                    f'Final Angular Error: {angular_error:.4f}')

            input('Press any key to continue...') # flush the input buffer
            cont = input('Do you want to send another goal? (y/n): ').lower()
            if cont != 'y':
                break
    except Exception as e:
        print(f'Exception occurred: {e}')
    finally:
        print('Shutting down...')
        frame_controller.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frame Controller Goto')
    parser.add_argument('--config', type=str, default='g1.yaml', help='YAML file name under config/')
    parser.add_argument('--com', action='store_true', help='Use center of mass control')
    parser.add_argument('--real-robot', action='store_true',
                        help='Allow DDS domain 0 (the real robot); refused without it')
    args = parser.parse_args()
    main(com=args.com, config_name=args.config, real_robot=args.real_robot)
