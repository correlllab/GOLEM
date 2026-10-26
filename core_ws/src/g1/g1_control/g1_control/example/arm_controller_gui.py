import time
import argparse
import numpy as np
import tkinter as tk

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
from g1_control.core.controller.arm_controller import ArmController
from g1_control.utility.controller_config import (
    load_controller_config,
    init_channel_factory_for_commands,
)
from g1_control.utility.path_definition import (
    URDF_PATH,
)

def main(config_name='g1.yaml', real_robot=False):
    config= load_controller_config(config_name)
    # domain 0 needs --real-robot
    init_channel_factory_for_commands(
        config, real_robot,
        'waist and arm commands (an arm controller that moves the arms to home)',
    )
    # example usage
    arm_controller = ArmController(URDF_PATH,
                                   init=True,
                                   visualize=True,
                                   config=config)
    arm_controller.left_ee_target_pose = [0.3, 0.2, 0.1, 0.0, 0.0, 0.0]
    arm_controller.right_ee_target_pose = [0.3, -0.2, 0.1, 0.0, 0.0, 0.0]

    root = tk.Tk()
    root.title('Arm Controller')
    root.geometry('600x400')

    # pack sliders side by side
    left_frame = tk.Frame(root)
    right_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10)
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    # left hand sliders
    slider_lx = tk.Scale(left_frame, label="Left X",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_ly = tk.Scale(left_frame, label="Left Y",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lz = tk.Scale(left_frame, label="Left Z",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lr = tk.Scale(left_frame, label="Left Roll",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lp = tk.Scale(left_frame, label="Left Pitch",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_lyaw = tk.Scale(left_frame, label="Left Yaw",
                           from_=-np.pi, to=np.pi, resolution=0.01,
                           orient=tk.HORIZONTAL, length=250)
    slider_lx.pack(in_=left_frame, pady=5)
    slider_ly.pack(in_=left_frame, pady=5)
    slider_lz.pack(in_=left_frame, pady=5)
    slider_lr.pack(in_=left_frame, pady=5)
    slider_lp.pack(in_=left_frame, pady=5)
    slider_lyaw.pack(in_=left_frame, pady=5)

    # right hand sliders
    slider_rx = tk.Scale(right_frame, label="Right X",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_ry = tk.Scale(right_frame, label="Right Y",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_rz = tk.Scale(right_frame, label="Right Z",
                         from_=-1.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_rr = tk.Scale(right_frame, label="Right Roll",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_rp = tk.Scale(right_frame, label="Right Pitch",
                         from_=-np.pi, to=np.pi, resolution=0.01,
                         orient=tk.HORIZONTAL, length=250)
    slider_ryaw = tk.Scale(right_frame, label="Right Yaw",
                           from_=-np.pi, to=np.pi, resolution=0.01,
                           orient=tk.HORIZONTAL, length=250)
    slider_rx.pack(in_=right_frame, pady=5)
    slider_ry.pack(in_=right_frame, pady=5)
    slider_rz.pack(in_=right_frame, pady=5)
    slider_rr.pack(in_=right_frame, pady=5)
    slider_rp.pack(in_=right_frame, pady=5)
    slider_ryaw.pack(in_=right_frame, pady=5)

    # left hand target initialization
    left_ee_position = arm_controller.left_ee_target_pose[:3]
    slider_lx.set(left_ee_position[0])
    slider_ly.set(left_ee_position[1])
    slider_lz.set(left_ee_position[2])
    left_ee_rpy = arm_controller.left_ee_target_rpy
    slider_lr.set(left_ee_rpy[0])
    slider_lp.set(left_ee_rpy[1])
    slider_lyaw.set(left_ee_rpy[2])

    # Right hand target initialization
    right_ee_position = arm_controller.right_ee_target_pose[:3]
    slider_rx.set(right_ee_position[0])
    slider_ry.set(right_ee_position[1])
    slider_rz.set(right_ee_position[2])
    right_ee_rpy = arm_controller.right_ee_target_rpy
    slider_rr.set(right_ee_rpy[0])
    slider_rp.set(right_ee_rpy[1])
    slider_ryaw.set(right_ee_rpy[2])

    root.update()

    try:
        while True:
            start_time = time.time()
            root.update()
            # update left hand target
            lx = slider_lx.get()
            ly = slider_ly.get()
            lz = slider_lz.get()
            lr = slider_lr.get()
            lp = slider_lp.get()
            lyaw = slider_lyaw.get()
            arm_controller.left_ee_target_pose = [lx, ly, lz, lr, lp, lyaw]

            # update right hand target
            rx = slider_rx.get()
            ry = slider_ry.get()
            rz = slider_rz.get()
            rr = slider_rr.get()
            rp = slider_rp.get()
            ryaw = slider_ryaw.get()
            arm_controller.right_ee_target_pose = [rx, ry, rz, rr, rp, ryaw]

            arm_controller.control_step_reduced()
            # arm_controller.sim_step_reduced()

            # print errors
            left_error_linear = np.linalg.norm(arm_controller.left_ee_error[:3])
            left_error_angular = np.linalg.norm(arm_controller.left_ee_error[3:])
            right_error_linear = np.linalg.norm(arm_controller.right_ee_error[:3])
            right_error_angular = np.linalg.norm(arm_controller.right_ee_error[3:])

            print(f'Left Error Linear: {left_error_linear:.4f}, '
                  f'Left Error Angular: {left_error_angular:.4f}, '
                  f'Right Error Linear: {right_error_linear:.4f}, '
                  f'Right Error Angular: {right_error_angular:.4f}')

            time.sleep(max(0.0, arm_controller.dt - (time.time() - start_time)))
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
