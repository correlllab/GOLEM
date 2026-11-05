#!/usr/bin/env python3
"""
Drive the H1-2's frame_task_server and Magpie grippers from a slider window (Tk).

    ros2 run golem_bringup slider_debugger.py [--ros-args -p target_frames:="[...]"]

Twelve sliders (x, y, z in m and roll, pitch, yaw in rad for each arm) set /frame_task
targets for the GraspGenX gripper-base frames, left_graspgenx_frame and
right_graspgenx_frame, the frames the grasp skill commands, so a grasp pose can be
replayed by hand to watch the IK converge or contort. Two more sliders set each
gripper's gripper/set_position (0 closed, 1 open, scaled by gripper_max_mm). Send sends
all of them; each send cancels the arm goal in flight, and closing the window cancels
the last one.

The pose is the driven frame's position and xyz-Euler angles in pelvis. For the
graspgenx frames +z is the approach axis (roll, pitch, yaw 0 points it along pelvis +z);
1.5708 0 1.5708 matches the wrist's identity orientation.

ROS parameters: target_frames (the two frames driven, left then right),
left_gripper_service, right_gripper_service, gripper_max_mm (mm, the fully open width).
"""

import math
import tkinter as tk

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, Point, Quaternion
from builtin_interfaces.msg import Duration

from custom_ros_messages.action import FrameTask
from magpie_msgs.srv import SetGripperPosition


# The default target_frames: the frames the grasp skill commands.
TARGET_FRAMES = ('left_graspgenx_frame', 'right_graspgenx_frame')
SPIN_PERIOD_MS = 100  # ms between ROS spins from the Tk loop
GOAL_DURATION_SEC = 30  # s, each /frame_task goal's duration


class SliderDebugger(Node):
    """Send /frame_task goals and gripper positions from the sliders' values."""

    def __init__(self):
        """Declare the parameters and create the action and service clients."""
        super().__init__('slider_debugger')

        self.declare_parameter('left_gripper_service', '/left/gripper/set_position')
        self.declare_parameter('right_gripper_service', '/right/gripper/set_position')
        self.declare_parameter('gripper_max_mm', 85.0)
        self.declare_parameter('target_frames', list(TARGET_FRAMES))
        self.left_gripper_srv_name = self.get_parameter('left_gripper_service').value
        self.right_gripper_srv_name = self.get_parameter('right_gripper_service').value
        self.gripper_max_mm = float(self.get_parameter('gripper_max_mm').value)
        self.target_frames = list(self.get_parameter('target_frames').value)

        self.frame_task_cli = ActionClient(self, FrameTask, '/frame_task')
        self.left_gripper_cli = self.create_client(SetGripperPosition, self.left_gripper_srv_name)
        self.right_gripper_cli = self.create_client(SetGripperPosition,
                                                    self.right_gripper_srv_name)

        # m in pelvis, then rad (xyz-Euler) of the driven frame.
        self._initial_left = (0.3, 0.2, 0.1, 0.0, 0.0, 0.0)
        self._initial_right = (0.3, -0.2, 0.1, 0.0, 0.0, 0.0)

        self._goal_handle = None
        self._warned_left_grip = False
        self._warned_right_grip = False

    def _build_pose(self, x, y, z, r, p, yw) -> Pose:
        """Return the Pose of a position (m) and xyz-Euler angles (rad)."""
        qx, qy, qz, qw = Rotation.from_euler('xyz', [r, p, yw]).as_quat()
        return Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
        )

    def send_targets(self, left_xyzrpy, right_xyzrpy):
        """Cancel the goal in flight and send both frames' targets, if the server is up."""
        if not self.frame_task_cli.server_is_ready():
            return
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._goal_handle = None

        goal = FrameTask.Goal()
        goal.frame_names = list(self.target_frames)
        goal.frame_targets = [
            self._build_pose(*left_xyzrpy),
            self._build_pose(*right_xyzrpy),
        ]
        goal.duration = Duration(sec=GOAL_DURATION_SEC, nanosec=0)

        send_future = self.frame_task_cli.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        """Keep the accepted goal's handle so the next send can cancel it."""
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'frame_task send failed: {exc}')
            return
        if handle is not None and handle.accepted:
            self._goal_handle = handle

    def send_gripper(self, side: str, slider_val: float):
        """Set one gripper's opening (0 closed, 1 open); warn once if its service is down."""
        cli = self.left_gripper_cli if side == 'left' else self.right_gripper_cli
        warned_attr = '_warned_left_grip' if side == 'left' else '_warned_right_grip'
        srv_name = self.left_gripper_srv_name if side == 'left' else self.right_gripper_srv_name
        if not cli.service_is_ready():
            if not getattr(self, warned_attr):
                self.get_logger().warn(
                    f'{srv_name} not available; gripper commands will be skipped')
                setattr(self, warned_attr, True)
            return
        req = SetGripperPosition.Request()
        req.position = float(slider_val) * self.gripper_max_mm
        req.speed = 1.0
        cli.call_async(req)

    def cancel_active_goal(self):
        """Cancel the /frame_task goal in flight, if any."""
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._goal_handle = None


def _build_wrist_sliders(parent, label_prefix: str, init_xyzrpy):
    """Return {axis: Scale} of one frame's six sliders, set to init_xyzrpy."""
    sliders = {}
    for name, lo, hi in (('X', -1.0, 1.0), ('Y', -1.0, 1.0), ('Z', -1.0, 1.0)):
        s = tk.Scale(parent, label=f'{label_prefix} {name}',
                     from_=lo, to=hi, resolution=0.01,
                     orient=tk.HORIZONTAL, length=260)
        s.pack(pady=2)
        sliders[name] = s
    for name in ('Roll', 'Pitch', 'Yaw'):
        s = tk.Scale(parent, label=f'{label_prefix} {name}',
                     from_=-math.pi, to=math.pi, resolution=0.01,
                     orient=tk.HORIZONTAL, length=260)
        s.pack(pady=2)
        sliders[name] = s
    sliders['X'].set(init_xyzrpy[0])
    sliders['Y'].set(init_xyzrpy[1])
    sliders['Z'].set(init_xyzrpy[2])
    sliders['Roll'].set(init_xyzrpy[3])
    sliders['Pitch'].set(init_xyzrpy[4])
    sliders['Yaw'].set(init_xyzrpy[5])
    return sliders


def _read_wrist(sliders):
    """Return the six sliders' values as (x, y, z, roll, pitch, yaw)."""
    return (sliders['X'].get(), sliders['Y'].get(), sliders['Z'].get(),
            sliders['Roll'].get(), sliders['Pitch'].get(), sliders['Yaw'].get())


def main():
    """Open the slider window and spin the node from the Tk loop until it closes."""
    rclpy.init()
    node = SliderDebugger()

    root = tk.Tk()
    root.title('H1 Slider Debugger')

    left_frame = tk.LabelFrame(root, text=f'{node.target_frames[0]} + gripper',
                               padx=8, pady=8)
    right_frame = tk.LabelFrame(root, text=f'{node.target_frames[1]} + gripper',
                                padx=8, pady=8)
    left_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor='n')
    right_frame.pack(side=tk.RIGHT, padx=10, pady=10, anchor='n')

    left_sliders = _build_wrist_sliders(left_frame, 'Left', node._initial_left)
    right_sliders = _build_wrist_sliders(right_frame, 'Right', node._initial_right)

    left_grip = tk.Scale(left_frame, label='Left Gripper (0=closed, 1=open)',
                         from_=0.0, to=1.0, resolution=0.01,
                         orient=tk.HORIZONTAL, length=260)
    left_grip.set(0.5)
    left_grip.pack(pady=2)
    right_grip = tk.Scale(right_frame, label='Right Gripper (0=closed, 1=open)',
                          from_=0.0, to=1.0, resolution=0.01,
                          orient=tk.HORIZONTAL, length=260)
    right_grip.set(0.5)
    right_grip.pack(pady=2)

    def on_send():
        """Send the arm targets and both gripper positions."""
        if not rclpy.ok():
            return
        left_xyzrpy = _read_wrist(left_sliders)
        right_xyzrpy = _read_wrist(right_sliders)
        node.send_targets(left_xyzrpy, right_xyzrpy)
        node.send_gripper('left', left_grip.get())
        node.send_gripper('right', right_grip.get())

    send_button = tk.Button(root, text='Send', command=on_send,
                            width=20, height=2)
    send_button.pack(side=tk.BOTTOM, pady=10)

    def on_close():
        """Cancel the arm goal and close the window."""
        node.cancel_active_goal()
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol('WM_DELETE_WINDOW', on_close)

    def spin_tick():
        """Spin the node once and schedule the next spin."""
        if rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            root.after(SPIN_PERIOD_MS, spin_tick)

    root.after(SPIN_PERIOD_MS, spin_tick)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        node.cancel_active_goal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
