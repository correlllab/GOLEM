"""
Start the H1-2's two Magpie gripper nodes, /left/gripper/* and /right/gripper/*.

Both start at once. h12_real_drivers.launch.py and h12_safety_grippers.launch.py include
this file, so each gripper's port and finger calibration live in one place.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Start the left and right gripper nodes."""
    # gripper_node has no notion of a side (its name, topic and services are all
    # relative), so each side is a namespace. auto_detect_port must be False with
    # two grippers attached, or both race for the first /dev/ttyUSB* or ACM*. The
    # ports are /dev/serial/by-id/ paths, which follow each board's USB serial, so
    # they survive reboots and replugs and never swap sides.
    left_gripper = Node(
        package='magpie_control',
        executable='gripper_node',
        name='gripper_node',
        namespace='left',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'auto_detect_port': False,
            # OpenRB-150 serial ...FF101F14
            'port': ('/dev/serial/by-id/'
                     'usb-ROBOTIS_OpenRB-150_B0935A515157375037202020FF101F14-if00'),
            'use_eflesh': False,
            # deg, this gripper's calibration: min and max are the measured servo
            # angles at the open and closed stops. theta_90 (the parallel-jaw
            # reference) is estimated 26.6 deg (finger1) and 26.9 deg (finger2)
            # from the closed stop, the offsets measured on the right gripper.
            'finger1theta_min': 87.98,
            'finger1theta_max': 170.97,
            'finger1theta_90': 144.37,
            'finger2theta_min': 136.01,
            'finger2theta_max': 211.73,
            'finger2theta_90': 162.91,
        }],
    )

    right_gripper = Node(
        package='magpie_control',
        executable='gripper_node',
        name='gripper_node',
        namespace='right',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'auto_detect_port': False,
            # OpenRB-150 serial ...FF052729
            'port': ('/dev/serial/by-id/'
                     'usb-ROBOTIS_OpenRB-150_77E78BAE50323952582E3120FF052729-if00'),
            'use_eflesh': False,
            # deg, this gripper's calibration: min and max are the measured servo
            # angles at the open and closed stops; theta_90 is 26.6 deg (finger1)
            # and 26.9 deg (finger2) from the closed stop, as on the left.
            'finger1theta_min': 86.51,
            'finger1theta_max': 168.91,
            'finger1theta_90': 142.31,
            'finger2theta_min': 131.67,
            'finger2theta_max': 210.26,
            'finger2theta_90': 158.57,
        }],
    )

    return LaunchDescription([
        left_gripper,
        right_gripper,
    ])
