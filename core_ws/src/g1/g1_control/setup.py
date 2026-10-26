import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'g1_control'

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob('launch/*.py')),
    ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ('share/' + package_name + '/config', glob('config/*.yaml')),
]

setup(
    name='g1_control',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    package_data={
        'g1_control.tests': ['configs/*.yaml'],
    },
    data_files=data_files,
    install_requires=['setuptools'],# + requirements,
    zip_safe=True,
    maintainer='tonyzyt2000',
    maintainer_email='zhangyt2000@gmail.com',
    description='Upper-body (waist + arms) IK controller for the Unitree G1',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_state_publisher = g1_control.ros2.joint_state_publisher:main',
            'dual_arm_server = g1_control.ros2.dual_arm_server:main',
            'dual_arm_client = g1_control.ros2.dual_arm_client:main',
            'frame_task_server = g1_control.ros2.frame_task_server:main',
            'frame_task_client = g1_control.ros2.frame_task_client:main',
        ],
    },
)
