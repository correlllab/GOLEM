"""Install golem_upper_ik: its Python package, launch files and each robot's configs."""
from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'golem_upper_ik'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
]
# Every robot's config/<robot>/ directory. The annotated example.*.yaml files document the
# schemas and stay in the source tree, so no loader or listing of the installed configs
# can take one for a real config. A --symlink-install build directory keeps dangling links
# to configs deleted from the source; they are skipped.
for robot_dir in sorted(glob('config/*/')):
    data_files.append(('share/' + package_name + '/' + os.path.dirname(robot_dir),
                       [path for path in glob(robot_dir + '*.yaml')
                        if os.path.exists(path)
                        and not os.path.basename(path).startswith('example.')]))

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxconway',
    maintainer_email='maxlconway@gmail.com',
    description='Upper-body (waist and arms) IK controller for the Unitree H1-2 and G1',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frame_task_server = golem_upper_ik.ros2.frame_task_server:main',
            'joint_state_publisher = golem_upper_ik.ros2.joint_state_publisher:main',
        ],
    },
)
