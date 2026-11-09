"""Install golem_bringup's launch files, RViz configs and slider_debugger.py."""
import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'golem_bringup'


def files(pattern):
    """Return the files matching pattern that exist."""
    # A --symlink-install build directory keeps dangling links to files deleted from the
    # source; they are skipped.
    return [path for path in sorted(glob(pattern)) if os.path.isfile(path)]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # One entry per subdirectory, installed at the same relative path:
        # ros2 launch finds a launch file by name anywhere under the share
        # directory, so each file is installed once.
        (os.path.join('share', package_name, 'launch', 'h12'), files('launch/h12/*.launch.py')),
        (os.path.join('share', package_name, 'launch', 'g1'), files('launch/g1/*.launch.py')),
        (os.path.join('share', package_name, 'rviz', 'h12'), files('rviz/h12/*.rviz')),
        (os.path.join('share', package_name, 'rviz', 'g1'), files('rviz/g1/*.rviz')),
    ],
    scripts=['scripts/slider_debugger.py'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='Launch files that start the H1-2 and the G1, on the real robot or in the sim.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
)
