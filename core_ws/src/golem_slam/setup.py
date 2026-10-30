import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'golem_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.json')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='Shared SLAM and navigation for the H1-2 and G1.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'g1_body_to_pelvis = golem_slam.g1_body_to_pelvis:main',
            'level_cloud = golem_slam.level_cloud:main',
            'self_filter = golem_slam.self_filter:main',
            'level_odom = golem_slam.level_odom:main',
            'record_slam_bag = golem_slam.record_slam_bag:main',
        ],
    },
)
