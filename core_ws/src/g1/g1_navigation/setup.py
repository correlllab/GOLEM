from glob import glob

from setuptools import find_packages, setup

package_name = 'g1_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niraj Pudasaini',
    maintainer_email='npudasai@nd.edu',
    description='G1 navigation: FAST_LIO on the upside-down MID-360, slam_toolbox and Nav2 '
                'driving /cmd_vel',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'livox_to_pointcloud = g1_navigation.livox_to_pointcloud:main',
            'level_frame = g1_navigation.level_frame:main',
            'body_to_pelvis = g1_navigation.body_to_pelvis:main',
            'frontier_explore = g1_navigation.frontier_explore:main',
        ],
    },
)
