from glob import glob

from setuptools import find_packages, setup

package_name = 'g1_dex3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niraj Pudasaini',
    maintainer_email='npudasai@nd.edu',
    description='Dex3-1 hand driver for the Unitree G1 (rt/dex3/* over DDS)',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dex3_hands = g1_dex3.hand_node:main',
        ],
    },
)
