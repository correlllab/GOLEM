from glob import glob

from setuptools import find_packages, setup

package_name = 'g1_locomotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Niraj Pudasaini',
    maintainer_email='npudasai@nd.edu',
    description=("HOMIE's walking policy on the Unitree G1's legs, "
                 "through g1_safety_layer's split mode"),
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'locomotion_node = g1_locomotion.locomotion_node:main',
        ],
    },
)
