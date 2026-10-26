from glob import glob
from setuptools import find_packages, setup

package_name = 'g1_safety_layer'

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/config', glob('config/*.yaml')),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'], # + requirements,
    data_files=data_files,
    zip_safe=True,
    maintainer='tonyzyt2000',
    maintainer_email='zhangyt2000@gmail.com',
    description='Safety layer relaying G1 low-level commands to rt/arm_sdk or rt/lowcmd',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'safety_node = g1_safety_layer.ros2.safety_node:main',
        ],
    },
)
