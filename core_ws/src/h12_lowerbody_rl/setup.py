import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'h12_lowerbody_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/policies', ['policies/UPSTREAM_LICENSE']),
        # Policy weights + configs, installed under share/<pkg>/policies/<name>/.
        (os.path.join('share', package_name, 'policies', 'walk'),
            glob('policies/walk/*')),
        (os.path.join('share', package_name, 'policies', 'fame'),
            glob('policies/fame/*')),
        (os.path.join('share', package_name, 'policies', 'almi'),
            glob('policies/almi/*')),
        (os.path.join('share', package_name, 'policies', 'fame_finetune'),
            glob('policies/fame_finetune/*')),
        (os.path.join('share', package_name, 'policies', 'almi_manip_stand'),
            glob('policies/almi_manip_stand/*')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxconway',
    maintainer_email='maxlconway@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'walking_node = h12_lowerbody_rl.scripts.walking_node:main',
            'fame_node = h12_lowerbody_rl.scripts.fame_node:main',
            'lowerbody_controller_node = '
            'h12_lowerbody_rl.scripts.lowerbody_controller_node:main',
            'sim2real_gap_monitor = '
            'h12_lowerbody_rl.scripts.sim2real_gap_monitor:main',
        ],
    },
)
