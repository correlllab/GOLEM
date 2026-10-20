import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'lmg_deploy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + [
        # Each exported policy + its deploy config, installed under share/<pkg>/policies/<name>/.
        (os.path.join('share', package_name, 'policies', os.path.basename(d)), glob(os.path.join(d, '*')))
        for d in sorted(glob('policies/*')) if os.path.isdir(d)
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxconway',
    maintainer_email='maxlconway@gmail.com',
    description='Deploys the Locomanipulation_game MARL (legs + IK arms) policy on the H1-2',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'marl_deploy_node = lmg_deploy.marl_deploy_node:main',
        ],
    },
)
