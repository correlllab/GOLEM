"""Install lmg_deploy's Python package, deploy configs, policies and marl_deploy_node."""
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
        # The annotated example.*.yaml configs document the keys and stay in the source tree.
        (os.path.join('share', package_name, 'config'),
            [path for path in glob('config/*.yaml') if not os.path.basename(path).startswith('example.')]),
    ] + [
        # Each policy directory (networks, deploy config, export.yaml) installs to share/lmg_deploy/policies/<name>/.
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
