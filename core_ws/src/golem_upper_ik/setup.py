from glob import glob

from setuptools import find_packages, setup

package_name = 'golem_upper_ik'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/config/h12', glob('config/h12/*.yaml')),
    ('share/' + package_name + '/config/g1', glob('config/g1/*.yaml')),
]

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
        'console_scripts': [],
    },
)
