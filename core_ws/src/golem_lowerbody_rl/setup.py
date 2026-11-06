import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'golem_lowerbody_rl'


def files(pattern):
    """Return the files matching pattern that exist, sorted."""
    # A --symlink-install build directory keeps dangling links to files deleted from the
    # source; they are skipped.
    return [path for path in sorted(glob(pattern)) if os.path.isfile(path)]


def config_files():
    """
    Return (install dir, files) for every directory under config/<robot>/.

    The annotated example.*.yaml files document the schemas and stay in the source tree, so
    no loader or listing of the installed configs can take one for a real config.
    """
    entries = []
    for robot_dir in sorted(glob('config/*/')):
        for directory, _, names in sorted(os.walk(robot_dir)):
            paths = sorted(os.path.join(directory, name) for name in names
                           if not name.startswith('example.'))
            paths = [path for path in paths if os.path.isfile(path)]
            if paths:
                entries.append((os.path.join('share', package_name,
                                             os.path.normpath(directory)), paths))
    return entries


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), files('launch/*.launch.py')),
        # the H1-2 policies' weights and READMEs; their specs are config/h12/policies/
        (os.path.join('share', package_name, 'policies', 'walk'),
            files('policies/walk/*')),
        (os.path.join('share', package_name, 'policies', 'fame'),
            files('policies/fame/*')),
        (os.path.join('share', package_name, 'policies', 'almi'),
            files('policies/almi/*')),
        # HOMIE's network with its licence (CC BY-NC-SA) and README; homie_policy.py and
        # the rest of the folder serve its MuJoCo script and stay in the source tree
        (os.path.join('share', package_name, 'policies', 'homie'),
            ['policies/homie/deploy.onnx', 'policies/homie/LICENSE',
             'policies/homie/README.md']),
    ] + config_files(),
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maxconway',
    maintainer_email='maxlconway@gmail.com',
    description='Leg controller: runs the RL locomotion policies on the leg joints.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lowerbody_controller_node = golem_lowerbody_rl.node:main',
        ],
    },
)
