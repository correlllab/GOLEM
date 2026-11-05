"""Install the model_server Python package and its four server executables."""

from setuptools import find_packages, setup

package_name = 'model_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # weights/ is not installed: the servers read the checkpoints from the
        # source tree (sam_server and yolo_server through their module's real
        # path under --symlink-install, graspgen_server through
        # GRASPGENX_CHECKPOINT_DIR), and installing them would copy gigabytes
        # into install/share on every build.
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='max',
    maintainer_email='maxlconway@gmail.com',
    description='ROS 2 model servers for Gemini, SAM3, GraspGenX and YOLO-World.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gemini_server = model_server.gemini_server:RunGeminiServer',
            'sam_server = model_server.sam_server:RunSamServer',
            'graspgen_server = model_server.graspgen_server:main',
            'yolo_server = model_server.yolo_server:RunYoloServer',
        ],
    },
)
