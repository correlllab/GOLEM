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
        # NOTE: weights/ is intentionally NOT in data_files. The servers read
        # checkpoints from the source tree's model_server/weights/: sam_server
        # and yolo_server via __file__ (_WEIGHTS_DIR), which under
        # `colcon build --symlink-install` points back at the source tree, and
        # graspgen_server via GRASPGENX_CHECKPOINT_DIR, which the ros compose
        # points there. Adding the multi-GB .pt/.pth here would copy them into
        # install/share on every build for no benefit.
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
