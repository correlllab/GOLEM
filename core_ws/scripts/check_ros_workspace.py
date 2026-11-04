"""Check imports and source compilation without initializing ROS/DDS or devices."""

import argparse
import importlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import site
import sys
import py_compile

from build_ros_workspace import INSTALL, LOCAL_ROS, ROOT, VENV, WORKSPACE, package_path, packages


def native_origin(module, package, source=False):
    origin = Path(module.__file__).resolve()
    allowed = [INSTALL / package, WORKSPACE / 'build' / package]
    if source:
        allowed.append(package_path(package))
    assert any(origin.is_relative_to(path) for path in allowed), (package, origin)
    return origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built', action='store_true')
    parser.add_argument('--profile', choices=('core', 'interfaces', 'manipulation'), default='manipulation')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 10), sys.version
    assert Path(sys.prefix) == VENV, sys.prefix
    assert site.ENABLE_USER_SITE is False
    assert os.environ['PYTHONNOUSERSITE'] == '1'
    assert os.environ.get('ROS_DISTRO') == 'humble'
    for module in ('colcon_core', 'ament_package', 'em', 'catkin_pkg', 'lark',
                   'numpy', 'scipy', 'yaml', 'serial', 'rclpy', 'cv2', 'cv_bridge'):
        imported = importlib.import_module(module)
        print(f'PASS import: {module} ({getattr(imported, "__file__", "built-in")})')
    import numpy
    assert int(numpy.__version__.split('.')[0]) == 1, numpy.__version__
    # An actual bridge conversion exercises the compiled NumPy ABI, without ROS init.
    from cv_bridge import CvBridge
    image = numpy.zeros((2, 3, 3), dtype=numpy.uint8)
    bridge = CvBridge()
    numpy.testing.assert_array_equal(
        bridge.imgmsg_to_cv2(bridge.cv2_to_imgmsg(image, encoding='bgr8'), 'bgr8'), image)
    assert version('cyclonedds') == '0.10.2'
    import cyclonedds.internal
    from unitree_sdk2py.core import channel
    import unitree_sdk2py
    sdk = Path(unitree_sdk2py.__file__).resolve()
    assert sdk.is_relative_to(ROOT / 'unitree_sdk2_python'), sdk
    # Source egg-info on PYTHONPATH lacks installer provenance; inspect only the
    # venv's installed distribution rather than inherited or source metadata.
    installed_sdks = [item for item in distributions(path=[str(Path(sys.prefix) / 'lib/python3.10/site-packages')])
                      if item.metadata['Name'].replace('_', '-').lower() == 'unitree-sdk2py']
    assert len(installed_sdks) == 1, installed_sdks
    installed = json.loads(installed_sdks[0].read_text('direct_url.json'))
    assert installed['url'] == (ROOT / 'unitree_sdk2_python').as_uri(), installed
    assert channel.ChannelFactoryInitialize is not None
    print(f'PASS: NumPy {numpy.__version__}, cv_bridge conversion, CycloneDDS C load, SDK {sdk}')
    if LOCAL_ROS.is_dir():
        from ament_index_python.packages import get_package_prefix
        from rosidl_generator_py import import_type_support
        for package in ('rosidl_generator_dds_idl', 'nav2_msgs'):
            assert Path(get_package_prefix(package)).resolve() == LOCAL_ROS, package
            origin = Path(importlib.import_module(package).__file__).resolve()
            assert origin.is_relative_to(LOCAL_ROS), origin
        support = import_type_support('nav2_msgs')
        assert Path(support.__file__).resolve().is_relative_to(LOCAL_ROS), support.__file__
        print('PASS: rootless DDS IDL generator and nav2_msgs Python/C type support')
    if args.built:
        from ament_index_python.packages import get_package_prefix
        for package in packages(args.profile):
            prefix = Path(get_package_prefix(package)).resolve()
            assert prefix == INSTALL / package, (package, prefix)
            count = 0
            for source in package_path(package).rglob('*.py'):
                if any(part in ('_install_staging', '__pycache__', 'build') for part in source.parts):
                    continue
                output = (WORKSPACE / 'build' / package / 'native-bytecode'
                          / source.relative_to(package_path(package))).with_suffix('.pyc')
                output.parent.mkdir(parents=True, exist_ok=True)
                py_compile.compile(str(source), cfile=str(output), doraise=True)
                count += 1
            print(f'PASS native ament prefix and Python 3.10 compilation: {package} ({count} files)')
        for module in ('estop.estop_node', 'h12_safety_layer.ros2.safety_node'):
            native_origin(importlib.import_module(module), module.split('.')[0], source=True)
            print(f'PASS built import: {module} (no constructor called)')
        if args.profile in ('interfaces', 'manipulation'):
            from rosidl_generator_py import import_type_support
            generated = ['custom_ros_messages', 'magpie_msgs']
            if args.profile == 'manipulation':
                generated.append('unitree_hg')
            for package in generated:
                native_origin(importlib.import_module(package), package)
                native_origin(import_type_support(package), package)
                print(f'PASS generated C/Python type support: {package}')
            for module in ('custom_ros_messages.action', 'custom_ros_messages.srv',
                           'magpie_msgs.action', 'magpie_msgs.srv'):
                importlib.import_module(module)
    print('PASS: limited native checks; controller/skills/leg runtime NOT qualified; no actuation')


if __name__ == '__main__':
    main()
