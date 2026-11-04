"""Import checks only: never initialize ROS, DDS, rendering, or simulator objects."""

import argparse
import hashlib
import importlib
from importlib.metadata import distributions, version
import json
from pathlib import Path
import site
import sys

from setup_simulator import LIVOX, LIVOX_SHA, ROOT, SIM, VENV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridges', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 10) and sys.base_prefix == '/usr'
    assert Path(sys.prefix) == VENV and site.ENABLE_USER_SITE is False
    for package, expected in (('numpy', '2.2.5'), ('mujoco', '3.3.1'), ('cyclonedds', '0.10.2')):
        assert version(package) == expected, (package, version(package))
    for name in ('numpy', 'mujoco', 'PIL', 'yaml', 'cyclonedds.internal', 'rclpy', 'robosuite',
                 'unitree_sdk2py.core.channel', 'unitree_interface'):
        module = importlib.import_module(name)
        print(f'PASS import: {name} ({module.__file__})')
        if name in ('numpy', 'mujoco', 'robosuite'):
            assert Path(module.__file__).resolve().is_relative_to(VENV), module.__file__
    import unitree_sdk2py
    assert Path(unitree_sdk2py.__file__).resolve().is_relative_to(ROOT / 'unitree_sdk2_python')
    installed = [d for d in distributions(path=[str(VENV / 'lib/python3.10/site-packages')])
                 if d.metadata['Name'].replace('_', '-').lower() == 'unitree-sdk2py']
    assert len(installed) == 1
    assert json.loads(installed[0].read_text('direct_url.json'))['url'] == (ROOT / 'unitree_sdk2_python').as_uri()
    from importlib.metadata import distribution
    provenance = json.loads(distribution('robosuite').read_text('direct_url.json'))
    assert provenance['vcs_info']['commit_id'] == '5ce6643f3092639d08f7b0f90ed1c6a84f50552c', provenance
    if args.bridges:
        provenance = json.loads((VENV / 'livox-interfaces/provenance.json').read_text())
        assert provenance['revision'] == LIVOX_SHA and provenance['source'] == str(LIVOX), provenance
        for name, digest in provenance['sha256'].items():
            for path in (LIVOX / 'msg' / name,
                         VENV / 'livox-interfaces/install/share/livox_ros_driver2/msg' / name):
                assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        from rosidl_generator_py import import_type_support
        for package, prefix in (('magpie_msgs', ROOT / 'core_ws/install/magpie_msgs'),
                                ('livox_ros_driver2', VENV / 'livox-interfaces/install')):
            module = importlib.import_module(package)
            support = import_type_support(package)
            allowed = [prefix]
            if package == 'magpie_msgs':
                allowed.append(ROOT / 'core_ws/build/magpie_msgs')
            for origin in (module.__file__, support.__file__):
                assert any(Path(origin).resolve().is_relative_to(path) for path in allowed), origin
            print(f'PASS native generated Python/C type support: {package}')
        for name in ('mujoco_ros_bridge', 'magpie_hand_bridge'):
            module = importlib.import_module(name)
            assert Path(module.__file__).resolve().parent == SIM
            print(f'PASS actual bridge import: {name} (no constructor called)')
        # Exercise generated C conversions with NumPy 2, not just Python imports.
        from livox_ros_driver2.msg import CustomMsg, CustomPoint
        from magpie_msgs.msg import GripperState
        from rclpy.serialization import deserialize_message, serialize_message
        message = CustomMsg(point_num=1, points=[CustomPoint(x=1.25, line=2)])
        restored = deserialize_message(serialize_message(message), CustomMsg)
        assert restored.point_num == 1 and restored.points[0].x == 1.25
        assert restored.points[0].line == 2
        gripper = GripperState(position=42.5, force=1.25, finger_positions=[21.0, 21.5])
        restored_gripper = deserialize_message(serialize_message(gripper), GripperState)
        assert restored_gripper.position == 42.5 and restored_gripper.force == 1.25
        assert list(restored_gripper.finger_positions) == [21.0, 21.5]
        from sensor_msgs.msg import Image
        image = Image(height=1, width=1, encoding='rgb8', step=3, data=[11, 22, 33])
        assert list(deserialize_message(serialize_message(image), Image).data) == [11, 22, 33]
        print('PASS native Livox/Magpie/Image CDR serialization without ROS initialization')
    assert 'cv_bridge' not in sys.modules
    assert 'robocasa' not in sys.modules
    print('PASS: isolated simulator dependencies; no nodes/DDS/loop started. Runtime NOT qualified.')


if __name__ == '__main__':
    main()
