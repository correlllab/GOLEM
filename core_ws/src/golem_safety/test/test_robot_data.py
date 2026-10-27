"""The installed robot data files, checked against the tables they were built from."""
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import pytest
import yaml

FIXTURES = Path(__file__).parent / 'fixtures'

SCHEMA = {
    'h12': {
        'robot': 'h12', 'mode_machine': 6, 'mode_pr': 0, 'motor_slots': 35,
        'modes': ['full_body_mode', 'split_mode'],
        'lower_slots': [0, 11], 'upper_slots': [12, 26], 'joints': 27,
    },
    'g1': {
        'robot': 'g1', 'mode_machine': 5, 'mode_pr': 0, 'motor_slots': 35,
        'modes': ['arm_sdk_mode', 'split_mode', 'full_body_mode'],
        'lower_slots': [0, 11], 'upper_slots': [12, 28], 'joints': 29,
        'arm_sdk_weight_slot': 29,
    },
}


def load_robot(robot):
    share = Path(get_package_share_directory('golem_safety'))
    return yaml.safe_load((share / 'robots' / f'{robot}.yaml').read_text())


@pytest.mark.parametrize('robot', sorted(SCHEMA))
def test_robot_file_has_the_schema_and_robot_facts(robot):
    data = load_robot(robot)
    expected = dict(SCHEMA[robot])
    joint_count = expected.pop('joints')
    assert set(data) == set(expected) | {'joints'}
    for key, value in expected.items():
        assert data[key] == value, key
    assert len(data['joints']) == joint_count
    for index, joint in enumerate(data['joints']):
        assert set(joint) == {'name', 'slot', 'position', 'velocity', 'effort'}
        assert joint['slot'] == index, joint['name']
        low, high = joint['position']
        assert low < high, joint['name']
        assert joint['velocity'] > 0 and joint['effort'] > 0, joint['name']


def test_g1_data_matches_g1_description():
    joints = pytest.importorskip('g1_description.joints')
    layout = pytest.importorskip('g1_description.layout')
    data = load_robot('g1')

    assert len(data['joints']) == len(joints.BODY)
    for entry, joint in zip(data['joints'], joints.BODY):
        assert entry['name'] == joint.name
        assert entry['slot'] == joint.slot
        assert entry['position'] == [joint.lower, joint.upper], joint.name
        assert entry['velocity'] == joint.velocity, joint.name
        assert entry['effort'] == joint.effort, joint.name

    assert data['motor_slots'] == layout.LOWCMD_MOTOR_SLOTS
    assert data['mode_pr'] == layout.MODE_PR
    assert data['arm_sdk_weight_slot'] == layout.ARM_SDK_WEIGHT_SLOT
    assert data['lower_slots'] == [layout.LEG_SLOTS.start, layout.LEG_SLOTS.stop - 1]
    assert data['upper_slots'] == [layout.ARM_SDK_SLOTS.start, layout.ARM_SDK_SLOTS.stop - 1]


def test_h12_data_matches_todays_table():
    table = json.loads((FIXTURES / 'h12_joint_limits.json').read_text())
    data = load_robot('h12')

    names = table['joint_names']
    assert len(data['joints']) == len(names) == 27
    for index, entry in enumerate(data['joints']):
        name = names[index]
        assert entry['name'] == name
        assert entry['slot'] == index, name
        assert entry['position'] == [
            table['position_limits'][index]['low'],
            table['position_limits'][index]['high'],
        ], name
        assert entry['velocity'] == table['velocity_limits'][index], name
        assert entry['effort'] == table['torque_limits'][index], name


def test_h12_shoulder_roll_keeps_the_table_limits_not_the_urdfs():
    by_name = {joint['name']: joint for joint in load_robot('h12')['joints']}
    assert by_name['left_shoulder_roll_joint']['position'] == [-0.38, 3.4]
    assert by_name['right_shoulder_roll_joint']['position'] == [-3.4, 0.38]
