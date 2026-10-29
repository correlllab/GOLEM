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
        'modes': ['split_mode', 'full_body_mode'],
        'lower_slots': [0, 11], 'upper_slots': [12, 28], 'joints': 29,
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
    assert len({joint['name'] for joint in data['joints']}) == joint_count
    for index, joint in enumerate(data['joints']):
        assert set(joint) == {'name', 'slot', 'position', 'velocity', 'effort'}
        assert joint['slot'] == index, joint['name']
        low, high = joint['position']
        assert low < high, joint['name']
        assert joint['velocity'] > 0 and joint['effort'] > 0, joint['name']


def test_g1_profile_has_no_arm_sdk():
    '''The G1 runs split_mode or full_body_mode, and no slot past the body carries a value'''
    data = load_robot('g1')
    assert data['modes'] == ['split_mode', 'full_body_mode']
    assert set(data) == {'robot', 'mode_machine', 'mode_pr', 'motor_slots', 'modes',
                         'lower_slots', 'upper_slots', 'joints'}


def check_against_table(robot, fixture, joint_count):
    table = json.loads((FIXTURES / fixture).read_text())
    data = load_robot(robot)

    names = table['joint_names']
    assert len(data['joints']) == len(names) == joint_count
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


def test_h12_data_matches_todays_table():
    check_against_table('h12', 'h12_joint_limits.json', 27)


def test_g1_data_matches_todays_table():
    check_against_table('g1', 'g1_joint_limits.json', 29)


def test_g1_slot_groups_partition_the_body():
    '''Legs 0-11, waist 12-14, left arm 15-21, right arm 22-28'''
    names = [joint['name'] for joint in load_robot('g1')['joints']]
    legs, waist, left_arm, right_arm = names[0:12], names[12:15], names[15:22], names[22:29]
    assert all(name.split('_')[1] in ('hip', 'knee', 'ankle') for name in legs)
    assert [name.split('_')[0] for name in legs] == ['left'] * 6 + ['right'] * 6
    assert waist == ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
    assert all(name.startswith('left_') for name in left_arm)
    assert all(name.startswith('right_') for name in right_arm)
    assert [name.removeprefix('left_') for name in left_arm] == \
        [name.removeprefix('right_') for name in right_arm]
    assert all(name.startswith(('left_shoulder', 'right_shoulder')) for name in
               left_arm[:3] + right_arm[:3])


def test_h12_shoulder_roll_keeps_the_table_limits_not_the_urdfs():
    by_name = {joint['name']: joint for joint in load_robot('h12')['joints']}
    assert by_name['left_shoulder_roll_joint']['position'] == [-0.38, 3.4]
    assert by_name['right_shoulder_roll_joint']['position'] == [-3.4, 0.38]
