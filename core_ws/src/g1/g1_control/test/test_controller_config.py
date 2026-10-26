from textwrap import dedent

import pytest

from g1_control.utility.joint_definition import UPPER_BODY_INDEX
from g1_control.utility.controller_config import (
    load_controller_config,
)


def _gain_list(value):
    values = ', '.join([str(value)] * len(UPPER_BODY_INDEX))
    return f'[{values}]'


def _write_config(tmp_path, planner_section):
    config_path = tmp_path / 'test.yaml'
    content = dedent(
        f'''
        gains:
          kp: {_gain_list(1.0)}
          kd: {_gain_list(0.1)}
          ki: {_gain_list(0.0)}
        '''
    )
    content = f'{content}\n{planner_section.lstrip()}'
    config_path.write_text(
        content,
        encoding='utf-8',
    )
    return config_path


def test_load_controller_config_preserves_planner_section(tmp_path):
    _write_config(
        tmp_path,
        dedent(
            '''
            planner:
              planner: RRTstar
              timeout: 2.5
              range: 0.15
              moving_speed: 0.2
              dt: 0.02
              min_interpolation_steps: 3
              max_interpolation_steps: 250
              frame_names:
                - left_grasp_frame
            '''
        ),
    )

    config = load_controller_config('test.yaml', config_dir=tmp_path)

    assert config['planner'] == {
        'planner': 'RRTstar',
        'timeout': 2.5,
        'range': 0.15,
        'moving_speed': 0.2,
        'dt': 0.02,
        'min_interpolation_steps': 3,
        'max_interpolation_steps': 250,
        'frame_names': ['left_grasp_frame'],
    }


def test_load_controller_config_rejects_non_mapping_planner(tmp_path):
    _write_config(
        tmp_path,
        dedent(
            '''
            planner:
              - RRTConnect
            '''
        ),
    )

    with pytest.raises(ValueError, match='planner must be a mapping'):
        load_controller_config('test.yaml', config_dir=tmp_path)
