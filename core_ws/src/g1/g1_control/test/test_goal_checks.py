from types import SimpleNamespace
from g1_control.utility.goal_checks import (
    unsupported_frame_task_fields, unsupported_named_config_fields)

def test_absolute_full_speed_frame_task_is_supported():
    assert unsupported_frame_task_fields(SimpleNamespace(relative=False, slow_mode=False)) == []

def test_relative_frame_task_is_rejected():
    assert unsupported_frame_task_fields(SimpleNamespace(relative=True, slow_mode=False)) == ['relative']

def test_slow_mode_frame_task_is_rejected():
    assert unsupported_frame_task_fields(SimpleNamespace(relative=False, slow_mode=True)) == ['slow_mode']

def test_both_fields_are_reported_in_order():
    assert unsupported_frame_task_fields(SimpleNamespace(relative=True, slow_mode=True)) == ['relative', 'slow_mode']

def test_named_config_slow_mode_is_rejected():
    assert unsupported_named_config_fields(SimpleNamespace(slow_mode=True)) == ['slow_mode']
    assert unsupported_named_config_fields(SimpleNamespace(slow_mode=False)) == []
