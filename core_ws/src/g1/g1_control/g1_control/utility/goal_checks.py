'''Goal fields the G1 controller does not implement'''


def _set_fields(goal, names):
    return [name for name in names if getattr(goal, name)]


def unsupported_frame_task_fields(goal):
    '''Names of the FrameTask goal fields that are set but not implemented'''
    return _set_fields(goal, ('relative', 'slow_mode'))


def unsupported_named_config_fields(goal):
    '''Names of the NamedConfig goal fields that are set but not implemented'''
    return _set_fields(goal, ('slow_mode',))
