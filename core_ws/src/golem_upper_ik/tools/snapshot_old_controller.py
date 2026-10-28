#!/usr/bin/env python3
'''
Writes what the old arm controllers hold and derive, for golem_upper_ik's data-layer tests.

From h12_ros2_controller and g1_control it writes, under test/fixtures/old_controller/ of
this package:

  <robot>.json      the joint tables (body, upper-body and IK joints, the old joint limits),
                    the named configurations and the start-up sequence, the planner's default
                    frames, and for each carried config what that package's own
                    load_controller_config returns from it (topics, network, controller,
                    frequency, planner, gains, the clip limits, logging)
  g1_pairing.json   g1_control's config_pairing verdict (accepted, refused with its message,
                    or the exception raised) for every combination in pairing_cases(), and its
                    default_controller_config for every safety config name there

The tests import neither old package. Both must import, so run it in the G1-mode ros
container after the workspace is built and sourced:

  python3 /home/code/core_ws/src/golem_upper_ik/tools/snapshot_old_controller.py
'''
import itertools
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
SOURCES = PACKAGE.parent
OUTPUT = PACKAGE / 'test' / 'fixtures' / 'old_controller'

CONFIGS = {
    'h12': ('safety_split', 'sim_safety_split', 'safety_full', 'debug'),
    'g1': ('g1', 'g1_split', 'sim', 'sim_split'),
}

# config names given to g1_control's pairing: its own, spelled as a name, a file and a path,
# custom ones (by name a real one and a sim one), and an H1-2 one
PAIRING_CONTROLLERS = (
    'g1', 'g1_split', 'sim', 'sim_split', 'g1.yaml', 'config/sim_split.yaml',
    '/opt/configs/g1_split.yaml', 'custom', 'sim_custom', 'safety_split',
)
PAIRING_SAFETIES = (
    'arm_sdk', 'split', 'full', 'sim_arm_sdk', 'sim_split', 'arm_sdk.yaml',
    '/opt/configs/sim_split.yaml', 'custom', 'sim_custom', 'relax_safety_split',
)


def floats(values):
    return [float(x) for x in values]


def describe_config(config):
    '''The old loader's result as JSON values; arrays as lists of floats'''
    limits = config['limits']
    q_clip = limits['q_clip_limits']
    return {
        'mode': config.get('mode'),
        'topics': dict(config['topics']),
        'network': dict(config['network']),
        'controller': dict(config['controller']),
        'frequency': dict(config['frequency']),
        'planner': dict(config['planner']),
        'gains': {name: floats(config['gains'][name]) for name in ('kp', 'kd', 'ki')},
        'clip': {
            'q_low': floats(q_clip[:, 0]),
            'q_high': floats(q_clip[:, 1]),
            'dq': floats(limits['dq_clip_limits']),
            'tau': floats(limits['tau_clip_limits']),
        },
        'logging': dict(config['logging']),
    }


def joint_limits(module):
    return {
        'position': [[float(limit['low']), float(limit['high'])]
                     for limit in module.JOINT_POSITION_LIMITS],
        'velocity': floats(module.JOINT_VELOCITY_LIMITS),
        'effort': floats(module.JOINT_TORQUE_LIMITS),
    }


def h12_startup_sequence():
    '''The named configs UpperController._startup_routine moves through, in order'''
    from h12_ros2_controller.core.controller.upper_controller import UpperController
    from h12_ros2_controller.utility.named_config import NAMED_CONFIGS

    class Recorder:
        def __init__(self):
            self.names = []

        def _move_to_reduced_configuration(self, q_reduced, *args, **kwargs):
            self.names.append(next(name for name, q in NAMED_CONFIGS.items() if q is q_reduced))

        def _move_torso_to_target(self, *args, **kwargs):
            pass

    recorder = Recorder()
    UpperController._startup_routine(recorder, 0.0, 0.0)
    return recorder.names


def snapshot_h12():
    from h12_ros2_controller.core.planner import PlannerConfig
    from h12_ros2_controller.utility import joint_definition, joint_limits as limits_module
    from h12_ros2_controller.utility.controller_config import load_controller_config
    from h12_ros2_controller.utility.named_config import NAMED_CONFIGS

    config_dir = SOURCES / 'h12_ros2_controller' / 'config'
    return {
        'robot': 'h12',
        'body_joints': list(joint_definition.BODY_JOINTS),
        'upper_body_joints': list(joint_definition.UPPER_BODY_JOINTS),
        'upper_body_index': list(joint_definition.UPPER_BODY_INDEX),
        'lower_body_index': list(joint_definition.LOWER_BODY_INDEX),
        'enabled_joints': list(joint_definition.ENABLED_JOINTS),
        'joint_limits': joint_limits(limits_module),
        'named_configs': {name: floats(q) for name, q in NAMED_CONFIGS.items()},
        'startup_sequence': h12_startup_sequence(),
        'planner_frame_names': list(PlannerConfig().frame_names),
        'configs': {
            name: describe_config(load_controller_config(name, config_dir=config_dir))
            for name in CONFIGS['h12']
        },
    }


def snapshot_g1():
    from g1_control.core.planner import PlannerConfig
    from g1_control.utility import joint_definition, joint_limits as limits_module
    from g1_control.utility.controller_config import load_controller_config
    from g1_control.utility.named_config import NAMED_CONFIGS, STARTUP_SEQUENCE

    config_dir = SOURCES / 'g1' / 'g1_control' / 'config'
    return {
        'robot': 'g1',
        'body_joints': list(joint_definition.BODY_JOINTS),
        'upper_body_joints': list(joint_definition.UPPER_BODY_JOINTS),
        'upper_body_index': list(joint_definition.UPPER_BODY_INDEX),
        'lower_body_index': list(joint_definition.LOWER_BODY_INDEX),
        'enabled_joints': list(joint_definition.ENABLED_JOINTS),
        'waist_joints': list(joint_definition.WAIST_JOINTS),
        'ee_frames': [joint_definition.LEFT_EE_FRAME, joint_definition.RIGHT_EE_FRAME],
        'joint_limits': joint_limits(limits_module),
        'named_configs': {name: floats(q) for name, q in NAMED_CONFIGS.items()},
        'startup_sequence': list(STARTUP_SEQUENCE),
        'planner_frame_names': list(PlannerConfig().frame_names),
        'configs': {
            name: describe_config(load_controller_config(name, config_dir=config_dir))
            for name in CONFIGS['g1']
        },
    }


def snapshot_g1_pairing():
    from g1_control.utility.config_pairing import default_controller_config, pairing_refusal

    cases = []
    for controller, safety, sim, locomotion in itertools.product(
            PAIRING_CONTROLLERS, PAIRING_SAFETIES, (None, False, True), (False, True)):
        case = {'controller': controller, 'safety': safety, 'sim': sim, 'locomotion': locomotion}
        try:
            refusal = pairing_refusal(controller, safety, sim=sim, locomotion=locomotion)
        except Exception as error:  # the verdict is the exception's type
            case.update(verdict='raises', exception=type(error).__name__)
        else:
            case.update(verdict='accepted' if refusal is None else 'refused', message=refusal)
        cases.append(case)
    return {
        'cases': cases,
        'default_controller_config': {
            safety: default_controller_config(safety) for safety in PAIRING_SAFETIES
        },
    }


def write(name, data):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / name
    target.write_text(json.dumps(data, indent=1) + '\n')
    print(f'wrote {target.relative_to(PACKAGE)}')


def main():
    write('h12.json', snapshot_h12())
    write('g1.json', snapshot_g1())
    write('g1_pairing.json', snapshot_g1_pairing())


if __name__ == '__main__':
    main()
