"""Offline launch composition tests with no ROS actions or processes executed."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'launch/h12_native_experiment.launch.py'


class Configuration:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context[self.name]


class NativeExperimentTests(unittest.TestCase):
    def setUp(self):
        def action(kind):
            return lambda *args, **kwargs: SimpleNamespace(kind=kind, args=args, **kwargs)
        definitions = {
            'ament_index_python.packages': {'get_package_share_directory': lambda name: '/native/' + name},
            'launch': {'LaunchDescription': list},
            'launch.actions': {name: action(name) for name in (
                'DeclareLaunchArgument', 'IncludeLaunchDescription', 'OpaqueFunction')},
            'launch.launch_description_sources': {'PythonLaunchDescriptionSource': lambda path: path},
            'launch.substitutions': {'LaunchConfiguration': Configuration,
                                     'EnvironmentVariable': action('EnvironmentVariable')},
            'launch_ros.actions': {'Node': action('Node')},
        }
        modules = {}
        for name, attributes in definitions.items():
            module = ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        with patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('native_experiment', SOURCE)
            self.launch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.launch)
        self.context = {'controller_python': '/usr/bin/python3', 'lowerbody': 'fame_finetune',
                        'upper_controller': 'frame',
                        'start_position_verified': 'false', 'marker_catalog_path': str(SOURCE)}

    def compose(self):
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '1'}):
            return self.launch._launch(self.context)

    def test_default_contains_no_actuator_or_skill_actions(self):
        actions = self.compose()
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].args[0], '/native/h12_bringup/launch/h12_state.launch.py')
        state = dict(actions[0].launch_arguments)
        self.assertEqual(state['joint_state_prefix'], '/usr/bin/python3')
        self.assertEqual(state['use_sim_time'], 'true')
        self.assertEqual(actions[1].executable, 'aruco_localizer')
        self.assertEqual(actions[1].parameters[0]['use_sim_time'], True)
        self.assertTrue(actions[1].parameters[0]['publish_visualization'])
        self.assertEqual(actions[1].parameters[0]['publish_annotated'].name, 'publish_annotated')
        self.assertEqual(actions[1].parameters[0]['image_transport'].name, 'image_transport')

    def test_explicit_verified_start_has_one_baseline_upper_and_one_standing(self):
        self.context['start_position_verified'] = 'true'
        for policy in ('fame_finetune', 'almi_manip_stand'):
            self.context['lowerbody'] = policy
            actions = self.compose()
            self.assertEqual(len(actions), 5)
            control = dict(actions[2].launch_arguments)
            lower = dict(actions[3].launch_arguments)
            self.assertTrue(actions[2].args[0].endswith('/h12_control.launch.py'))
            self.assertEqual(control['frame_task_prefix'], '/usr/bin/python3')
            self.assertEqual(control['safety_config'], 'sim_safety_split.yaml')
            self.assertEqual(control['upper_controller'], 'frame')
            self.assertEqual(control['frame_task_config'], 'sim_safety_split.yaml')
            self.assertEqual(lower['start_position_verified'], 'true')
            self.assertEqual(lower['lowerbody'], policy)
            self.assertEqual(lower['use_estimator'], 'false')
            self.assertTrue(lower['lowerbody_params'].endswith('/config/lowerbody_experiment.yaml'))
            self.assertEqual(actions[4].executable, 'skills')
            self.assertEqual(actions[4].parameters[0]['target_backend'], 'marker')
            self.assertNotIn('h12_models', str(actions))
            self.assertNotIn('h12_navigation', str(actions))

    def test_counter_selection_is_independent_of_policy_and_requires_verification(self):
        self.context['upper_controller'] = 'counter'
        for policy in ('fame_finetune', 'almi_manip_stand'):
            self.context['lowerbody'] = policy
            self.context['start_position_verified'] = 'false'
            self.assertEqual(len(self.compose()), 2)
            self.context['start_position_verified'] = 'true'
            actions = self.compose()
            self.assertEqual(len(actions), 5)
            control = dict(actions[2].launch_arguments)
            self.assertEqual(control['upper_controller'], 'counter')
            self.assertEqual(control['frame_task_config'], 'counter_safety_split.yaml')
            self.assertEqual(control['safety_config'], 'sim_safety_split.yaml')
            self.assertEqual(dict(actions[3].launch_arguments)['lowerbody'], policy)

    def test_invalid_upper_controller_is_rejected_even_without_actuation(self):
        for upper in ('', 'robust3c', 'Frame', 'counter; touch /invalid'):
            self.context['upper_controller'] = upper
            with self.assertRaisesRegex(RuntimeError, 'upper_controller must be frame or counter'):
                self.compose()

    def test_control_hook_passes_typed_parameter_to_exactly_one_upper(self):
        def action(*args, **kwargs):
            return SimpleNamespace(args=args, **kwargs)
        descriptions = ModuleType('launch_ros.parameter_descriptions')
        descriptions.ParameterValue = lambda value, value_type: SimpleNamespace(
            value=value, value_type=value_type)
        with patch.dict(sys.modules, {'launch_ros.parameter_descriptions': descriptions}), \
                patch.object(sys.modules['launch.actions'], 'TimerAction', action, create=True), \
                patch.dict(sys.modules, {'launch.conditions': SimpleNamespace(IfCondition=action)}):
            spec = importlib.util.spec_from_file_location('control_launch',
                SOURCE.with_name('h12_control.launch.py'))
            control = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(control)
            actions = control.generate_launch_description()
        upper = [node for timer in actions if hasattr(timer, 'actions')
                 for node in timer.actions if node.executable == 'frame_task_server']
        self.assertEqual(len(upper), 1)
        self.assertEqual(upper[0].name, 'frame_task_server')
        self.assertEqual(len(upper[0].arguments), 2)
        self.assertEqual(upper[0].arguments[0], '--config')
        parameter = upper[0].parameters[1]['upper_controller']
        self.assertIs(parameter.value_type, str)
        for mode in ('frame', 'counter'):
            self.assertEqual(parameter.value.perform({'upper_controller': mode}), mode)
        declared = next(item for item in actions if item.args == ('upper_controller',))
        self.assertEqual(declared.default_value, 'frame')
        self.assertEqual(declared.choices, ['frame', 'counter'])

    def test_invalid_domains_are_rejected_before_actions(self):
        for domain in ('0', '', '-1', '233', 'garbage'):
            with patch.dict(os.environ, {'ROS_DOMAIN_ID': domain}):
                with self.assertRaisesRegex(RuntimeError, 'real robot bus'):
                    self.launch._launch(self.context)

    def test_no_walking_robust_or_ambiguous_arming_values(self):
        for policy in ('fame', 'almi', 'walk', 'mjpc', 'robust'):
            self.context['lowerbody'] = policy
            with self.assertRaisesRegex(RuntimeError, 'lowerbody'):
                self.compose()
        self.context['lowerbody'] = 'fame_finetune'
        self.context['start_position_verified'] = 'yes'
        with self.assertRaisesRegex(RuntimeError, 'true or false'):
            self.compose()

    def test_no_custom_controller_interpreter_without_absolute_executable(self):
        self.context['controller_python'] = 'python3'
        with self.assertRaisesRegex(RuntimeError, 'run_experiment'):
            self.compose()

    def test_actuation_requires_catalog_without_custom_safety_implementation(self):
        self.context['start_position_verified'] = 'true'
        with patch.dict(sys.modules, {'h12_safety_layer.core.config': None}):
            actions = self.compose()
        self.assertEqual(dict(actions[2].launch_arguments)['safety_config'],
                         'sim_safety_split.yaml')
        self.context['marker_catalog_path'] = ''
        with self.assertRaisesRegex(RuntimeError, 'absolute marker catalog'):
            self.compose()

    def test_launch_default_is_unverified(self):
        descriptions = self.launch.generate_launch_description()
        verified = next(item for item in descriptions if item.args == ('start_position_verified',))
        self.assertEqual(verified.default_value, 'false')
        upper = next(item for item in descriptions if item.args == ('upper_controller',))
        self.assertEqual(upper.default_value, 'frame')
        lower = next(item for item in descriptions if item.args == ('lowerbody',))
        self.assertEqual(lower.default_value, 'fame_finetune')
        annotated = next(item for item in descriptions if item.args == ('publish_annotated',))
        self.assertEqual(annotated.default_value, 'true')
        transport = next(item for item in descriptions if item.args == ('image_transport',))
        self.assertEqual(transport.default_value, 'raw')

    def test_experiment_preserves_confirm_interlock(self):
        import yaml
        package = SOURCE.parents[1]
        lower = yaml.safe_load((package / 'config/lowerbody_experiment.yaml').read_text())
        params = lower['lowerbody_controller_node']['ros__parameters']
        self.assertEqual(params['control_hz'], 50.0)
        self.assertEqual(params['standing_lowstate_timeout'], 0.1)
        self.assertIs(params['engage_wait_for_confirm'], True)


if __name__ == '__main__':
    unittest.main()
