'''The viewer launch file: one robot's ROS copy to robot_state_publisher, sliders and RViz'''
import importlib.util
import os

import pytest
import yaml

from golem_ros2_models import paths


def display_launch():
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory('golem_ros2_models'), 'launch',
                        'display.launch.py')
    spec = importlib.util.spec_from_file_location('display_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robot_argument():
    from launch.actions import DeclareLaunchArgument
    description = display_launch().generate_launch_description()
    arguments = {action.name: action for action in description.entities
                 if isinstance(action, DeclareLaunchArgument)}
    assert arguments['robot'].default_value[0].text == 'h12'
    assert list(arguments['robot'].choices) == list(paths.ROBOTS)


@pytest.mark.parametrize('robot', paths.ROBOTS)
def test_nodes(robot):
    from launch import LaunchContext
    from launch.utilities import perform_substitutions
    context = LaunchContext()
    context.launch_configurations['robot'] = robot
    nodes = display_launch().launch_setup(context)
    assert sorted(node.node_package for node in nodes) == [
        'joint_state_publisher_gui', 'robot_state_publisher', 'rviz2']
    publisher = next(n for n in nodes if n.node_package == 'robot_state_publisher')
    parameters = {perform_substitutions(context, key): perform_substitutions(context, value)
                  for key, value in publisher._Node__parameters[0].items()}
    # launch_ros passes a multi-line string parameter on as a YAML scalar
    description = yaml.safe_load(parameters['robot_description'])
    with open(paths.urdf_path(robot, ros=True)) as f:
        assert description == f.read()
    assert f'package://golem_ros2_models/{robot}/meshes/' in description
