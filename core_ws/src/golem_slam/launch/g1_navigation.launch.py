"""G1 navigation through the shared golem_slam pipeline."""
from golem_slam.navigation_launch import generate_navigation_launch_description


def generate_launch_description():
    return generate_navigation_launch_description('g1')
