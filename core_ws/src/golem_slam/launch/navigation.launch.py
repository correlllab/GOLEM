"""
Start golem_slam's navigation graph for the robot argument.

The graph, its arguments and its start order are in golem_slam/navigation_launch.py,
which this file wraps so the package's code and tests can import it.
"""
from golem_slam import navigation_launch


def generate_launch_description():
    """Return golem_slam.navigation_launch's description."""
    return navigation_launch.generate_launch_description()
