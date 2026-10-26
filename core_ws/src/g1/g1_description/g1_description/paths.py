"""Installed locations of the G1 URDFs (needs the package built and sourced)."""

import os

VARIANTS = {
    'with_hand': 'g1_29dof_with_hand_rev_1_0',  # our robot: body + both Dex3-1 hands
    'body': 'g1_29dof_rev_1_0',  # body only, rubber hands
}


def assets_dir():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('g1_description'), 'assets')


def urdf_path(variant='with_hand', ros=False):
    """Path to the URDF for `variant`.

    ros=True gives the copy with package:// mesh URIs, for robot_state_publisher
    and RViz. The default is upstream's file, whose relative 'meshes/...' paths
    resolve against assets_dir() (e.g. for Pinocchio).
    """
    return os.path.join(assets_dir(), VARIANTS[variant] + ('_ros' if ros else '') + '.urdf')
