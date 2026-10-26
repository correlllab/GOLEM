from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from g1_description.paths import urdf_path

# useful path
PACKAGE_NAME = 'g1_control'
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

# G1 29-DoF with both Dex3-1 hands, installed by g1_description.
# pinocchio resolves the upstream file's relative mesh paths; ROS needs package://
URDF_PIN_PATH = urdf_path('with_hand')
URDF_ROS_PATH = urdf_path('with_hand', ros=True)
# data path
HOME = str(Path.home())
LOG_PATH = f'{HOME}/.ros/log/{PACKAGE_NAME}'
CONFIG_DIR = Path(f'{PACKAGE_PATH}/config')
