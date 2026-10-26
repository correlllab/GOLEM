from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# G1 models from the leprechaun_g1 submodule, for running examples without ROS
ASSETS_PATH = PROJECT_ROOT.parents[3] / 'third_party' / 'leprechaun_g1' / 'assets' / 'g1_description'

URDF_PATH = str(ASSETS_PATH / 'g1_29dof_with_hand_rev_1_0.urdf')
URDF_BODY_PATH = str(ASSETS_PATH / 'g1_29dof_rev_1_0.urdf')
