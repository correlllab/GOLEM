"""The Unitree G1 (29 DoF + Dex3-1 hands) RoboCasa sim.

h12_mujoco.py runs it for --g1 or GOLEM_ROBOT=g1 (g1_mujoco.main). Import it
only as a package (``from g1 import g1_mujoco``) with golem_robocasa on sys.path,
never with this directory on sys.path: its unitree_interface and sim_names share
their names with the H1-2 sim's modules. The modules here import each other
relatively and the shared mujoco_ros_bridge and measurement_bridge absolutely.
"""
