"""Run the Unitree G1 (29 DoF, Dex3-1 hands) in the RoboCasa sim.

h12_mujoco.py hands --g1 and GOLEM_ROBOT=g1 to g1_mujoco.main. Import this
package as ``g1`` with golem_robocasa on sys.path, never with this directory on
sys.path: its unitree_interface and sim_names share their names with the H1-2
sim's modules. The modules here import each other relatively and the shared
golem_robocasa modules absolutely.
"""
