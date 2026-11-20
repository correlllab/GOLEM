"""Run the Unitree G1 (29 DoF, Dex3-1 hands) in the RoboCasa sim.

golem_sim.py runs g1_robot.G1Robot for --g1 and GOLEM_ROBOT=g1. Import this
package as ``g1`` with golem_robocasa on sys.path, never with this directory on
sys.path: its unitree_interface and sim_names share their names with the H1-2
sim's modules. The modules here import each other relatively and the shared
golem_robocasa modules absolutely.
"""
