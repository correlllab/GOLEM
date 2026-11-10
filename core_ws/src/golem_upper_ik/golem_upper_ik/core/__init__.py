"""Compute the controller, robot-agnostic: models, dynamics, IK, planner, ROS I/O, commands.

robot_model and robot_dynamics are the Pinocchio models; ik_solver the Pink IK; planner the
OMPL planner in its own process; ros_io the ROS I/O; low_cmd_handler and follow_gate the
command path; controller the controllers built on them.
"""
