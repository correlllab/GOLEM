#!/bin/bash
# Shell functions for driving the H1-2 on the Mac stack, from inside the
# golem_ros container. docker/mac/docker-compose.yml mounts this directory, so
# edits on the Mac apply without a rebuild.
#
# Usage (inside golem_ros):
#   source /home/code/h12_sim_scripts/robot_cli.sh
#   rob_poses                       # list the named postures
#   rob_pose t_pose                 # move to a named posture (default 3 s)
#   rob_pose arms_overhead 4        # ... over 4 s
#   rob_home                        # the home posture
#   rob_grip right close            # close / open a gripper
#   rob_gripset left 50             # set a gripper's opening, mm
#   rob_eepose right                # print a hand's pose (pelvis frame)
#   rob_go 0.4 0 0.3                # walk (GOLEM_LOWERBODY=almi); rob_stop / rob_stand stop
#   rob_odom                        # FAST-LIO position (GOLEM_SLAM=1 or GOLEM_NAV2=1)
#   rob_explore                     # /skill/frontier_explore (no server on the Mac stack)
# For Cartesian reaching, send a /frame_task goal (core_ws/src/golem_upper_ik/README.md).
# The MuJoCo viewer, with GOLEM_DISPLAY=vnc: http://localhost:6080/vnc.html.

# The ROS environment of the Mac bringup (safe to source again).
source /opt/ros/humble/setup.bash 2>/dev/null
source /opt/core_ws/install/setup.bash 2>/dev/null
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# List the H1-2's named postures (named_configs in
# core_ws/src/golem_upper_ik/config/h12/robot.yaml, which is authoritative).
rob_poses() {
    echo "home t_pose arms_front_45 arms_overhead arms_asym arms_front_yaw init_1 init_2 init_3 prep staging_right staging_left"
}

# Move the arms to a named posture: rob_pose <name> [seconds].
rob_pose() {
    local cfg="${1:?usage: rob_pose <name> [secs] — see rob_poses}" secs="${2:-3}"
    echo "[rob_pose] -> $cfg over ${secs}s"
    ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
        "{config_name: $cfg, duration: {sec: $secs, nanosec: 0}}"
}

# Move the arms to home: rob_home [seconds].
rob_home() { rob_pose home "${1:-3}"; }

# Open or close a gripper: rob_grip <left|right> <open|close>.
rob_grip() {
    local side="${1:?usage: rob_grip <left|right> <open|close>}" act="${2:?open|close}"
    ros2 service call "/${side}/gripper/${act}" std_srvs/srv/Trigger
}

# Set a gripper's opening between the fingers: rob_gripset <left|right> <mm>
# (0 closed; the sim's magpie_hand_bridge.py spans 0..OPEN_MM, 110).
rob_gripset() {
    local side="${1:?usage: rob_gripset <left|right> <mm>}" pos="${2:?opening width, mm}"
    ros2 service call "/${side}/gripper/set_position" magpie_msgs/srv/SetGripperPosition \
        "{position: $pos}"
}

# Print one sample of a hand's pose, pelvis frame: rob_eepose [left|right].
# The ros2 CLI's echo can hang on a QoS mismatch with a live topic, hence the
# 8 s timeout.
rob_eepose() {
    local side="${1:-right}"
    timeout 8 ros2 topic echo "/${side}_ee_pose" --once \
        || echo "[rob_eepose] no sample in 8s (QoS-match CLI artifact; the topic is live)"
}

# Walking needs GOLEM_LOWERBODY=almi: the ALMI policy walks on a non-zero
# /cmd_vel and stands on zero.

# Command zero velocity once; ALMI stands.
rob_stand() { ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1; }
# Publish /cmd_vel at 20 Hz for secs (default 6): rob_go [vx] [vy] [wz] [secs],
# in m/s, m/s, rad/s (default 0.4 0 0). vx > 0 is forward, wz > 0 turns left.
rob_go() {
    local vx="${1:-0.4}" vy="${2:-0.0}" wz="${3:-0.0}" secs="${4:-6}"
    echo "[rob_go] vx=$vx vy=$vy wz=$wz for ${secs}s"
    timeout "$secs" ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: $vx, y: $vy}, angular: {z: $wz}}"
}

# Command zero velocity once; ALMI stands.
rob_stop() {
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
}

# Print FAST-LIO's position of its IMU (body) in camera_init, the frame of its
# first scan; needs GOLEM_SLAM=1 or GOLEM_NAV2=1.
rob_odom() { timeout 5 ros2 topic echo /Odometry --field pose.pose.position --once; }

# Send a /skill/frontier_explore goal with feedback: navigate to the nearest
# unexplored frontier of the SLAM map until it is mapped; Ctrl-C cancels. It
# needs GOLEM_SLAM=1 GOLEM_NAV2=1 and golem_skills' action server, which the
# Mac image does not build, so on the Mac stack the goal finds no server.
rob_explore() {
    ros2 action send_goal /skill/frontier_explore \
        custom_ros_messages/action/SkillFrontierExplore \
        "{min_frontier_cells: 0, blacklist_radius: 0.0, min_goal_distance: 0.0, goal_timeout: 0.0, timeout: {sec: 1800, nanosec: 0}}" \
        --feedback
}

echo "robot_cli loaded."
echo "  postures : rob_pose t_pose | rob_grip right close"
echo "  locomote : rob_go 0.4 0 0.3 (fwd+turn) -> rob_stop   (ALMI stand/walk; needs GOLEM_LOWERBODY=almi)"
echo "  explore  : rob_explore   (frontier mapping; needs GOLEM_SLAM=1 GOLEM_NAV2=1 and golem_skills, not built on the Mac)"
