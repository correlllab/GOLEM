#!/usr/bin/env bash
# Native ROS checker; build only message interfaces if the image lacks them.
set -eo pipefail
source /opt/ros/humble/setup.bash
if [ -f /home/code/core_ws/install/setup.bash ]; then source /home/code/core_ws/install/setup.bash; fi
if ! /usr/bin/python3 -c 'from unitree_hg.msg import LowCmd, LowState; from unitree_go.msg import MotorCmds, MotorStates; from livox_ros_driver2.msg import CustomMsg; from magpie_msgs.msg import GripperState; from magpie_msgs.srv import SetGripperPosition, SetGripperForce; from magpie_msgs.action import DeliGrasp' >/dev/null 2>&1; then
    workspace=$(mktemp -d)
    for package in unitree_hg unitree_go livox_ros_driver2 magpie_msgs; do
        mkdir -p "$workspace/src/$package/msg"
        if [ "$package" = unitree_hg ]; then
            msg_root=/home/code/core_ws/src/unitree_ros2/cyclonedds_ws/src/unitree/unitree_hg/msg
            for message in LowCmd LowState MotorCmd MotorState IMUState; do cp "$msg_root/$message.msg" "$workspace/src/$package/msg/"; done
        elif [ "$package" = unitree_go ]; then
            msg_root=/home/code/core_ws/src/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go/msg
            for message in MotorCmds MotorStates MotorCmd MotorState; do cp "$msg_root/$message.msg" "$workspace/src/$package/msg/"; done
        elif [ "$package" = magpie_msgs ]; then
            mkdir -p "$workspace/src/$package/srv" "$workspace/src/$package/action"
            cp /home/code/core_ws/src/magpie_msgs/msg/{GripperState,DeliGraspParams}.msg "$workspace/src/$package/msg/"
            cp /home/code/core_ws/src/magpie_msgs/srv/{SetGripperPosition,SetGripperForce}.srv "$workspace/src/$package/srv/"
            cp /home/code/core_ws/src/magpie_msgs/action/DeliGrasp.action "$workspace/src/$package/action/"
        else
            cp /home/code/core_ws/src/livox_ros_driver2/msg/Custom{Msg,Point}.msg "$workspace/src/$package/msg/"
        fi
        cat > "$workspace/src/$package/package.xml" <<XML
<?xml version="1.0"?>
<package format="3"><name>$package</name><version>0.0.0</version><description>Pinned wire test interfaces</description>
<maintainer email="test@example.invalid">GOLEM test</maintainer><license>MIT</license>
<buildtool_depend>ament_cmake</buildtool_depend><build_depend>rosidl_default_generators</build_depend>
<depend>std_msgs</depend><depend>action_msgs</depend><exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group></package>
XML
        cat > "$workspace/src/$package/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.8)
project($package)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
file(GLOB msg_files RELATIVE \${CMAKE_CURRENT_SOURCE_DIR} msg/*.msg srv/*.srv action/*.action)
rosidl_generate_interfaces(\${PROJECT_NAME} \${msg_files} DEPENDENCIES std_msgs action_msgs)
ament_package()
CMAKE
    done
    cd "$workspace"
    colcon build --packages-select unitree_hg unitree_go livox_ros_driver2 magpie_msgs --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
    source install/setup.bash
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1
exec /usr/bin/python3 /home/code/tests/isaaclab/check_robot_ros.py
