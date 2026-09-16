#!/usr/bin/env bash
# Build only the pinned interfaces in an ephemeral workspace; never the driver.
set -eo pipefail
source /opt/ros/humble/setup.bash
workspace=$(mktemp -d)
mkdir -p "$workspace/src/livox_ros_driver2/msg"
cp /home/code/core_ws/src/livox_ros_driver2/msg/Custom{Msg,Point}.msg "$workspace/src/livox_ros_driver2/msg/"
cat > "$workspace/src/livox_ros_driver2/package.xml" <<'XML'
<?xml version="1.0"?>
<package format="3">
  <name>livox_ros_driver2</name><version>0.0.0</version><description>Wire test interfaces</description>
  <maintainer email="test@example.invalid">GOLEM test</maintainer><license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend><build_depend>rosidl_default_generators</build_depend>
  <depend>std_msgs</depend><exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
</package>
XML
cat > "$workspace/src/livox_ros_driver2/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.8)
project(livox_ros_driver2)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "msg/CustomMsg.msg" "msg/CustomPoint.msg" DEPENDENCIES std_msgs)
ament_package()
CMAKE
cd "$workspace"
colcon build --packages-select livox_ros_driver2 --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
echo SUBSCRIBER_READY
exec /usr/bin/python3 /home/code/tests/isaaclab/check_livox_ros.py
