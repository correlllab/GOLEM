#!/usr/bin/env bash
# Runs inside the ROS test container; only the actual controller dependency closure is built.
set -eo pipefail
case "${ROS_DOMAIN_ID:-}" in
    ''|*[!0-9]*) echo 'Set an isolated ROS_DOMAIN_ID in 1..232' >&2; exit 2 ;;
esac
if (( 10#$ROS_DOMAIN_ID < 1 || 10#$ROS_DOMAIN_ID > 232 )); then
    echo 'ROS_DOMAIN_ID must be in 1..232 for controller tests' >&2
    exit 2
fi
export TEST_CASE=${TEST_CASE:-arms}
export SIMULATOR=${SIMULATOR:-isaac}
case "$TEST_CASE" in arms|locomotion) ;; *) echo 'TEST_CASE must be arms or locomotion' >&2; exit 2 ;; esac
case "$SIMULATOR" in isaac|robocasa) ;; *) echo 'SIMULATOR must be isaac or robocasa' >&2; exit 2 ;; esac
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export PYTHONUNBUFFERED=1
WS=/home/code/core_ws
# An isolated install avoids inheriting stale interfaces or building MJPC/vision.
# golem_upper_ik loads the H1-2 URDFs and the sphere URDF/SRDF from
# golem_ros2_models' installed share, which golem_ros2_models' setup.py stages from
# CL_Assets at ../../../CL_Assets relative to itself, even when colcon runs its
# build copy: keep the build base at the depth of src.
TEST_BUILD_BASE=$WS/build_stack_tests
TEST_INSTALL_BASE=$WS/install_stack_tests
cd "$WS"
colcon --log-base "$WS/log_stack_tests" build \
    --base-paths "$WS/src" \
    --build-base "$TEST_BUILD_BASE" \
    --install-base "$TEST_INSTALL_BASE" \
    --symlink-install \
    --packages-up-to golem_safety golem_upper_ik h12_lowerbody_rl unitree_hg unitree_go \
    --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
source "$TEST_INSTALL_BASE/setup.bash"
python3 /home/code/tests/stack/test_heartbeat_clock.py
echo "STACK_BUILD_READY case=$TEST_CASE simulator=$SIMULATOR domain=$ROS_DOMAIN_ID"
if [ "${1:-}" = --build-only ]; then
    exit 0
fi
python3 /home/code/h12_sim_scripts/wait_for_sim.py
exec ros2 launch /home/code/tests/stack/minimal_stack.launch.py
