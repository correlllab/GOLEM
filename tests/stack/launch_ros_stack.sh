#!/usr/bin/env bash
# Build the controllers a stack case needs into a test install, then launch them.
#
# Runs inside the ROS test container that run.sh starts.
# Usage: bash /home/code/tests/stack/launch_ros_stack.sh [--build-only]
#   --build-only   build, run test_heartbeat_clock.py and exit 0 without launching.
# Environment: ROS_DOMAIN_ID (required, 1..232), TEST_CASE (arms|locomotion, default arms),
#   SIMULATOR (isaac|robocasa, default isaac).
# Prints STACK_BUILD_READY once built; run.sh waits for that line.
# Exit: 2 for a bad domain, case or simulator; otherwise the failing step's status, or the launch's.
set -eo pipefail

# SAFETY: never on domain 0, the real robots' command bus.
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

# Build only the controllers' dependency closure into its own build and install bases, so the
# test inherits no stale interfaces from core_ws/install and builds neither MJPC nor vision.
# golem_ros2_models' setup.py stages the models from CL_Assets at ../../../CL_Assets relative to
# itself, even from colcon's build copy, so the build base sits at the depth of src.
TEST_BUILD_BASE=$WS/build_stack_tests
TEST_INSTALL_BASE=$WS/install_stack_tests
cd "$WS"
colcon --log-base "$WS/log_stack_tests" build \
    --base-paths "$WS/src" \
    --build-base "$TEST_BUILD_BASE" \
    --install-base "$TEST_INSTALL_BASE" \
    --symlink-install \
    --packages-up-to golem_safety golem_upper_ik golem_lowerbody_rl unitree_hg unitree_go \
    --cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3
source "$TEST_INSTALL_BASE/setup.bash"

# The safety heartbeat must run on a paused clock; checked before any sim is up.
python3 /home/code/tests/stack/test_heartbeat_clock.py
echo "STACK_BUILD_READY case=$TEST_CASE simulator=$SIMULATOR domain=$ROS_DOMAIN_ID"
if [ "${1:-}" = --build-only ]; then
    exit 0
fi

# Launch only once the H1-2 sim publishes rt/lowstate, then start the controllers.
python3 /home/code/h12_sim_scripts/wait_for_sim.py
exec ros2 launch /home/code/tests/stack/minimal_stack.launch.py
