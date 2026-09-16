#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
exec "$ROOT/tests/isaaclab/run_in_container.sh" 'export LD_LIBRARY_PATH=/workspace/golem/isaac/cyclonedds/install/lib:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH; export GOLEM_TEST_ROS_TRANSPORT=1; export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; export CYCLONEDDS_HOME=/workspace/golem/isaac/cyclonedds/install; cd /home/code/CL_isaaclab_sim; /isaac-sim/python.sh -m unittest discover -s /home/code/tests/isaaclab -p test_sensor_transport.py -v'
