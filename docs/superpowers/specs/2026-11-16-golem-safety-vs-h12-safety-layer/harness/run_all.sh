#!/bin/bash
# Run every scenario on both layers (old on domain 92, new on 93) into ./out.
# Usage (golem_ros container): bash run_all.sh [scenario ...]
cd /tmp/safety_diff
source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash
S="${@:-startup_idle relay_clip_split relay_clip_full nan_unowned_slot upper_silent lower_silent estop_q_edge estop_dq state_nan engage_mm0 foreign_mm second_writer hw_estop_silent hw_estop_trigger}"
for s in $S; do
  python3 driver.py old $s 92 out & python3 driver.py new $s 93 out; wait
done
