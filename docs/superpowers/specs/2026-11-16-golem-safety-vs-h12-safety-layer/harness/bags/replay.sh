#!/bin/bash
# Replay a real-robot bag's safety inputs, /lowstate and estop through golem_safety
# and record its /lowcmd beside the replayed inputs.
# Usage (golem_ros container): bash replay.sh <bag dir> <out bag dir> [config]
# Runs on ROS_DOMAIN_ID 93, never 0. Starts golem_safety's safety_node from /tmp/gs
# (sim:=true, use_estop:=true), a recorder, then plays the bag once; stops what it started.

BAG=$1; OUT=$2; CONFIG=${3:-relax_safety_split}
export ROS_DOMAIN_ID=93
source /opt/ros/humble/setup.bash; source /home/code/core_ws/install/setup.bash
TOPICS="/lowstate /safety/lowcmd_lower_in /safety/lowcmd_upper_in /estop/triggered /estop/plugged_in"
rm -rf "$OUT"
# The layer under test.
setsid /tmp/gs/install/golem_safety/lib/golem_safety/safety_node --ros-args -p robot:=h12 \
  -p config:=$CONFIG -p sim:=true -p use_estop:=true > "$OUT.safety.log" 2>&1 &
SAFETY=$!
# The recorder: the replayed inputs and the layer's output, on one clock.
setsid ros2 bag record -o "$OUT" $TOPICS /lowcmd > "$OUT.record.log" 2>&1 &
RECORD=$!
sleep 3
# The inputs, at the recorded rate.
ros2 bag play "$BAG" --topics $TOPICS > "$OUT.play.log" 2>&1
sleep 1
kill -INT -$RECORD -$SAFETY 2>/dev/null
sleep 2
kill -9 -$RECORD -$SAFETY 2>/dev/null
echo "replayed $BAG -> $OUT"
