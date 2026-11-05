#!/usr/bin/env bash
# Set up and run the containers of the G1 acceptance tests (a library, sourced by test_sim_*.sh).
#
# These are tests/sim/common.sh's containers for the G1 sim (robocasa with the G1 overlay,
# GOLEM_ROBOT=g1), with the build in $GOLEM_CACHE_DIR/g1_tests and the output in
# tests/results/g1-<name>-<time>-<pid>. tests/sim/common.sh documents the containers, the
# build, the sim's knobs and the exit codes.
G1_TESTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$G1_TESTS/../sim/common.sh"

# g1_test_shell_knobs NAME...: sim_test_shell_knobs.
g1_test_shell_knobs() { sim_test_shell_knobs "$@"; }

# g1_test_setup TAG NAME CASE DOMAIN: sim_test_setup for the G1, in the g1 suite.
#   TAG     the test's log prefix
#   NAME    the output directory's name: bringup, walk, nav or explore
#   CASE    the variant (arm, walk, nav, explore), recorded in run.json
#   DOMAIN  the DDS domain unless ROS_DOMAIN_ID is set; a positive integer
g1_test_setup() { sim_test_setup g1 g1 "$@"; }

# g1_test_log_load: sim_test_log_load.
g1_test_log_load() { sim_test_log_load; }
# g1_test_build_ros: sim_test_build_ros.
g1_test_build_ros() { sim_test_build_ros; }
# g1_test_start_sim PY [ARG...]: sim_test_start_sim.
g1_test_start_sim() { sim_test_start_sim "$@"; }
# g1_test_run_ros SECONDS SCRIPT: sim_test_run_ros.
g1_test_run_ros() { sim_test_run_ros "$@"; }
# g1_test_stop_sim: sim_test_stop_sim.
g1_test_stop_sim() { sim_test_stop_sim; }
# g1_test_report PY: sim_test_report.
g1_test_report() { sim_test_report "$@"; }
