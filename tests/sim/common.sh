#!/usr/bin/env bash
# Set up, run and tear down the containers of the closed-loop RoboCasa tests (a library).
#
# Sourced by tests/sim/test_sim_*.sh and, through tests/g1/common.sh, by the G1 acceptance
# tests; a test calls sim_test_setup first, then the other sim_test_* functions in run order.
#
# Both containers are the compose services as docker_run.sh starts them (robocasa, with the
# G1 overlay for the G1; ros from docker-compose.yml), with test_compose.yml on
# top, which sim_test_setup writes into the run's output directory:
#   - the robocasa container has no network and the ros container joins its
#     network namespace, both on its loopback with multicast, so no DDS leaves
#     the host and nothing talks to the robot;
#   - the sim mounts golem_robocasa read-only, and both mount tests/g1 and
#     tests/sim read-only (the tests' Python shares code between them);
#   - the ros container builds core_ws, both robots' packages (launch_ros.sh),
#     into $GOLEM_CACHE_DIR/<suite>_tests/{build,install,log}, so the checkout's
#     own core_ws/build and core_ws/install are left alone; CORE_WS=<dir> builds
#     that workspace in place instead.
# The tests of a suite share that build, so one runs at a time: each holds
# $GOLEM_CACHE_DIR/<suite>_tests.lock on fd 9, which its background jobs close.
#
# The sim gets every knob its robot's sim reads explicitly: the test's own, the
# ones from the shell (sim_test_shell_knobs) and the sim's defaults for the
# rest, so docker/.env never changes what is tested. Each sim ignores the other
# robot's knobs that the robocasa service also passes.
#
# Exit codes of the functions that exit: 1 a container or wait failed, or a check failed
# (sim_test_report); 2 a bad robot, domain or OUT; 3 the core_ws build failed.
SIM_TESTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SIM_TESTS/../../docker/scripts/docker_common.sh"

# Turns on multicast on the loopback interface (SIOCGIFFLAGS, then SIOCSIFFLAGS with
# IFF_MULTICAST), for DDS discovery inside the shared, network-less namespace.
MULTICAST="python3 -c \"import fcntl,socket,struct; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); \
f=struct.unpack('16sH',fcntl.ioctl(s,0x8913,struct.pack('16sH',b'lo',0)))[1]; \
fcntl.ioctl(s,0x8914,struct.pack('16sH',b'lo',f|0x1000))\""

# sim_test_shell_knobs NAME...: pass each NAME to the sim as the shell has it,
# empty (the sim's default) if unset. Call it before sim_test_setup, which reads
# docker/.env.
sim_test_shell_knobs() {
    local knob
    for knob in "$@"; do
        SIM_ENV+=(-e "$knob=${!knob:-}")
    done
}

# sim_json_string STRING: print STRING as a JSON string literal.
sim_json_string() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '"%s"' "$s"
}

# sim_test_setup ROBOT SUITE TAG NAME CASE DOMAIN: check the run's settings, write its
# compose overlay and take the suite's build lock.
#   ROBOT   h12 or g1: the sim's robot
#   SUITE   the tests' directory under tests/ (g1 or sim): the test's Python is
#           tests/SUITE/<py>, its build $GOLEM_CACHE_DIR/SUITE_tests
#   TAG     the test's log prefix
#   NAME    the output directory's name, tests/results/SUITE-NAME-<time>-<pid>
#   CASE    the variant, recorded in run.json
#   DOMAIN  the DDS domain unless ROS_DOMAIN_ID is set; a positive integer
# Reads CORE_WS, OUT, SIM_IMAGE, ROS_IMAGE and SIM_GPUS (the test scripts'
# headers) and sets OUT, SIM, ROS and COMPOSE. On exit the containers are
# removed and OUT gets run.json.
sim_test_setup() {
    SIM_TEST_ROBOT=$1 SIM_TEST_SUITE=$2 TAG=$3 TEST_NAME=$4 TEST_CASE=$5
    local domain=${ROS_DOMAIN_ID:-$6}
    case "$SIM_TEST_ROBOT" in
        h12|g1) ;;
        *) echo "[$TAG] the robot must be h12 or g1, not '$SIM_TEST_ROBOT'" >&2; exit 2 ;;
    esac
    if ! [ "$domain" -gt 0 ] 2>/dev/null; then
        echo "[$TAG] ROS_DOMAIN_ID must be a positive integer, not '$domain' (0 is the real robot)" >&2
        exit 2
    fi
    export ROS_DOMAIN_ID=$domain
    SIM_IMAGE=${SIM_IMAGE:-golem_sim_robocasa:latest}
    ROS_IMAGE=${ROS_IMAGE:-golem_ros:latest}
    SIM_GPUS=${SIM_GPUS-all}
    local core_ws='' out=''
    [ -z "${CORE_WS:-}" ] || core_ws=$(cd -- "$CORE_WS" && pwd)
    if [ -n "${OUT:-}" ]; then
        mkdir -p "$OUT"
        out=$(cd -- "$OUT" && pwd)
    fi
    export GOLEM_ROBOT=$SIM_TEST_ROBOT GOLEM_DISPLAY=headless
    ROBOT_FLAG=
    load_config
    configure_robot robocasa
    check_domain robocasa
    configure_display
    prepare_mounts robocasa
    prepare_mounts ros
    docker image inspect "$SIM_IMAGE" >/dev/null 2>&1 || fail "No $SIM_IMAGE image; build it with docker/scripts/docker_build.sh robocasa."
    docker image inspect "$ROS_IMAGE" >/dev/null 2>&1 || fail "No $ROS_IMAGE image; build it with docker/scripts/docker_build.sh ros."
    command -v flock >/dev/null || fail "flock (util-linux) is needed to share the tests/$SIM_TEST_SUITE build."
    local build=$GOLEM_CACHE_DIR/${SIM_TEST_SUITE}_tests
    mkdir -p "$build"
    exec 9>>"$build.lock"
    flock -n 9 || fail "Another tests/$SIM_TEST_SUITE test is running (it holds $build.lock); wait for it to finish."

    OUT=${out:-$GOLEM_ROOT/tests/results/$SIM_TEST_SUITE-$TEST_NAME-$(date +%Y%m%dT%H%M%S)-$$}
    mkdir -p "$OUT"
    # A run's stop file and recording would end the next one early.
    if [ -n "$(ls -A "$OUT")" ]; then
        echo "[$TAG] $OUT is not empty; use a new OUT" >&2
        exit 2
    fi
    local prefix=golem-$SIM_TEST_SUITE-$TEST_NAME-$ROS_DOMAIN_ID-$$
    SIM=$prefix-sim
    ROS=$prefix-ros
    DRIVE_RC='' REPORT_RC='' LOAD_PID=''

    local tests=$GOLEM_ROOT/tests shared workspace
    shared="      - {type: bind, source: \"$tests/g1\", target: /home/code/tests/g1, read_only: true}
      - {type: bind, source: \"$tests/sim\", target: /home/code/tests/sim, read_only: true}
      - {type: bind, source: \"$OUT\", target: /out}"
    if [ -n "$core_ws" ]; then
        workspace="      - {type: bind, source: \"$core_ws\", target: /home/code/core_ws}"
    else
        mkdir -p "$build/build" "$build/install" "$build/log"
        workspace="      - {type: bind, source: \"$build/build\", target: /home/code/core_ws/build}
      - {type: bind, source: \"$build/install\", target: /home/code/core_ws/install}
      - {type: bind, source: \"$build/log\", target: /home/code/core_ws/log}"
    fi
    cat > "$OUT/test_compose.yml" <<EOF
# This run's containers (tests/sim/common.sh), on top of the robocasa and ros services.
services:
  robocasa:
    image: "$SIM_IMAGE"
    network_mode: none
    environment:
      NVIDIA_VISIBLE_DEVICES: "${SIM_GPUS:-void}"
    volumes:
      - {type: bind, source: "$GOLEM_ROOT/golem_robocasa", target: /home/code/golem_robocasa, read_only: true}
$shared
  ros:
    image: "$ROS_IMAGE"
    network_mode: none
    volumes:
$shared
$workspace
EOF
    printf 'services:\n  ros:\n    network_mode: "container:%s"\n' "$SIM" > "$OUT/test_compose_ros_in_sim.yml"
    COMPOSE+=(-f "$OUT/test_compose.yml")
    trap sim_test_cleanup EXIT
}

# sim_test_cleanup: the EXIT trap. Remove the containers, hand OUT back to the user and write
# run.json (exit codes, sim state, domain, git state, image IDs); exit with the run's status.
sim_test_cleanup() {
    local status=$?
    trap - EXIT
    [ -z "$LOAD_PID" ] || kill "$LOAD_PID" 2>/dev/null || true
    local sim_state
    sim_state=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$SIM" 2>/dev/null || echo 'not started')
    docker rm -f "$ROS" "$ROS-build" "$SIM" >/dev/null 2>&1 || true
    # The containers write as root.
    "${COMPOSE[@]}" run --rm -T --no-deps --entrypoint chown robocasa -R "$(id -u):$(id -g)" /out \
        >/dev/null 2>&1 || true
    {
        printf '{\n  "robot": %s,\n  "test": %s,\n  "case": %s,\n' "$(sim_json_string "$SIM_TEST_ROBOT")" \
            "$(sim_json_string "$TEST_NAME")" "$(sim_json_string "$TEST_CASE")"
        printf '  "exit_code": %s,\n  "drive_exit": %s,\n  "report_exit": %s,\n  "sim": %s,\n' \
            "$status" "${DRIVE_RC:-null}" "${REPORT_RC:-null}" "$(sim_json_string "$sim_state")"
        printf '  "ros_domain_id": %s,\n  "commit": %s,\n  "submodules": %s,\n  "working_tree": %s,\n' \
            "$ROS_DOMAIN_ID" "$(sim_json_string "$(git -C "$GOLEM_ROOT" rev-parse HEAD)")" \
            "$(sim_json_string "$(git -C "$GOLEM_ROOT" submodule status --recursive 2>/dev/null)")" \
            "$(sim_json_string "$(git -C "$GOLEM_ROOT" status --short)")"
        printf '  "images": {%s: %s, %s: %s}\n}\n' \
            "$(sim_json_string "$SIM_IMAGE")" "$(sim_json_string "$(docker image inspect -f '{{.Id}}' "$SIM_IMAGE")")" \
            "$(sim_json_string "$ROS_IMAGE")" "$(sim_json_string "$(docker image inspect -f '{{.Id}}' "$ROS_IMAGE")")"
    } > "$OUT/run.json"
    echo "[$TAG] artifacts: $OUT"
    exit "$status"
}

# sim_test_log_load: log the host's load every 10 s, for the record (<out>/load.log).
sim_test_log_load() {
    ( while true; do uptime >> "$OUT/load.log"; sleep 10; done ) 9>&- &
    LOAD_PID=$!
}

# sim_test_build_ros: build core_ws before the sim starts, so that the sim never runs beside a
# whole build (<out>/ros_build.log); exit 3 if it fails.
sim_test_build_ros() {
    echo "[$TAG] building core_ws"
    if ! "${COMPOSE[@]}" run --rm -T --no-deps --name "$ROS-build" ros \
            /home/code/h12_sim_scripts/launch_ros.sh /bin/true > "$OUT/ros_build.log" 2>&1; then
        tail -20 "$OUT/ros_build.log" >&2
        echo "[$TAG] the core_ws build failed; see $OUT/ros_build.log" >&2
        exit 3
    fi
}

# sim_test_start_sim PY [ARG...]: start tests/<suite>/PY's sim role for the
# robot, headless in the layout 9 / style 9 / seed 42 kitchen, with the docker
# run ARGs (SIM_ENV), and waits until its bridges are up; its output goes to
# <out>/sim.log. The robot's own knobs that the test does not set get the sim's
# defaults: the G1 sim reads an empty GOLEM_CMD_TIMEOUT as its default (the
# H1-2 sim refuses one, so it gets none and keeps its default). The H1-2 sim
# imports magpie_msgs and custom_ros_messages, which launch_robocasa.sh builds
# into the robocasa service's msgs_ws first.
sim_test_start_sim() {
    local py=$1
    shift
    local robot_env prepare=''
    if [ "$SIM_TEST_ROBOT" = g1 ]; then
        robot_env=(-e GOLEM_CMD_TIMEOUT=)
    else
        prepare="GOLEM_PREPARE_ONLY=1 /home/code/h12_sim_scripts/launch_robocasa.sh && \
        source /home/code/msgs_ws/install/setup.bash && "
        robot_env=(-e GOLEM_SPAWN_MARGIN= -e GOLEM_SPAWN_OFFSET= -e GOLEM_LIDAR_SELF_FILTER=
                   -e GOLEM_CAMERA_DEPTH= -e GOLEM_CAMERA_POINTCLOUD= -e GOLEM_HAND_CAMERA_RES=
                   -e GOLEM_SIM_TETHER_ROT_STIFFNESS=0 -e GOLEM_SIM_TETHER_ROT_BODY=)
    fi
    "${COMPOSE[@]}" run -d --no-deps --name "$SIM" -e "GOLEM_ROBOT=$SIM_TEST_ROBOT" -e G1_DDS_INTERFACE=lo \
        -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e MUJOCO_GL=egl -e GOLEM_SIM_EXPERIMENT=0 \
        "${robot_env[@]}" "$@" robocasa bash -c "$MULTICAST && source /opt/ros/humble/setup.bash && \
        source /opt/livox_ws/install/setup.bash && $prepare cd /home/code/golem_robocasa && \
        exec python3 -u /home/code/tests/$SIM_TEST_SUITE/$py sim --out /out -- --headless --layout 9 --style 9 --seed 42" >/dev/null
    docker logs -f "$SIM" > "$OUT/sim.log" 2>&1 9>&- &
    for _ in $(seq 1 300); do
        grep -q "ROS bridges up" "$OUT/sim.log" && break
        docker ps -q --filter "name=^$SIM\$" | grep -q . || { echo "[$TAG] the sim exited" >&2; tail -20 "$OUT/sim.log" >&2; exit 1; }
        sleep 1
    done
    grep -q "ROS bridges up" "$OUT/sim.log" || { echo "[$TAG] the sim did not come up" >&2; exit 1; }
}

# sim_test_run_ros SECONDS SCRIPT: run the bash SCRIPT in the ros container, in
# the sim's network namespace with core_ws sourced, for at most SECONDS; its
# output goes to <out>/drive.log and its exit status to DRIVE_RC.
# set -m: a launch that SCRIPT starts in the background runs as its own job, so
# it gets SIGINT like Ctrl-C (a background job of a non-interactive shell would
# ignore it).
sim_test_run_ros() {
    local seconds=$1 script=$2
    set +e
    timeout "$seconds" "${COMPOSE[@]}" -f "$OUT/test_compose_ros_in_sim.yml" run --rm -T --no-deps --name "$ROS" \
        -e G1_DDS_INTERFACE=lo -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 ros bash -c "set -m
    source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash || exit 3
    $script" 2>&1 | tee "$OUT/drive.log"
    DRIVE_RC=${PIPESTATUS[0]}
    set -e
}

# sim_test_stop_sim: stop the sim with <out>/stop, which it looks for every 500
# physics steps before it saves its recording, or after 60 s with docker stop.
sim_test_stop_sim() {
    # The ros container may have made the stop file already, as root.
    [ -e "$OUT/stop" ] || touch "$OUT/stop"
    for _ in $(seq 1 60); do
        docker ps -q --filter "name=^$SIM\$" | grep -q . || break
        sleep 1
    done
    docker stop -t 30 "$SIM" >/dev/null 2>&1 || true
    [ -z "$LOAD_PID" ] || kill "$LOAD_PID" 2>/dev/null || true
    wait || true
    if [ "$DRIVE_RC" != 0 ]; then
        echo "[$TAG] the bringup or the test's steps failed (exit $DRIVE_RC); see $OUT" >&2
    fi
}

# sim_test_report PY: run tests/<suite>/PY's report role on the run (<out>/report.log and
# report.json); exit 0 only if the test's steps and every check passed, else 1.
sim_test_report() {
    set +e
    "${COMPOSE[@]}" run --rm -T --no-deps -e PYTHONDONTWRITEBYTECODE=1 robocasa \
        python3 "/home/code/tests/$SIM_TEST_SUITE/$1" report --out /out | tee "$OUT/report.log"
    REPORT_RC=${PIPESTATUS[0]}
    if [ "$DRIVE_RC" = 0 ] && [ "$REPORT_RC" = 0 ]; then
        exit 0
    fi
    exit 1
}
