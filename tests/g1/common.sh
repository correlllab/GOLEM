#!/usr/bin/env bash
# The containers of the G1 acceptance tests, sourced by test_sim_*.sh here.
#
# Both are the compose services as docker_run.sh --g1 starts them (the G1
# overlay with its mounts and GOLEM_ROBOT=g1), with test_compose.yml on top,
# which g1_test_setup writes into the run's output directory:
#   - the robocasa container has no network and the ros container joins its
#     network namespace, both on its loopback with multicast, so no DDS leaves
#     the host and nothing talks to the robot;
#   - the sim mounts golem_robocasa read-only;
#   - the ros container loads the run's own cyclonedds_local.xml, which pins no
#     interface, in place of the shared one in $GOLEM_CACHE_DIR, which a running
#     G1 ros container rereads in every process it starts;
#   - the ros container builds core_ws for the G1 (launch_ros.sh, GOLEM_ROBOT=g1)
#     into $GOLEM_CACHE_DIR/g1_tests/{build,install,log}, so the checkout's own
#     core_ws/build and core_ws/install are left alone; CORE_WS=<dir> builds
#     that workspace in place instead.
# The tests share that build, so one runs at a time: each holds
# $GOLEM_CACHE_DIR/g1_tests.lock on fd 9, which its background jobs close.
#
# The sim gets every knob the G1 sim reads explicitly: the test's own, the
# sensor knobs from the shell (g1_test_shell_knobs) and the sim's defaults for
# the rest, so docker/.env never changes what is tested. The H1-2-only knobs
# that the robocasa service also passes are ignored by the G1 sim.
G1_TESTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$G1_TESTS/../../docker/scripts/docker_common.sh"

# loopback multicast for DDS discovery inside the shared namespace
MULTICAST="python3 -c \"import fcntl,socket,struct; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); \
f=struct.unpack('16sH',fcntl.ioctl(s,0x8913,struct.pack('16sH',b'lo',0)))[1]; \
fcntl.ioctl(s,0x8914,struct.pack('16sH',b'lo',f|0x1000))\""

# g1_test_shell_knobs NAME...: passes each NAME to the sim as the shell has it,
# empty (the sim's default) if unset. Call it before g1_test_setup, which reads
# docker/.env.
g1_test_shell_knobs() {
    local knob
    for knob in "$@"; do
        SIM_ENV+=(-e "$knob=${!knob:-}")
    done
}

g1_json_string() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '"%s"' "$s"
}

# g1_test_setup TAG NAME CASE DOMAIN
#   TAG     the test's log prefix
#   NAME    the output directory's name: bringup, walk, nav-<odom source> or
#           explore-<odom source>
#   CASE    the variant (arm, walk, fastlio, sim_gt), recorded in run.json
#   DOMAIN  the DDS domain unless ROS_DOMAIN_ID is set; a positive integer
# Reads CORE_WS, OUT, SIM_IMAGE, ROS_IMAGE and SIM_GPUS (the test scripts'
# headers) and sets OUT, SIM, ROS and COMPOSE. On exit the containers are
# removed and OUT gets run.json.
g1_test_setup() {
    TAG=$1 TEST_NAME=$2 TEST_CASE=$3
    local domain=${ROS_DOMAIN_ID:-$4}
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
    # The containers' only interface is their loopback, which they get as
    # G1_DDS_INTERFACE below; the ros container's cyclonedds_local.xml pins none.
    export GOLEM_ROBOT=g1 G1_DDS_INTERFACE= GOLEM_DISPLAY=headless
    ROBOT_FLAG=
    load_config
    configure_robot robocasa
    check_domain robocasa
    configure_display
    prepare_mounts robocasa
    prepare_mounts ros
    docker image inspect "$SIM_IMAGE" >/dev/null 2>&1 || fail "No $SIM_IMAGE image; build it with docker/scripts/docker_build.sh robocasa."
    docker image inspect "$ROS_IMAGE" >/dev/null 2>&1 || fail "No $ROS_IMAGE image; build it with docker/scripts/docker_build.sh ros."
    command -v flock >/dev/null || fail 'flock (util-linux) is needed to share the G1 test build.'
    local build=$GOLEM_CACHE_DIR/g1_tests
    mkdir -p "$build"
    exec 9>>"$build.lock"
    flock -n 9 || fail "Another G1 acceptance test is running (it holds $build.lock); wait for it to finish."

    OUT=${out:-$GOLEM_ROOT/tests/results/g1-$TEST_NAME-$(date +%Y%m%dT%H%M%S)-$$}
    mkdir -p "$OUT"
    # a run's stop file and recording would end the next one early
    if [ -n "$(ls -A "$OUT")" ]; then
        echo "[$TAG] $OUT is not empty; use a new OUT" >&2
        exit 2
    fi
    echo '<CycloneDDS/>' > "$OUT/cyclonedds_local.xml"
    local prefix=golem-g1-$TEST_NAME-$ROS_DOMAIN_ID-$$
    SIM=$prefix-sim
    ROS=$prefix-ros
    DRIVE_RC='' REPORT_RC='' LOAD_PID=''

    local shared workspace
    shared="      - {type: bind, source: \"$G1_TESTS\", target: /home/code/tests/g1, read_only: true}
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
# This run's containers (tests/g1/common.sh), on top of the robocasa and ros services.
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
      - {type: bind, source: "$OUT/cyclonedds_local.xml", target: /home/code/cyclonedds_local.xml, read_only: true}
EOF
    printf 'services:\n  ros:\n    network_mode: "container:%s"\n' "$SIM" > "$OUT/test_compose_ros_in_sim.yml"
    COMPOSE+=(-f "$OUT/test_compose.yml")
    trap g1_test_cleanup EXIT
}

g1_test_cleanup() {
    local status=$?
    trap - EXIT
    [ -z "$LOAD_PID" ] || kill "$LOAD_PID" 2>/dev/null || true
    local sim_state
    sim_state=$(docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' "$SIM" 2>/dev/null || echo 'not started')
    docker rm -f "$ROS" "$ROS-build" "$SIM" >/dev/null 2>&1 || true
    # the containers write as root
    "${COMPOSE[@]}" run --rm -T --no-deps --entrypoint chown robocasa -R "$(id -u):$(id -g)" /out \
        >/dev/null 2>&1 || true
    {
        printf '{\n  "robot": "g1",\n  "test": %s,\n  "case": %s,\n' \
            "$(g1_json_string "$TEST_NAME")" "$(g1_json_string "$TEST_CASE")"
        printf '  "exit_code": %s,\n  "drive_exit": %s,\n  "report_exit": %s,\n  "sim": %s,\n' \
            "$status" "${DRIVE_RC:-null}" "${REPORT_RC:-null}" "$(g1_json_string "$sim_state")"
        printf '  "ros_domain_id": %s,\n  "commit": %s,\n  "submodules": %s,\n  "working_tree": %s,\n' \
            "$ROS_DOMAIN_ID" "$(g1_json_string "$(git -C "$GOLEM_ROOT" rev-parse HEAD)")" \
            "$(g1_json_string "$(git -C "$GOLEM_ROOT" submodule status --recursive 2>/dev/null)")" \
            "$(g1_json_string "$(git -C "$GOLEM_ROOT" status --short)")"
        printf '  "images": {%s: %s, %s: %s}\n}\n' \
            "$(g1_json_string "$SIM_IMAGE")" "$(g1_json_string "$(docker image inspect -f '{{.Id}}' "$SIM_IMAGE")")" \
            "$(g1_json_string "$ROS_IMAGE")" "$(g1_json_string "$(docker image inspect -f '{{.Id}}' "$ROS_IMAGE")")"
    } > "$OUT/run.json"
    echo "[$TAG] artifacts: $OUT"
    exit "$status"
}

# g1_test_log_load: the host's load every 10 s, for the record (<out>/load.log)
g1_test_log_load() {
    ( while true; do uptime >> "$OUT/load.log"; sleep 10; done ) 9>&- &
    LOAD_PID=$!
}

# g1_test_build_ros: builds core_ws for the G1 before the sim starts, so the sim
# never runs beside a whole build (<out>/ros_build.log)
g1_test_build_ros() {
    echo "[$TAG] building core_ws for the G1"
    if ! "${COMPOSE[@]}" run --rm -T --no-deps --name "$ROS-build" ros \
            /home/code/h12_sim_scripts/launch_ros.sh /bin/true > "$OUT/ros_build.log" 2>&1; then
        tail -20 "$OUT/ros_build.log" >&2
        echo "[$TAG] the core_ws build failed; see $OUT/ros_build.log" >&2
        exit 3
    fi
}

# g1_test_start_sim PY [ARG...]: starts tests/g1/PY's sim role, headless in the
# layout 9 / style 9 / seed 42 kitchen, with the docker run ARGs (SIM_ENV), and
# waits until its bridges are up; its output goes to <out>/sim.log
g1_test_start_sim() {
    local py=$1
    shift
    "${COMPOSE[@]}" run -d --no-deps --name "$SIM" -e G1_DDS_INTERFACE=lo -e PYTHONDONTWRITEBYTECODE=1 \
        -e PYTHONUNBUFFERED=1 -e MUJOCO_GL=egl -e GOLEM_CMD_TIMEOUT= -e GOLEM_ARM_SDK_TIMEOUT= \
        -e GOLEM_SIM_EXPERIMENT=0 "$@" robocasa bash -c "$MULTICAST && source /opt/ros/humble/setup.bash && \
        source /opt/livox_ws/install/setup.bash && cd /home/code/golem_robocasa && \
        exec python3 -u /home/code/tests/g1/$py sim --out /out -- --headless --layout 9 --style 9 --seed 42" >/dev/null
    docker logs -f "$SIM" > "$OUT/sim.log" 2>&1 9>&- &
    for _ in $(seq 1 300); do
        grep -q "ROS bridges up" "$OUT/sim.log" && break
        docker ps -q --filter "name=^$SIM\$" | grep -q . || { echo "[$TAG] the sim exited" >&2; tail -20 "$OUT/sim.log" >&2; exit 1; }
        sleep 1
    done
    grep -q "ROS bridges up" "$OUT/sim.log" || { echo "[$TAG] the sim did not come up" >&2; exit 1; }
}

# g1_test_run_ros SECONDS SCRIPT: runs the bash SCRIPT in the ros container, in
# the sim's network namespace with core_ws sourced, for at most SECONDS; its
# output goes to <out>/drive.log and its exit status to DRIVE_RC.
# set -m: a launch that SCRIPT starts in the background runs as its own job, so
# it gets SIGINT like Ctrl-C (a background job of a non-interactive shell would
# ignore it)
g1_test_run_ros() {
    local seconds=$1 script=$2
    set +e
    timeout "$seconds" "${COMPOSE[@]}" -f "$OUT/test_compose_ros_in_sim.yml" run --rm -T --no-deps --name "$ROS" \
        -e G1_DDS_INTERFACE=lo -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 ros bash -c "set -m
    source /opt/ros/humble/setup.bash && source /home/code/core_ws/install/setup.bash || exit 3
    $script" 2>&1 | tee "$OUT/drive.log"
    DRIVE_RC=${PIPESTATUS[0]}
    set -e
}

# g1_test_stop_sim: stops the sim with <out>/stop, which it looks for every 500
# physics steps before it saves its recording, or after 60 s with docker stop
g1_test_stop_sim() {
    # (the ros container may have made it already, as root)
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

# g1_test_report PY: tests/g1/PY's report role on the run (<out>/report.log, and
# report.json); exits 0 only if the test's steps and every check passed
g1_test_report() {
    set +e
    "${COMPOSE[@]}" run --rm -T --no-deps -e PYTHONDONTWRITEBYTECODE=1 robocasa \
        python3 "/home/code/tests/g1/$1" report --out /out | tee "$OUT/report.log"
    REPORT_RC=${PIPESTATUS[0]}
    if [ "$DRIVE_RC" = 0 ] && [ "$REPORT_RC" = 0 ]; then
        exit 0
    fi
    exit 1
}
