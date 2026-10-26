#!/usr/bin/env bash
# Records or checks the RoboCasa sim's ROS/DDS topic surface (see README.md).
#   run.sh h12|g1            compare against <robot>.json (GOLDEN=<path> overrides it)
#   run.sh h12|g1 --record   rewrite the golden
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
usage() { echo 'Usage: [GOLDEN=<path>] run.sh h12|g1 [--record]' >&2; exit 2; }
ROBOT=${1:-}
case "$ROBOT" in h12|g1) ;; *) usage ;; esac
RECORD=0
case "${2:-}" in '') ;; --record) RECORD=1 ;; *) usage ;; esac
[ $# -le 2 ] || usage
SURFACE=$ROOT/tests/robocasa/surface
GOLDEN=$(realpath -m "${GOLDEN:-$SURFACE/$ROBOT.json}")
case "$ROBOT" in
    h12) sim_args=(--headless --task AirDryFruit --seed 42) ;;
    g1)  sim_args=(--headless --layout 9 --style 9 --seed 42) ;;
esac
# The probe needs a domain with nothing else on it, so it does not use the shell's.
export ROS_DOMAIN_ID=194
# The golden describes the default sim: a knob set in the shell or docker/.env
# must not change what is recorded or checked. Empty means the sim's default
# (docker-compose.yml and docker-compose.g1.yml list what the service passes on).
for knob in GOLEM_CAMERAS GOLEM_CAMERA_SCALE GOLEM_CAMERA_DEPTH GOLEM_CAMERA_POINTCLOUD \
            GOLEM_CAMERA_HZ GOLEM_HAND_CAMERA_RES GOLEM_LIDAR GOLEM_LIDAR_POINTS GOLEM_LIDAR_HZ \
            GOLEM_SIM_ODOM GOLEM_SIM_EXPERIMENT GOLEM_SIM_TETHER_ROT_STIFFNESS \
            GOLEM_SIM_TETHER_ROT_BODY GOLEM_SPAWN_MARGIN GOLEM_SPAWN_BACKOFF GOLEM_SPAWN_OFFSET \
            GOLEM_CMD_TIMEOUT GOLEM_ARM_SDK_TIMEOUT; do
    export "$knob="
done
source "$ROOT/docker/scripts/docker_common.sh"
# A robot chosen in the shell or docker/.env must not apply; the argument decides.
export GOLEM_ROBOT=$ROBOT
load_config
[ "$ROBOT" = h12 ] || configure_robot robocasa
check_domain robocasa
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts robocasa
[ "$RECORD" = 1 ] || [ -f "$GOLDEN" ] || fail "No golden at $GOLDEN; record one with: tests/robocasa/surface/run.sh $ROBOT --record"

result="$ROOT/tests/results/surface-$ROBOT-$(date +%Y%m%dT%H%M%S)-$$"
mkdir -p "$result"
# Root in Docker writes the artifacts; let ordinary host users inspect and remove them.
chmod 777 "$result"
container="golem-surface-$ROBOT-$ROS_DOMAIN_ID-$$"
status=0
cleanup() {
    local code=$?
    trap - EXIT
    docker rm -f "$container" >/dev/null 2>&1 || true
    printf '{"robot": "%s", "record": %s, "exit_code": %s, "ros_domain_id": %s, "golden": "%s", "commit": "%s", "tree_dirty": %s, "image": "%s"}\n' \
        "$ROBOT" "$([ "$RECORD" = 1 ] && echo true || echo false)" "$code" "$ROS_DOMAIN_ID" "$GOLDEN" \
        "$(git -C "$ROOT" rev-parse HEAD)" "$([ -n "$(git -C "$ROOT" status --short)" ] && echo true || echo false)" \
        "$(docker image inspect -f '{{.Id}}' golem_sim_robocasa:latest 2>/dev/null || true)" > "$result/run.json"
    echo "Surface probe artifacts: $result"
    exit "$code"
}
trap cleanup EXIT

probe_args=(--robot "$ROBOT" --out /results/observed.json --sim-log /results/sim.log)
mounts=(-v "$SURFACE:/home/code/tests/robocasa/surface:ro" -v "$result:/results")
if [ "$RECORD" = 1 ]; then
    probe_args+=(--record)
else
    probe_args+=(--golden "/golden/$(basename "$GOLDEN")")
    mounts+=(-v "$GOLDEN:/golden/$(basename "$GOLDEN"):ro")
fi
probe_args+=(-- /home/code/h12_sim_scripts/launch_robocasa.sh "${sim_args[@]}")

echo "[surface] $ROBOT on DDS domain $ROS_DOMAIN_ID, golden $GOLDEN"
# The launcher builds the message workspace on every start. Doing that once up
# front (it exits after the build) lets the probe import the Magpie messages
# before the sim, which it then starts itself, comes up.
"${COMPOSE[@]}" run --rm -T --no-deps --name "$container" "${mounts[@]}" -e MUJOCO_GL=egl robocasa bash -lc \
    'source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash &&
     GOLEM_PREPARE_ONLY=1 /home/code/h12_sim_scripts/launch_robocasa.sh &&
     source /home/code/msgs_ws/install/setup.bash &&
     exec python3 -u /home/code/tests/robocasa/surface/probe.py "$@"' bash "${probe_args[@]}" \
    2>&1 | tee "$result/probe.log" || status=$?

if [ "$status" = 0 ] && [ "$RECORD" = 1 ]; then
    cp "$result/observed.json" "$GOLDEN"
    echo "[surface] recorded $GOLDEN"
fi
exit "$status"
