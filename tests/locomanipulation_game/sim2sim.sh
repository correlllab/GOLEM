#!/usr/bin/env bash
# The walking sim-to-sim test (sim2sim_walk.py) in a disposable robocasa container: the deploy controller on the
# compiled scene a RoboCasa run recorded, without ROS. Artifacts go to tests/results/sim2sim_walk-<name>-<time>/.
#
#   tests/locomanipulation_game/sim2sim.sh <name> <RoboCasa result dir> [sim2sim_walk.py args...]
#   e.g. tests/locomanipulation_game/sim2sim.sh variants tests/results/locomanipulation_game-runM_train_s5-* \
#            --variants robocasa,isaac_joints,stiff_contact,implicit_pd
#   The run's directory is mounted at /source: --replay /source/ros/commander.jsonl replays its events.
#
# LMG_POLICY_DIR: another export directory (networks, export.yaml, marl_golem.yaml, arm goals), mounted at /policy
# in place of the deploy package's policies/marl_golem. Needs no DDS: the container runs with ROS_DOMAIN_ID 194
# and publishes nothing.
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NAME=${1:?Usage: sim2sim.sh <name> <RoboCasa result dir> [args...]}
SOURCE=$(cd "${2:?Usage: sim2sim.sh <name> <RoboCasa result dir> [args...]}" && pwd)
shift 2
[[ -s "$SOURCE/record/model.mjb" ]] || { echo "no record/model.mjb in $SOURCE" >&2; exit 2; }
export ROS_DOMAIN_ID=194
source "$ROOT/docker/scripts/docker_common.sh"
load_config
export GOLEM_DISPLAY=headless
configure_display
prepare_mounts robocasa
policy=()
if [[ -n "${LMG_POLICY_DIR:-}" ]]; then
    POLICY_DIR=$(cd "$LMG_POLICY_DIR" && pwd)
    policy=(-v "$POLICY_DIR:/policy:ro")
    set -- --policy /policy/marl_golem.yaml "$@"
fi
result="$ROOT/tests/results/sim2sim_walk-$NAME-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$result"
chmod 777 "$result"
git -C "$ROOT" rev-parse HEAD >"$result/commit.txt" 2>/dev/null || true
git -C "$ROOT" status --short >>"$result/commit.txt" 2>/dev/null || true
echo "$SOURCE" >"$result/source.txt"
"${COMPOSE[@]}" run -T --rm --no-deps -e ROS_DOMAIN_ID=194 \
    -v "$ROOT/tests/locomanipulation_game:/home/code/tests/locomanipulation_game:ro" \
    -v "$ROOT/core_ws/src/lmg_deploy:/home/code/lmg_deploy:ro" \
    -v "$ROOT/core_ws/src/h12_safety_layer:/home/code/h12_safety_layer:ro" \
    -v "$SOURCE:/source:ro" -v "$result:/result" "${policy[@]}" \
    --entrypoint python3 robocasa /home/code/tests/locomanipulation_game/sim2sim_walk.py \
    --model /source/record/model.mjb --out /result "$@" < /dev/null 2>&1 | tee "$result/sim2sim.log"
echo "Artifacts: $result"
