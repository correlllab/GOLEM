#!/bin/bash
set -euo pipefail
HERE="$(/usr/bin/dirname -- "$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}")")"
if [[ $# == 0 || ${1:-} == --help || ${1:-} == -h ]]; then
    exec /usr/bin/env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
        /usr/bin/python3 "$HERE/environment/run_ros.py" --help
fi
domain="${ROS_DOMAIN_ID:-1}"
if [[ ! $domain =~ ^[0-9]{1,3}$ ]] || (( 10#$domain == 0 || 10#$domain > 232 )); then
    printf '%s\n' 'error: ROS_DOMAIN_ID must be 1..232; domain 0 is the real robot bus.' >&2
    exit 1
fi
export ROS_DOMAIN_ID="$((10#$domain))"
source "$HERE/environment/clean_shell.sh"
golem_clean_python runtime "$HERE/environment/run_ros.py" "$@"
