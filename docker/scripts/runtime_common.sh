#!/usr/bin/env bash
# Simulator-side guard also applies when users invoke Compose directly.
validate_sim_domain() {
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    case "$ROS_DOMAIN_ID" in ''|*[!0-9]*) echo 'Invalid simulation ROS_DOMAIN_ID' >&2; return 1;; esac
    [ "${#ROS_DOMAIN_ID}" -le 3 ] || return 1
    ROS_DOMAIN_ID=$((10#$ROS_DOMAIN_ID))
    if [ "$ROS_DOMAIN_ID" -lt 1 ] || [ "$ROS_DOMAIN_ID" -gt 232 ]; then
        echo 'Simulation ROS_DOMAIN_ID must be 1..232; 0 is reserved for the real robot.' >&2
        return 1
    fi
}
