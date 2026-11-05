#!/usr/bin/env bash
# In-container helpers for the launchers that must never run on the real
# robots' domain: launch_robocasa.sh, launch_isaac.sh and both Mac launchers.
# Sourced, not run. The check runs inside the container, so a direct
# `docker compose` run that skips the host wrappers is guarded too.

# SAFETY: refuse domain 0, the real robots' command bus.
# Export ROS_DOMAIN_ID (1 when unset) as a base-10 integer in 1..232; return 1
# otherwise.
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
