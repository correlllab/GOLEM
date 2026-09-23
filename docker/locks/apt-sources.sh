#!/bin/sh
# Dated, signed package archives; deliberately updated alongside dependency locks.
set -eu
. /etc/os-release
snapshot=https://snapshot.ubuntu.com/ubuntu/20260917T000000Z
# Ubuntu's snapshot includes both amd64 and arm64 archives.
cat > /etc/apt/sources.list <<EOF
deb $snapshot $VERSION_CODENAME main restricted universe multiverse
deb $snapshot $VERSION_CODENAME-updates main restricted universe multiverse
deb $snapshot $VERSION_CODENAME-security main restricted universe multiverse
EOF
# Noble images may use Deb822 Ubuntu sources instead of sources.list.
for source in /etc/apt/sources.list.d/ubuntu.sources; do
    [ ! -f "$source" ] || mv "$source" "$source.disabled"
done
# Vendor CUDA repositories are live and undated. The images use only the CUDA
# packages already present in their pinned bases, so resolve against snapshots.
for source in /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list; do
    [ ! -e "$source" ] || mv "$source" "$source.disabled"
done
# Fixed snapshots expire their freshness window; signatures remain mandatory.
printf '%s\n' 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99golem-snapshot
# Replace ROS sources (including modern Deb822 inline signing keys) as a unit.
for source in /etc/apt/sources.list.d/ros*.list /etc/apt/sources.list.d/ros*.sources; do
    [ ! -e "$source" ] || mv "$source" "$source.disabled"
done
case "$VERSION_CODENAME" in
    jammy) archive=http://snapshots.ros.org/humble/2026-08-07/ubuntu ;;
    noble) archive=http://snapshots.ros.org/jazzy/2026-09-11/ubuntu ;;
    *) echo "Unsupported Ubuntu release: $VERSION_CODENAME" >&2; exit 1 ;;
esac
printf 'deb [signed-by=/opt/golem-locks/ros-snapshot.asc] %s %s main\n' "$archive" "$VERSION_CODENAME" > /etc/apt/sources.list.d/ros2.list
