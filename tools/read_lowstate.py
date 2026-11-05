"""Print the real G1's rt/lowstate once a second; it only subscribes, so it commands nothing.

In the ros container (docker/scripts/docker_run.sh ros), which mounts
this directory at /home/code/tools:

    python3 /home/code/tools/read_lowstate.py

Outside the containers, in the Python 3.10 environment of docs/g1/dds.md
(Reading the robot without the stack):

    conda activate g1
    python tools/read_lowstate.py

It reads domain 0, always, on the laptop's port to the robot: the interface
is G1_DDS_INTERFACE, default enp130s0. It runs until interrupted.

MIT License, Copyright (c) 2026 Correll Robotics Lab.
"""

import os
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Motor slot -> joint name, in the order G1JointIndex uses in the SDK's
# example/g1/low_level/g1_low_level_example.py. Slots 29-34 are unused.
# On 23-DoF and locked-waist units slots 13-14 carry no motor; on 23-DoF
# units 20, 21, 27 and 28 don't either.
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

latest = None


def on_lowstate(msg: LowState_):
    """Keep the latest rt/lowstate sample; runs on a DDS thread."""
    global latest
    latest = msg


# Domain 0 is the real robot's bus; this script only subscribes on it.
ChannelFactoryInitialize(0, os.environ.get("G1_DDS_INTERFACE") or "enp130s0")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_lowstate, 10)

while True:
    time.sleep(1)
    if latest is None:
        print("no data yet")
        continue
    s = latest
    print("mode_machine:", s.mode_machine, " mode_pr:", s.mode_pr, " tick:", s.tick)
    print("imu rpy:", [round(a, 3) for a in s.imu_state.rpy])
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i:2d}  {name:22s} q = {s.motor_state[i].q:7.3f} rad")
    print()
