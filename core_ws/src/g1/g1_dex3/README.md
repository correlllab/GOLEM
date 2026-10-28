# g1_dex3

ROS 2 driver for the G1's two Dex3-1 hands. It talks to the hands directly over
Unitree DDS (`rt/dex3/{left,right}/cmd` and `state`, `unitree_hg` `HandCmd_` /
`HandState_`); the hands are not on the `rt/lowcmd` path or the safety layer.

## ROS interface

Per hand, `<side>` is `left` or `right`:

- `/<side>/hand/set_pose` (`custom_ros_messages/srv/SetHandPose`): 7 joint targets
    in `HandCmd_` order (thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0,
    index_1), rad, reached by linear interpolation over `duration` seconds, or
    longer: no joint moves faster than `max_speed`. So `duration: 0` (the
    srv's "go directly") is ramped too, e.g. 0.75 s for 1.5 rad at the default
    2 rad/s. A negative or non-finite `duration` is refused. Targets are
    clipped to the URDF limits in `g1_description`.
- `/<side>/hand/open`, `/<side>/hand/close` (`std_srvs/Trigger`): named poses,
    reached over `pose_duration` (default 1 s; the default `max_speed` never
    stretches them, from any pose inside the limits).
- `/<side>/hand/release` (`std_srvs/Trigger`): sends Unitree's stop (timeout bit,
    zero gains; the fingers go limp) and stops commanding that hand.
- `/<side>/hand/pressure` (`std_msgs/Float32MultiArray`, sensors x 12).
- `/joint_states`: the finger joints, so `robot_state_publisher` can draw the
    hands; the body joints come from `golem_upper_ik`'s `joint_state_publisher`.

```bash
ros2 launch g1_dex3 hands.launch.py   # options: on_shutdown:=silent max_speed:=1.0
ros2 service call /left/hand/close std_srvs/srv/Trigger
ros2 service call /right/hand/set_pose custom_ros_messages/srv/SetHandPose \
  "{positions: [0.0, -0.3, -0.6, 0.6, 0.6, 0.6, 0.6], duration: 1.0}"
```

## Parameters

Besides `sides`, `kp`, `kd`, `rate_hz`, `pose_duration`, `joint_state_hz` and
`network_interface` (empty: `G1_DDS_INTERFACE`):

- `domain_id` (default -1, unset): the Unitree DDS domain is `ROS_DOMAIN_ID`'s
    (0 is the real robot, 1 the sim). With `ROS_DOMAIN_ID` unset or empty, this
    parameter must give it; with neither, the node refuses to start. There is
    no fallback to domain 0. The node also refuses a `domain_id` that disagrees
    with `ROS_DOMAIN_ID`, and a Unitree domain other than the one its own ROS
    side runs on. With `ROS_DOMAIN_ID` unset, ROS 2 runs on its default domain 0,
    so in practice set `ROS_DOMAIN_ID`: without it only `domain_id:=0` starts.
    Write it in plain decimal: rcl reads a leading 0 as octal (`0141` is domain
    97), and the node refuses the mismatch. The startup log names the domain
    and interface.
- `on_shutdown` (default `release`): what a hand this node was commanding gets
    when the node exits.
    - `release`: Unitree's stop (timeout bit, zero gains), so the fingers go
      limp and drop what they hold. The default: once the driver is gone,
      nothing keeps the grip under control, so the hand is left in a known
      state.
    - `silent`: no stop; the driver just stops publishing, leaving the hand
      with its last command. What the Dex3 then does is unknown: the Unitree
      code we have read (the Dex3 examples in `unitree_ros2` and
      `unitree_sdk2`, `xr_teleoperate`'s Dex3 controller) documents no command
      timeout. The only hint is that both Dex3 examples re-send their grip only
      once a second, so a command presumably lasts at least that long. Do not
      rely on `silent` holding an object until it has been tried on our hands.
    - `on_shutdown` is applied on SIGINT, SIGTERM and SIGHUP (e.g. the
      terminal it runs in closing), however many arrive (Ctrl-C in a terminal
      reaches the node twice: from the terminal, and from `ros2 launch` passing
      it on), and the node then exits 0. SIGKILL and a crash end the node
      without it, so they always behave like `silent`.
- `max_speed` (default 2.0 rad/s): cap on any finger joint's commanded speed,
    for every move including the first. A first-session choice, below every
    URDF velocity limit (12 rad/s, thumb_0 3.14 rad/s).

## Behavior

- A hand is left alone until its first command: nothing is published to it, so
    another program driving it is not disturbed. The first command starts from
    the hand's measured pose. It needs `rt/dex3/<side>/state` from the last
    0.5 s with finite joint positions, or it is refused.
- While commanding, `HandCmd_` streams at `rate_hz` (default 100) holding the
    current target. Timing uses the monotonic clock, so a wall-clock step (NTP)
    cannot cut a move short. On shutdown, see `on_shutdown`.
- Gains start at Unitree's Dex3 example values, kp 1.5 and kd 0.1 (launch args
    `kp`, `kd`). The hand's real gains are not in any model; tune on hardware.
    Negative or non-finite gains are refused.
- The motor `mode` byte follows Unitree's example: motor index in bits 0-3,
    status 1 in bits 4-6, timeout bit 7.

## Poses

Closing directions were checked with forward kinematics on the with-hand URDF.
The hands are mirrored: the left thumb closes toward its upper limits and the
left fingers toward their lower limits, the right hand the other way.

- `open`: all zeros, fingers straight.
- `close`: fingers at 70% of their closing range, thumb_2 at 60%, thumb_1 at 50%,
    thumb_0 at 0; the thumb ends about 1 cm from the finger tips.

## Before using it on the robot

- Nothing else may be publishing `rt/dex3/{left,right}/cmd`; two publishers
    fight over the hands.
- Whether `thumb_0` mirrors between hands is still unknown on our unit.

## Tests

```bash
PYTHONPATH=.:../g1_description:$PYTHONPATH python3 -m pytest test
```

The DDS tests run on domain 97 inside one process against a fake hand; they never
touch the robot. `test_hand_node.py` needs rclpy from the sourced workspace (the
`:$PYTHONPATH` above); without it, it is skipped.
