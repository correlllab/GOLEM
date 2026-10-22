# Common commands

Handy `ros2 action send_goal` snippets for driving the H1-2 stack. Run these
from a shell **inside `golem_ros`** with the workspace sourced:

```bash
docker exec -it golem_ros /bin/bash
source /opt/ros/humble/setup.bash
source /home/code/core_ws/install/setup.bash
```

Quick sanity checks:

```bash
ros2 topic hz /lowstate          # sim/robot is alive and publishing state
ros2 action list | grep -E 'skill|frame_task|named_config'
```

**Timeouts:** every `/skill/*` goal takes a `timeout` Duration. **Omit it (or send
zero) and the skill runs to completion** — it then ends only by finishing,
failing, or being canceled, never on a clock. Give a non-zero `timeout` when you
actually want a bound. Ctrl-C on `send_goal` cancels the in-flight goal.

---

## Real robot: pick up a screwdriver by its handle

Two machines on `ROS_DOMAIN_ID=0`, run natively (no Docker). Every terminal on
both machines first runs:

```bash
export ROS_DOMAIN_ID=0
source ~/GOLEM/core_ws/install/setup.bash
```

**1. Onboard PC — launch** (robot on the gantry, legs hanging free):

```bash
ros2 launch h12_bringup h12_real_robot_bringup.launch.py start_position_verified:=true
```

The leg controller logs `waiting for /safety/heartbeat before pre-posing the
legs`, ramps the legs into the ALMI crouch, then holds and logs
`legs pre-posed (settled, err=...) — holding`. If it reports `NOT SETTLED` or
the legs jump, Ctrl-C and do not engage.

**2. Onboard PC, second terminal — engage** (feet loaded, gantry slack):

```bash
ros2 service call /lowerbody/confirm_engage std_srvs/srv/Trigger
```

**3. Desktop — perception and skills:**

```bash
ros2 launch h12_bringup h12_real_desktop_bringup.launch.py
# add use_hand_cameras:=true when the grasp uses visual_servo
```

Wait for `h12_skills` to report ready; it looks for frame_task, the grippers
and the model servers on startup.

**4. Grasp.** Lay the screwdriver flat in reach of the arm, shaft pointing
straight away from the robot. `top_down` closes the fingers along pelvis ±Y
whatever the object's orientation, so this puts them across the handle. Name
the **handle**, not the tool: the grasp lands on the centroid of what Gemini +
SAM segment, and for the whole screwdriver that sits near the handle–shaft
joint. Pick an arm whose gripper node is running (`ros2 node list | grep
gripper`).

```bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'screwdriver handle', arm: 'right', top_down: false, timeout: {sec: 180, nanosec: 0}}" \
  --feedback
```

Drop `top_down: true` to try the heuristic side approaches instead.

**5. Lift.** The grasp skill closes the gripper and stops. Raise the hand
straight up from where it closed with a relative `frame_task` goal. With
`relative: true` each target is an offset from the frame's current pose: the
position is added in pelvis axes (`z: 0.05` = 5 cm up), and an identity
orientation (`w: 1.0`, not all zeros) keeps the wrist as it is. Use
`left_graspgenx_frame` for the left arm.

```bash
ros2 action send_goal /frame_task custom_ros_messages/action/FrameTask \
  "{relative: true, plan: false, frame_names: ['right_graspgenx_frame'],
    frame_targets: [{position: {x: 0.0, y: 0.0, z: 0.05},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}],
    duration: {sec: 3, nanosec: 0}}" \
  --feedback
```

Then carry the handle back by raising the arm to `prep`:

```bash
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'prep', plan: true, slow_mode: true, duration: {sec: 30, nanosec: 0}}" \
  --feedback
```

Ctrl-C on a `send_goal` cancels that arm motion; the e-stop stops everything.

---

## Frontier exploration (`/skill/frontier_explore`)

Autonomous frontier-based exploration over the slam_toolbox `/map`. Needs nav2 +
slam_toolbox up and the robot in **walk mode** so nav2's `/cmd_vel` is followed.
All numeric goal fields accept `0` to fall back to the node default.

```bash
ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
  "{min_frontier_cells: 6, blacklist_radius: 0.4, min_goal_distance: 0.3, goal_timeout: 120.0, timeout: {sec: 3000, nanosec: 0}}" \
  --feedback
```

Field reference:

| Field | Meaning | Default (0 →) |
|---|---|---|
| `min_frontier_cells` | ignore frontier clusters smaller than this | 6 |
| `blacklist_radius` (m) | blacklist a failed goal within this radius | 0.6 |
| `min_goal_distance` (m) | skip frontiers closer than this to the robot | 0.7 |
| `goal_timeout` (s) | cancel a single nav goal after this | 120 |
| `timeout` (Duration) | overall exploration budget | node default |

---

## Grasp (`/skill/grasp`)

Pick up a named object. Does **not** lift after grasping — the skill computes the
approach and grasp width itself. `arm` is `"left"` or `"right"`; `""` (or
`"none"`) auto-selects the arm currently closest to the object.

```bash
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'vertical fridge handle', arm: 'right', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

```bash
# left arm, tabletop object
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'mug', arm: 'left', timeout: {sec: 60, nanosec: 0}}" \
  --feedback
```

```bash
# top-down + visual servo: auto arm, steep from above (battery-workcell part).
# No timeout -> runs to completion; cancel with Ctrl-C to stop it.
ros2 action send_goal /skill/grasp custom_ros_messages/action/SkillGrasp \
  "{target_object: 'screw', arm: '', top_down: true, visual_servo: true}" \
  --feedback
```

Optional goal fields (omit / `false` / `0` = off):

| Field | Meaning |
|---|---|
| `top_down` | use the `top_down` synthesis method for this goal, overriding `grasp_method` (below): ONE synthetic grasp above the object — the tier-1 orientation (forward azimuth, fingers across pelvis ±Y, level wrist) pitched `TOP_DOWN_PITCH_DEG` = 80° below horizontal, base backed 0.1146 m off along that approach so the fingers close on the cloud centroid. Still steeper than the arm's ~45° practical reach, and there is no second candidate to fall back to, so an unreachable pose aborts the skill. |
| `min_downward_pitch_deg` | every synthesis method keeps only grasps approaching at least this far below horizontal (a *minimum* pitch, not proximity to vertical) |
| `visual_servo` | at the pre-grasp, close the loop on the arm's hand camera: translate (orientation frozen) until the object sits at a fixed setpoint in the camera optical frame — `VISUAL_SERVO_TARGET_XY_M` = (+12, 0) mm laterally, `VISUAL_SERVO_DEPTH_M` = 0.10 m in range — then close **there** instead of running the contact descent. Tolerances are **per axis**, because the camera axes mean different things to a parallel gripper: 1 mm across the jaws (optical X — what a grasp depends on), 2 mm in range (optical Z — whether the jaws close around the part or above it), 5 mm along the jaws (optical Y — the jaws span 106 mm, so this is the forgiving one). 90 s budget (~50 corrections), and it gives up early if the error stops improving. If it loses sight of the object for 5 s, **or** reads it closer than 5.5 cm on two consecutive frames, it backs straight up 5 cm (pelvis +Z) and keeps servoing — up to twice between the two cases. If it still can't converge the skill **aborts** — there is no open-loop fallback, so the result tells you the grasp was never aligned. The 7 YOLO battery classes are boxed by the hand YOLO detector; any other object is boxed by Gemini on the hand frame (seconds per correction). Both need `yolo_server` running, which supplies the hand depth. |

**Synthesis method.** Without `top_down`, the grasp candidates come from the
skills node's `grasp_method` parameter (default `heuristic`). Every method
gets the same segmented cloud and runs through the same approach / contact /
servo executor, so switching methods changes only the grasps:

```bash
ros2 param set /h12_skills grasp_method antipodal      # takes effect on the next goal
ros2 param describe /h12_skills grasp_method
```

The method names and what each does are in
`h12_skills/h12_skills/grasp_synthesis/` (registry: `METHODS` in its
`__init__.py`); an unknown name is rejected by `ros2 param set`.
`visual_servo` composes with any of them.

Every target goes through Gemini + SAM, on the head camera and in the visual
servo — the pipeline the sim grasp tables measure. A battery-workcell YOLO class
label as `target_object` (e.g. `OrageCover`) is sent to Gemini as a readable
phrase (`GEMINI_NAMES` in `skills/grasp.py`); a descriptive name such as
`'orange cover with a pink dot'` is used as given. `BATTERY_USE_GEMINI = False`
in `skills/grasp.py` routes those labels to the head/hand YOLO detectors and the
raw bounding-box cloud instead. The hand YOLO stream (`use_yolo:=true`) is still
needed for the servo's aligned depth and intrinsics.

The full sequence runs: pre-grasp standoff → contact move (`APPROACH_DIST −` the
`contact_offset` parameter, default 2.5 cm, = 7.5 cm along the approach) → force close at
`base.GRIP_FORCE_N`. It does **not** lift afterwards.

---

## Pick and place (`/skill/pick_place`)

Grasp a named object, carry it, place it on a named destination and release.
Internally sends its own `/skill/grasp` goal, so everything the grasp section
above says about detection routing still applies to `target_object`. Like
`/skill/grasp`, `arm` is optional — `""`, `"none"`, or omitting it entirely
lets the grasp skill pick the arm nearest the object, and the carry/place/release
motions then follow whichever hand it chose.

```bash
# auto arm, top-down + visual servo pick of a battery-workcell screw onto a plate.
# No timeout -> runs to completion (see the budget note below); Ctrl-C cancels.
ros2 action send_goal /skill/pick_place custom_ros_messages/action/SkillPickPlace \
  "{target_object: 'Screw', place_target: 'plate', top_down: true, visual_servo: true, arm: 'right'}" \
  --feedback
```

```bash
# explicit right arm, graspgen candidates (no orientation bias)
ros2 action send_goal /skill/pick_place custom_ros_messages/action/SkillPickPlace \
  "{target_object: 'mug', place_target: 'sink', arm: 'right', timeout: {sec: 300, nanosec: 0}}" \
  --feedback
```

Optional goal fields (omit / `false` = off):

| Field | Meaning |
|---|---|
| `arm` | `""` / `"none"` / omitted = auto: the grasp skill takes the arm whose gripper is currently closest to the object. Anything other than `left`/`right`/empty aborts the goal. |
| `top_down` | forwarded verbatim to the inner `/skill/grasp` — same 80°-below-horizontal synthetic grasp, same reachability caveat as the grasp table above. **Also changes the place:** instead of lowering the object onto the target, the graspgenx frame goes to `TOP_DOWN_PLACE_HEIGHT` = 25 cm straight above the place point and releases, so the object is dropped the remaining ~14 cm (the fingers sit ~11 cm below that frame). Off by default: the normal place descent is vertical and grasp-agnostic. |
| `visual_servo` | forwarded verbatim to the inner `/skill/grasp` — same hand-camera alignment and the same abort-on-failure behavior. Off by default. ⚠️ Budget: the servo alone can take 90 s, and pick_place spends its Gemini place-target detect *before* the grasp, so prefer omitting `timeout` (run to completion) when using this. |

The `place_target` is located with Gemini + SAM (not YOLO, whatever
`target_object` routes to) and its **top face** becomes the drop point, anchored
in the `camera_init` world frame so pelvis drift during the long grasp doesn't
move it. The object lands in its pick-time orientation.

Sequence: detect place target → grasp → lift 10 cm in 5 cm hops → transit to a
10 cm standoff above the release pose → vertical descent → release 3 cm above
rest height → back the fingers out 15 cm along the reverse approach.

With `top_down: true` the release pose changes only in *where* it is — the
graspgenx frame goes 25 cm above the place point rather than to a computed
set-down pose. Everything else (lift, standoff, descent, retreat) is unchanged.

**Budget:** the last 60 s of `timeout` (`PLACE_RESERVE_SEC`) are reserved for
carry/place/release, so a slow Gemini detect can't starve the placement half —
the inner grasp gets everything else. Omit `timeout` to run to completion; the
reserve is then moot (there is no finite budget to split) and the inner grasp is
handed "no deadline" too.

⚠️ With `visual_servo`, a finite `timeout` is easy to under-set. The place-target
Gemini detect runs BEFORE the grasp and can take ~3 min on its own, then 60 s
comes off for the reserve — so a 300 s goal can leave the inner grasp less than
the servo's own 90 s budget. The servo then gets cut off by the goal deadline and
aborts, which reads as a servo failure when it is really budget starvation.

---

## Named arm configs (`/named_config`)

Drive the 14 arm joints to a predefined posture (defined in
`h12_ros2_controller/utility/named_config.py`). `plan: false` (or omitted) servos
directly; `plan: true` runs the planner first. `duration` of `0` uses the
controller default settle time.

The available configs are the keys of `NAMED_CONFIGS` in that file. The grasp
skill itself uses `prep` (arms clear of the head camera) and
`staging_left` / `staging_right` (pre-approach posture for the grasping arm).

```bash
# arms down to the home baseline
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{config_name: 'home', duration: {sec: 0, nanosec: 0}}" \
  --feedback
```

```bash
# T-pose, planned, over 4 s
ros2 action send_goal /named_config custom_ros_messages/action/NamedConfig \
  "{plan: true, config_name: 't_pose', duration: {sec: 4, nanosec: 0}}" \
  --feedback
```

---

## Frame task / arm IK (`/frame_task`)

Command one or more end-effector frames to a Cartesian target. Targets are
`geometry_msgs/Pose` in the **`pelvis`** frame; orientation is a quaternion
(`w: 1.0` = identity). `frame_names` and `frame_targets` must be the same length.
Common frames: `left_wrist_yaw_link`, `right_wrist_yaw_link`.

```bash
# move the right wrist to a point in front of the robot (pelvis frame)
ros2 action send_goal /frame_task custom_ros_messages/action/FrameTask \
  "{plan: false,
    frame_names: ['right_wrist_yaw_link'],
    frame_targets: [{position: {x: 0.30, y: -0.20, z: 0.10},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}],
    duration: {sec: 3, nanosec: 0}}" \
  --feedback
```

```bash
# dual-arm: both wrists to symmetric targets, planned
ros2 action send_goal /frame_task custom_ros_messages/action/FrameTask \
  "{plan: true,
    frame_names: ['left_wrist_yaw_link', 'right_wrist_yaw_link'],
    frame_targets: [{position: {x: 0.24, y: 0.20, z: 0.09},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}},
                    {position: {x: 0.24, y: -0.20, z: 0.09},
                     orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}],
    duration: {sec: 4, nanosec: 0}}" \
  --feedback
```

---

## Loco-manipulation game policy (`/marl_deploy_node`)

The Locomanipulation_game MARL policy (`lmg_deploy`) drives all 27
joints: legs walk on `/cmd_vel`, arms reach wrist-pose goals (held in the world
by the SLAM odometry, so keep `use_slam` on). You drive it yourself (see
`lmg_deploy/README.md` for all launch arguments):

```bash
ros2 launch h12_bringup lmg_sim_bringup.launch.py policy:=<name>   # a policies/<name>/ of lmg_deploy
```

Walk:

```bash
ros2 topic echo /marl_deploy_node/state --once                 # expect "nav" once engaged
ros2 run teleop_twist_keyboard teleop_twist_keyboard           # walk
```

To reach, then go back to walking:

```bash
ros2 topic pub --once /marl_deploy_node/arm_goal geometry_msgs/msg/PoseArray \
  "{poses: [{position: {x: 0.30, y: 0.25, z: 0.05}, orientation: {w: 1.0}},
            {position: {x: 0.30, y: -0.25, z: 0.05}, orientation: {w: 1.0}}]}"
ros2 topic echo /marl_deploy_node/wrist_error                  # metres, left and right
ros2 service call /marl_deploy_node/end_arm_goal std_srvs/srv/Trigger
```

Arm goals are `left_wrist_yaw_link` then `right_wrist_yaw_link` poses in the
pelvis frame; `/cmd_vel` is ignored until the goal ends.
