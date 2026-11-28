# golem_ompl

Plans collision-free arm paths for the Unitree H1-2 and G1, for any arm controller: the
`planner_server` node serves `/plan_arm_trajectory`
(`custom_ros_messages/srv/PlanArmTrajectory`). It imports `golem_upper_ik`'s robot model,
profile, IK solver and controller configs; every robot-specific value comes from there.

## The service

A request carries:

- `measured`: every body joint's measured angle. It places the joints that do not plan (the
  torso or waist, the legs) and is a pose goal's first IK seed (`'current'`).
- `start`: every arm joint at the caller's last command, the path's first waypoint. A path from
  the measured angles would step the command down to where gravity lets the arm sag.
- `position_lower`, `position_upper`: the position range the path keeps within, one value per
  `measured` joint (the caller's clip range); both empty for the URDF's limits.
- the goal: a grasp-frame pose in the pelvis frame for each arm that moves (`left_pose`,
  `right_pose`; an empty `frame_id` keeps that arm at its start), or `goal_joints`, every joint
  of each arm that moves.
- the timing: `dt` (s of robot time between waypoints, the caller's control period),
  `frame_speed` (m/s of the planner frames; 0 for the config's `planner.moving_speed`) and
  `max_acceleration` (rad/s^2; 0 for no bound).
- the budgets, s of steady time: `ik_timeout` (each IK seed's; 0 for the config's
  `controller.timeout`) and `plan_timeout` (OMPL's search; 0 for `planner.timeout`).

The response's `trajectory` holds every arm joint (the robot's `enabled_joints`, slot order),
one point per waypoint at `index * dt`, the first at the start, the arms that do not move held
at their start; `goal` is where it ends and `ik_seed` the pose goal's seed. `outcome` is
`PLANNED`, `NO_IK_SOLUTION` (no seed converged: `IK failed to resolve current targets`),
`NO_PATH` (the planner's reason) or `ERROR` (a malformed request, a robot without a
self-collision model, or a planner failure), and `message` says why. The service definition
has every field's meaning, units and allowed values.

## How it plans (`golem_ompl/arm_planner.py`)

`ArmPlanner` holds two of `golem_upper_ik`'s `RobotModel`s, built at start in the config's clip
range:

- The IK's. For each pose goal its reduced model is re-locked at the measured angles of the
  joints that do not plan, and `golem_upper_ik`'s `IKSolver` iterates from its seeds in turn:
  the measured arm angles (`'current'`), then the named configs `home`, `t_pose`,
  `arms_front_45`, in two passes (`SEED_FIRST_PASS`). A converged seed's solution is taken only
  once the planner finds a path to it; otherwise the next seed to converge is tried, and the
  last one's reason is returned once none is left.
- The planner's. `ReducedJointPlanner` (OMPL, in the IK joints) checks the joint limits,
  self-collision (the robot's sphere model) and the planner frames' x/y/z workspace
  bounds (the config's `planner` section) at every waypoint, and the workspace bounds between
  them. It tries the straight joint path first, timed along a smoothstep, then OMPL's planner,
  simplified and shortcut, and stretches either until no joint changes its velocity faster than
  `max_acceleration`. Its reduced model and self-collision model keep the joints that do not
  plan locked at 0; the measured angles place them only for the workspace bounds.

Only the joints that move the targeted grasp frames (or the named arms) plan; the other arm
keeps its start. A request in another position range narrows both models and rebuilds the
planner; the next request in the same range reuses them.

## The node (`planner_server`)

```bash
ros2 run golem_ompl planner_server --ros-args -p robot:=h12 -p config:=safety_split
ros2 launch golem_ompl planner.launch.py robot:=g1 config:=sim_split use_sim_time:=true
```

`robot` and `config` (a `golem_upper_ik` controller config by name or path, the arm
controller's own) are required. The node builds its models, then serves the service, so the
service appearing means the planner is ready (`Planner server ready (...)`). The service runs
on a callback group of its own: one plan at a time, later requests waiting their turn, the
node's other callbacks going on. A robot without a self-collision model serves the service and
refuses every request. On SIGINT or SIGTERM the process exits at once, a plan in progress with
it (`golem_upper_ik`'s `utility.run`).

`golem_bringup` starts it beside the arm controller, on the arm controller's config: the
H1-2's `h12_control.launch.py` (with `use_frame_task`) and the G1's `g1_core.launch.py`.
`golem_upper_ik`'s `frame_task_server` sends its `plan: true` goals here.

## Tests

```bash
docker/scripts/docker_run.sh ros --restart /home/code/h12_sim_scripts/launch_ros.sh bash -lc \
  'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 && cd /home/code/core_ws && source install/setup.bash && colcon test --packages-select golem_ompl --event-handlers console_direct+; colcon test-result --verbose'
```

The ROS tests run on `ROS_DOMAIN_ID` 93 (`test/conftest.py`). `test_arm_planner.py` checks, on
both robots' models, that a pose goal and a joint goal give exactly the path `golem_upper_ik`'s
controller planned in process, the seed rule, the request's timing, a new position range and a
robot without a self-collision model; `test_planner_server.py` the request parsing and its
refusals, the node's responses, and the installed console script and launch file;
`test_reduced_joint_planner.py` the planner on a two- or three-joint stand-in for the robot
model: its endpoint checks, the acceleration bound, the pinned arm and the workspace bounds.
