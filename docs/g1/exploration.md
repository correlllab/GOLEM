# Frontier exploration

The frontier explorer walks the G1 to the edges of the known map, one after
another, until nothing reachable is left unexplored. It needs navigation, and
has only run in the sim; the real bringup does not start `golem_skills` yet.

The G1 bringup starts `golem_skills` (`use_skills:=true`), which serves
`/skill/frontier_explore` (`custom_ros_messages/action/SkillFrontierExplore`)
on both robots.

## Check it

```bash
tests/g1/test_sim_explore.sh
```

The robot explores the kitchen from the nav test's spawn. A passing run ends
with `PASSED: <n>/<n> checks` and exits 0.

## Run it yourself

Start the sim as in [Navigation](navigation.md), then the bringup with skills
enabled, and, once the walking node has released the elastic band, send a goal
from a second shell in `golem_ros`:

```bash
ros2 launch golem_bringup g1_sim.launch.py use_nav:=true use_skills:=true

ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
    "{timeout: {sec: 120}}" --feedback
```

**What to expect:** the feedback shows the phase (`exploring`, `navigating`,
`done`). The result is a success when no frontier is left, or an abort at the
timeout. `goals_reached` counts goals that Nav2 reached and goals whose
frontier the lidar saw on the way there; blocked or failed goals are not
counted. The result message gives each count. A goal field left at 0 takes the
node's default (see `custom_ros_messages/action/SkillFrontierExplore.action`). The test is
the exercised path, but the shared pipeline currently fails its coverage and
stability checks; see [validation status](navigation.md#validation-status-2026-10-29).
These commands have not been run by hand.
