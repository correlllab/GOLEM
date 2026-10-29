# Frontier exploration

The frontier explorer walks the G1 to the edges of the known map, one after
another, until nothing reachable is left unexplored. It needs navigation, so
it is **sim only**.

On the G1 it is `g1_navigation`'s `frontier_explore` node. It serves
`/skill/frontier_explore` with the same action type as the H1-2's skills node
(`custom_ros_messages/action/SkillFrontierExplore`), so callers do not change.
`h12_skills` does not run on the G1; never run both explorers on one DDS domain.

## Check it

```bash
tests/g1/test_sim_explore.sh
GOLEM_SIM_ODOM=1 tests/g1/test_sim_explore.sh    # with the sim's ground-truth odometry
```

The robot explores the kitchen from the nav test's spawn. A passing run ends
with `PASSED: <n>/<n> checks` and exits 0. The g1_navigation README's
"Exploring in the G1 sim" section has the checks and the measured runs.

## Run it yourself

Start the sim as in [Navigation](navigation.md), then the bringup with the
explorer, and, once the walking node has released the elastic band, send a goal
from a second shell in `golem_ros`:

```bash
ros2 launch golem_bringup g1_sim.launch.py use_nav:=true frontier_explore:=true

ros2 action send_goal /skill/frontier_explore custom_ros_messages/action/SkillFrontierExplore \
    "{timeout: {sec: 120}}" --feedback
```

**What to expect:** the feedback shows the phase (`exploring`, `navigating`,
`done`). The result is a success when no frontier is left, or an abort at the
timeout. `goals_reached` counts goals that Nav2 reached and goals whose
frontier the lidar saw on the way there; blocked or failed goals are not
counted. The result message gives each count. A goal field left at 0 takes the
node's default (the action definition in `custom_ros_messages` and the
g1_navigation README list them). The test is
the tested path; these commands have not been run by hand.

More: the Frontier exploration section of the
[g1_navigation README](https://github.com/correlllab/GOLEM/blob/main/core_ws/src/g1_navigation/README.md#frontier-exploration).
