# golem_ros2_models

The robot models of the Unitree H1-2 (with both Magpie grippers) and the G1 (29 DoF, with both
Dex3-1 hands), installed for ROS and Pinocchio. No nodes, no joint table: each robot's joint
table is `golem_safety`'s robot profile.

## Where the files come from

`setup.py` reads the repo's `CL_Assets` submodule at `../../../CL_Assets`, so the package
builds only from `<repo>/core_ws/src/golem_ros2_models` with `CL_Assets` checked out and its
meshes pulled from Git-LFS. Every x86 `ros` container mounts `CL_Assets` at
`/home/code/CL_Assets`. The build stops with a message when:

- `CL_Assets` is missing, or a file the package installs is;
- a file it installs is still a Git-LFS pointer (run `git lfs pull` in `CL_Assets`);
- a URDF names a mesh in a form the rewrite does not cover: each rewrite must change every
  mesh reference in its file. The G1's sphere model must have none.

Generated files are staged in `_install_staging/` (gitignored), cleared on every build.

## What it installs

Under `share/golem_ros2_models/`:

| Path | What | From |
|---|---|---|
| `h12/h1_2_magpie.urdf` | the H1-2 for Pinocchio; mesh paths relative to `h12/` | `CL_Assets/ros_assets/`, `../meshes/h1_2/` → `meshes/`, `../meshes/magpie/` → `meshes/magpie/` |
| `h12/h1_2_magpie_ros.urdf` | its ROS copy, for `robot_state_publisher` and RViz: `package://golem_ros2_models/h12/meshes/...` URIs, and the 12 Magpie finger hinges fixed (nothing publishes their joint states, so as moving joints they would have no TF) | generated from `h1_2_magpie.urdf` with `rigidify_gripper` and `assert_well_formed` from `CL_Assets/ros_assets/gen_ros_urdf.py` |
| `h12/h1_2_magpie_sphere.urdf` | the H1-2's sphere-packed self-collision model | `CL_Assets/ros_assets/`, mesh paths rewritten as above |
| `h12/h1_2_magpie_sphere_collision.srdf` | the collision pairs it leaves out | `CL_Assets/ros_assets/`, verbatim |
| `h12/meshes/*.STL`, `h12/meshes/magpie/*.stl` | the body and gripper meshes | `CL_Assets/meshes/h1_2/`, `CL_Assets/meshes/magpie/` |
| `h12/LICENSE.unitree_ros` | Unitree's BSD 3-Clause licence for the H1-2 model | this package's `licenses/` |
| `g1/g1_29dof_with_hand_rev_1_0.urdf` | the G1 for Pinocchio; mesh paths relative to `g1/` | `CL_Assets/g1/g1_description/`, verbatim |
| `g1/g1_29dof_with_hand_rev_1_0_ros.urdf` | its ROS copy: `package://golem_ros2_models/g1/meshes/...` URIs, nothing else changed (the Dex3-1 joints stay articulated) | generated |
| `g1/g1_sphere.urdf` | the G1's sphere-packed self-collision model (24 spheres, no meshes) | `CL_Assets/g1/`, verbatim |
| `g1/g1_sphere_collision.srdf` | the collision pairs it leaves out | `CL_Assets/g1/`, verbatim |
| `g1/meshes/*.STL` | the G1 and Dex3-1 meshes | `CL_Assets/g1/g1_description/meshes/` |
| `g1/LICENSE.unitree_ros` | Unitree's BSD 3-Clause licence for the G1 model | `CL_Assets/g1/` |
| `launch/display.launch.py`, `rviz/model.rviz` | the viewer | this package |

`CL_Assets/README.md` says where each robot's files come from.

### The G1's sphere model covers no Dex3-1 finger

The lab packed `g1_sphere.urdf` on the body-only G1 with rubber hands: each hand is two
spheres on a `<side>_rubber_hand` link, fixed to `<side>_wrist_yaw_link` where the Dex3-1 palm
mounts. A self-collision check with it covers the palms but not the fingers. Its SRDF, unlike
the H1-2's, keeps the pelvis and hip-pitch spheres active against the torso, arms and hands.

## Python API

`golem_ros2_models.paths` (needs the package built and sourced):

- `ROBOTS = ('h12', 'g1')`.
- `model_dir(robot)`: `share/golem_ros2_models/<robot>`.
- `urdf_path(robot, ros=False)`: the Pinocchio URDF, or with `ros=True` its ROS copy.
- `collision_paths(robot)`: `(sphere URDF, SRDF)`.
- `model_paths(robot)`: `{'urdf_path', 'urdf_ros_path', 'collision'}`, as `golem_upper_ik`'s
  robot profiles name them.

Any other robot name raises `ValueError`.

## Viewer

```bash
ros2 launch golem_ros2_models display.launch.py              # the H1-2
ros2 launch golem_ros2_models display.launch.py robot:=g1    # the G1
```

It starts `robot_state_publisher` on the robot's ROS copy, `joint_state_publisher_gui` and
`rviz2`.

## Tests

```bash
colcon test --packages-select golem_ros2_models --event-handlers console_direct+
colcon test-result --verbose --test-result-base build/golem_ros2_models
```

- `test_models.py`: the installed files of both robots: the paths exist; every mesh reference
  resolves to a real mesh, not an LFS pointer; each ROS copy has its Pinocchio copy's links
  and joints; Pinocchio loads each model with its meshes and each sphere model with its SRDF;
  each sphere model's body joints match the model's.
- `test_build.py`: `setup.py`'s refusals, on a copy of the package beside a copy of
  `CL_Assets`.
- `test_display_launch.py`: the viewer's nodes and its `robot_description` for each robot.
