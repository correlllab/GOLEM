# Build input provenance

Updating a pin is an explicit dependency change that requires rebuilding and
validating the affected images. The build selects dated Ubuntu and ROS archives,
exact source revisions, external base digests, and per-install Python
constraints. Downloaded kitchen assets are verified against a complete-tree
digest. This is dependency-version reproducibility, not a claim of
byte-identical OCI layers: timestamps, compiler outputs, and isolated Python
build backends require separate verification.

## Source revisions

| Source | Revision | Image |
| --- | --- | --- |
| CycloneDDS | 5041f3560c088c99e5088b2b8520b69169621196 | golem_base, Isaac |
| SAM3 | 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7 | ROS image |
| GraspGenX | b9429097728cb1c430dd78b92edf17ba318aad03 | ROS image |
| robosuite | 5ce6643f3092639d08f7b0f90ed1c6a84f50552c | RoboCasa image |
| robocasa | 4f8a2980def75a55dff96b990745b83540425f09 | RoboCasa image |
| robosuite_models | 2d63a6b4319bda772420daf10594ce0de5bd793d | RoboCasa image |
| mimicgen | 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb | RoboCasa image |
| Unitree Python SDK (Isaac) | 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5 | Isaac image |
| rl_games (IsaacLab extra) | 6b3534f29568158e9e29ec8bf83cc88fce5f0cae | Isaac pip VCS metadata |
| robomimic (IsaacLab extra) | 7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9 | Isaac pip VCS metadata |
| Livox-SDK2 | 08f523c930b2f0ba1e98a6afaa8d7476bf479908 | ROS, RoboCasa (one commit: `test_every_image_builds_livox_sdk2_at_one_commit`) |
| rosdistro signing key | e31287f892ecd01b170d1cbfa298db35abe4a74f | Isaac (downloaded, referenced by no apt source) |

The protected `livox_ros_driver2` gitlink stays at `13eb05e` (`CLAUDE.md` §8).
CLIP, GraspGenX's and IsaacLab's revisions are pinned in their Dockerfile
steps. The base image's Unitree Python SDK is the repository's submodule
checkout.

CUDA and Isaac base digests come from locally installed images' `RepoDigests`.
The Humble ROS digest (Isaac's `magpie_humble` stage) was resolved from its
registry manifest index and needs fresh build validation. Local `golem_base`
tags intentionally refer to the base built by the repository wrapper, not to a
remotely published digest.

Guard against regressing source pins with:

```sh
python3 -m unittest discover -s tests/docker -p test_source_pins.py
```


## Package archives and Python transitions

Ubuntu uses the `20260917T000000Z` snapshot; ROS Humble uses `2026-08-07` and
Jazzy uses `2026-09-11`. `apt-sources.sh` configures those archives and retains
signature verification. The vendored ROS snapshot key fingerprint is
`4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA`; it differs from the live ROS key.
It was obtained from Ubuntu's keyserver and checked against the ROS snapshot
maintainers' documented fingerprint. Expired Release metadata freshness checks
are disabled for historical snapshots; package signatures are not disabled.

The Ubuntu snapshot is HTTPS-only (HTTP redirects to HTTPS); TLS peer
verification stays enabled.

`apt-sources.sh` also disables inherited `cuda*`/`nvidia*` APT lists, which
are live and undated. No build step installs from them; CUDA libraries come
from the digest-pinned base images.

Runtime constraints are passed with `-c` and do not reach pip or uv isolated
build environments, so sdist build backends (setuptools, scikit-build-core,
setuptools-scm, ...) still resolve to current releases. Applying the runtime
locks there would be wrong: the ROS image clamps `setuptools==59.6.0` at runtime
while sdists such as GraspGenX build with `setuptools>=64`. Locking backends
needs separate build-constraint files captured per install step.

References: [Ubuntu snapshots](https://snapshot.ubuntu.com/),
[ROS snapshot key](https://github.com/osrf/docker_images/blob/master/ros/noetic/ubuntu/focal/ros-core/Dockerfile).

Each `*-amd64.txt` is the common Python constraint set. Numbered files include
that set and declare intentional per-install transitions (NumPy, Hugging Face,
MuJoCo, and build tools).

The G1 stack's Python dependency is the ROS image's last install step,
before its final setuptools/wheel/numpy clamp (which reuses
`ros-amd64-06.txt`), and MuJoCo 3.17.0 the last step of the RoboCasa image:

| Step file | Image | Direct pins | Other pins added to the common set |
| --- | --- | --- | --- |
| `ros-amd64-11.txt` | golem_ros | `onnxruntime==1.23.2` | `coloredlogs==15.0.1`, `flatbuffers==25.12.19`, `humanfriendly==10.0` |
| `robocasa-amd64-05.txt` | golem_sim_robocasa | `mujoco==3.17.0` | `websockets==16.1.1` |

Neither step moves an already-locked package (NumPy, protobuf, sympy, ...).

The ROS locks keep NumPy 1.26 for apt's scipy and pinocchio, which fixes the
kinematics closure at Pinocchio 2.7 / Boost 1.83 / eigenpy 3.5.1 / hpp-fcl
2.4.4 and OpenCV at 4.10.0.84. The ROS image first uninstalls the base image's
whole Pinocchio/cmeel closure (NumPy 2, cmeel-boost 1.90); the
`--ignore-installed` reinstall would otherwise leave two cmeel-boost,
urdfdom and assimp versions side by side
(`test_ros_removes_base_kinematics_closure_before_reinstall`). MuJoCo in the
ROS image is golem_base's; the RoboCasa image installs 3.3.1 (RoboCasa's pin)
and then 3.17.0 over it (`robocasa-amd64-05.txt`).

The Isaac locks keep Isaac Sim's bundled torch 2.7.0+cu128, NumPy 1.26 and
psutil 5.9.8 (rl-games needs psutil<6), and select stable-baselines3 2.8.0
(newest accepting torch>=2.3), ipython 9.10.1 (9.13+ requires psutil>=7;
9.11-9.12 require Python 3.12), opencv-python 4.11.0.86 (matching the bundled
headless build), and pandas 2.2.3 with tzdata 2025.2, which stable-baselines3
2.8 requires (`test_isaac_locks_stay_compatible_with_bundled_packages`).

`robocasa-assets.digest` hashes the sorted `sha256sum` listing of every file
under `/opt/robocasa/robocasa/models/assets` (123,434 files at capture). This
catches changed contents, additions and removals without committing a large
file manifest. Updating assets requires explicit review and regenerating the
digest from the approved tree, not accepting a mismatch during a normal build.
