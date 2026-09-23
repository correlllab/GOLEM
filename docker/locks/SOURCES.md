# Build input provenance

Source and external image pins were recorded on 2026-09-17. Updating a pin is
an explicit dependency change that requires rebuilding and validating the
affected images. The build now selects dated Ubuntu and ROS archives, exact source revisions,
external base digests, and per-install Python constraints. Downloaded kitchen
assets are verified against a complete-tree digest. This is dependency-version
reproducibility, not a claim of byte-identical OCI layers: timestamps, compiler
outputs, and isolated Python build backends require separate verification.

## Existing x86 images used as evidence

| Image | Local image ID at audit |
| --- | --- |
| golem_base | 85410e75af53 |
| golem_ros | d1cef5abb357 |
| golem_sim_robocasa | 53b0719cdd36 |
| golem_sim_isaac | cabe3c218720 |

Inspected with `docker compose run --rm --no-deps --entrypoint bash` through the
shared `docker_common.sh` configuration. No simulation was launched. Git SHAs
came from each image's retained `.git` directory; IsaacLab extras came from
`pip freeze` VCS metadata. uv's version was `0.12.15` in the base and its children.
The macOS Dockerfiles reuse the same source revisions; no arm64 built image was
available for validation.

| Source | Revision | Evidence |
| --- | --- | --- |
| CycloneDDS | 5041f3560c088c99e5088b2b8520b69169621196 | All three x86 child images |
| SAM3 | 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7 | ROS image |
| GraspGenX | b9429097728cb1c430dd78b92edf17ba318aad03 | ROS image |
| robosuite | 5ce6643f3092639d08f7b0f90ed1c6a84f50552c | RoboCasa image |
| robocasa | 4f8a2980def75a55dff96b990745b83540425f09 | RoboCasa image |
| robosuite_models | 2d63a6b4319bda772420daf10594ce0de5bd793d | RoboCasa image |
| mimicgen | 72bd767c255545f462e7ccfb2731f2e5d4c1d9bb | RoboCasa image |
| Unitree Python SDK (Isaac) | 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5 | Isaac image |
| rl_games (IsaacLab extra) | 6b3534f29568158e9e29ec8bf83cc88fce5f0cae | Isaac pip VCS metadata |
| robomimic (IsaacLab extra) | 7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9 | Isaac pip VCS metadata |
| Livox-SDK2 | 08f523c930b2f0ba1e98a6afaa8d7476bf479908 | Upstream HEAD; all three public header SHA256s match installed ROS image headers |
| rosdistro signing key | e31287f892ecd01b170d1cbfa298db35abe4a74f | Upstream rosdistro HEAD |

Livox's build deletes the clone, so matching headers is evidence of interface
compatibility, not proof the SDK binary was built from this exact revision.
The protected `livox_ros_driver2` gitlink remains unchanged at `13eb05e`.
Existing explicit CLIP, IsaacLab, MJPC and Unitree C++ revisions remain unchanged.
The base SDK COPY still uses the repository's Unitree Python submodule checkout.

CUDA and Isaac base digests come from locally installed images' `RepoDigests`.
Humble ROS and Ubuntu 22.04 digests were resolved from registry manifest indexes;
the latter retains platform selection for arm64. These two registry pins need
fresh build validation. Local `golem_base` tags intentionally refer to the base
built by the repository wrapper, not to a remotely published digest.

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

The Ubuntu snapshot is HTTPS-only (HTTP redirects to HTTPS). The bare arm64
`ubuntu:22.04` base has no CA bundle, so `mac/BaseDockerfile.arm64` copies
`/etc/ssl/certs/ca-certificates.crt` from the digest-pinned multi-arch
`ros:humble-ros-base-jammy` index before running `apt-sources.sh`. TLS peer
verification stays enabled. The bootstrap was probe-built on amd64 from the same
two indexes; a native arm64 build has not been run.

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
MuJoCo, and build tools). The arm64 sets reuse portable versions without CUDA
packages; native Mac build/runtime validation remains required.

A fresh resolver exposed incompatible packages in the old installed ROS image:
NumPy 1.26 coexisted with cmeel-boost 1.90 and OpenCV 5, which require NumPy 2.
The ROS locks now use Pinocchio 2.7 / Boost 1.83 / eigenpy 3.5.1 / hpp-fcl 2.4.4
and OpenCV 4.10.0.84. They must be validated together after rebuilding; this is
an intentional dependency correction, not a claim those versions were all in
the old image. MuJoCo remains 3.2.3 for ROS and 3.3.1 for RoboCasa.
The ROS image first uninstalls the base image's whole Pinocchio/cmeel closure;
otherwise the `--ignore-installed` reinstall left cmeel-boost 1.83 and 1.90
(and two urdfdom/assimp versions) installed side by side.

The old Isaac image was likewise inconsistent (`pip check`): stable-baselines3
2.9.0 requires torch>=2.8 beside Isaac Sim's bundled torch 2.7.0+cu128;
rl-games 1.6.1 requires psutil<6 beside psutil 7.2.2; opencv-python 5.0
requires NumPy 2. The Isaac locks keep Isaac Sim's bundled torch, NumPy 1.26
and psutil 5.9.8, and select stable-baselines3 2.8.0 (newest accepting
torch>=2.3), ipython 9.10.1 (9.13+ requires psutil>=7; 9.11-9.12 require
Python 3.12), opencv-python 4.11.0.86 (matching the bundled headless build),
and pandas 2.2.3 with tzdata 2025.2, which stable-baselines3 2.8 requires.

`robocasa-assets.digest` hashes the sorted `sha256sum` listing of every file
under `/opt/robocasa/robocasa/models/assets` (123,434 files at capture). This
catches changed contents, additions and removals without committing a large
file manifest. Updating assets requires explicit review and regenerating the
digest from the approved tree, not accepting a mismatch during a normal build.
