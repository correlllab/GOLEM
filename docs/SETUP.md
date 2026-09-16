# Setup (x86 / Linux + NVIDIA)

One-time setup for the preferred platform. For Apple Silicon see
[`MACOS.md`](MACOS.md).

## Prerequisites

- Docker with Compose v2, and the NVIDIA Container Toolkit.
- Git LFS — the URDF meshes, MuJoCo XML, and Isaac USD assets are LFS objects.

## 1. Clone with submodules and assets

Most of the tree is git submodules; builds fail cryptically without them.

```bash
git lfs install
git submodule update --init --recursive
```

## 2. Configure `docker/.env`

Runtime settings live in `docker/.env`, which is git-ignored because it holds
your API key. Copy the template once and edit it — nothing needs to be exported
in your shell:

```bash
cp docker/.env.example docker/.env
# GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
# ROS_DOMAIN_ID=1      # any non-zero value; 0 is reserved for the real robot
```

Both `docker compose` and `docker/scripts/docker_run.sh` load the file, so every
container and terminal sees the same values. `ROS_DOMAIN_ID` reaches all
containers; `GEMINI_API_KEY` reaches only the `ros` container, where the vision
servers and `h12_skills` need it. The wrappers load the file and preserve exported shell overrides, matching
Compose precedence. Optional `GOLEM_ASSETS_DIR` and `GOLEM_CACHE_DIR` relocate
host mounts without editing YAML.

**Domain 0 is the real robot's command bus.** Unset/empty defaults to `1`;
explicit zero is rejected for simulations. Linux ROS-only launches on zero
require confirmation.

## 3. SAM3 weights

The `sam_server` segmentation model is not tracked in git. Download it from the
gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) repo:

```bash
huggingface-cli login                          # one-time; accept the license
cd core_ws/src/model_server/weights
huggingface-cli download facebook/sam3 --local-dir .cache
mv .cache/sam3.pt sam3.pt                      # the name must be exactly sam3.pt
```

`core_ws/src/model_server/weights/README.md` covers the auto-download fallback.

## 4. Build the images

```bash
docker/scripts/docker_build.sh                 # all three profiles
docker/scripts/docker_build.sh robocasa ros    # a subset
docker/scripts/docker_build.sh isaac           # isaac only
```

`robocasa` and `ros` inherit from `golem_base`, which is built first
automatically whenever either is selected. `isaac` is self-contained.

The build/run scripts resolve their own location, so they can be invoked from
any directory.

For how the images are layered, what is pinned and why, see
[`../docker/BUILD.md`](../docker/BUILD.md).

## Upgrading a checkout from before the HAMS → GOLEM rename

The rename was a clean break, with no compatibility fallbacks:

- **Env vars:** every `HAMS_*` toggle is now `GOLEM_*`. The run scripts abort if
  they see a stale `HAMS_*` export, because these toggles default to off and
  would otherwise be ignored silently. The guard cannot fire if you bypass the
  scripts with `docker compose … up`, where the stale variable is simply dropped.
- **Containers/images:** `hams_base`/`hams_ros`/`hams_sim_*` are now
  `golem_base`/`golem_ros`/`golem_sim_*`. Retag rather than rebuild:
  `for i in base ros sim_robocasa; do docker tag hams_$i:latest golem_$i:latest; done`
- **Checkouts:** the robot PC's checkout is `/home/unitree/GOLEM`. Set
  `GOLEM_ASSETS_DIR` to override it if yours is not renamed yet.
- **Remote:** `git remote set-url origin git@github.com:correlllab/GOLEM.git`.

For paired startup and the two-machine smoke test, see [Docker testing](DOCKER_TESTING.md).
