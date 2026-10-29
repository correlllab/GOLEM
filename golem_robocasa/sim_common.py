"""What the H1-2 and G1 sims share: the robot selection, the command line, the
task listing and resolution, and the viewer framing.

Stdlib only at import time (mujoco and robocasa are imported inside the
functions that need them): h12_mujoco.py picks the robot with it before it
imports either robot's modules.
"""
import argparse
import math
import os
import random
import sys
from collections.abc import Callable, Mapping, Sequence

ROBOTS = ("h12", "g1")
ROBOT_LABELS = {"h12": "the H1-2", "g1": "the G1"}

# Options a robot's loop does not implement; parse_args refuses them for that robot.
_UNSUPPORTED = {"h12": "--duration", "g1": "--timing"}

# Tasks only the H1-2 runs. Its battery_env registers them, and the G1 process never
# imports that module, so resolve_task names them by this list.
H12_ONLY_TASKS = ("Battery",)

# Viewer free-camera spawn pose, anchored to the robot so the passive viewer
# opens framed on the robot wherever it spawns (RoboCasa places the base by
# layout/seed and place_robot_collision_free may back it off), instead of
# MuJoCo's default whole-scene framing. Top-down (straight down) view: lookat
# follows the robot base body, elevation looks straight down, and azimuth (the
# in-plane rotation of the top-down image) is offset from the base yaw so the
# robot's facing points a consistent way in frame at any spawn orientation.
# Spawn-time only — the user can orbit/zoom freely after.
VIEW_CAM_DISTANCE  = 3.0     # m, camera height above lookat (straight down)
VIEW_CAM_AZIMUTH   = 90.0    # deg, ADDED to robot yaw — rotates the top-down image
VIEW_CAM_ELEVATION = -90.0   # deg, -90 = look straight down
VIEW_CAM_LOOKAT_DZ = 0.0     # m, focal point offset along z (irrelevant for top-down)


def add_robot_args(parser: argparse.ArgumentParser) -> None:
    """--h12 / --g1, mutually exclusive, into dest 'robot' (None when neither is given)."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--h12", dest="robot", action="store_const", const="h12",
                       help="run the Unitree H1-2 (the default)")
    group.add_argument("--g1", dest="robot", action="store_const", const="g1",
                       help="run the Unitree G1 (GOLEM_ROBOT=g1 selects it too)")


def select_robot(argv: Sequence[str], env: Mapping[str, str] = os.environ) -> str:
    """The robot to run: --h12 or --g1 in argv, else GOLEM_ROBOT (h12 or g1, any
    case; empty counts as unset), else 'h12'. A flag beats the environment.
    Exits with an error when argv has both flags or GOLEM_ROBOT names another robot.
    Matches whole tokens only; parse_args catches an abbreviated flag."""
    flags = {arg[2:] for arg in argv if arg in ("--h12", "--g1")}
    if len(flags) > 1:
        sys.exit("Choose one robot: --h12 or --g1.")
    if flags:
        return flags.pop()
    robot = env.get("GOLEM_ROBOT", "").strip().lower()
    if not robot:
        return "h12"
    if robot not in ROBOTS:
        sys.exit(f"GOLEM_ROBOT must be {' or '.join(ROBOTS)}, not {robot!r}.")
    return robot


def add_common_args(parser: argparse.ArgumentParser, *,
                    task_help: str = "RoboCasa env name (e.g. TurnOnToasterOven). "
                                     "Omit to launch a random registered kitchen task.") -> None:
    """The options both runners take: --headless, --task, --layout, --style, --seed,
    --timing, --duration, and the robot pair (add_robot_args). task_help replaces
    the --task description for a runner that has tasks of its own."""
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo passive viewer (default: viewer on)",
    )
    parser.add_argument("--task", default=None, help=task_help)
    parser.add_argument("--layout", type=int, default=None, help="RoboCasa kitchen layout id")
    parser.add_argument("--style", type=int, default=None, help="RoboCasa kitchen style id")
    parser.add_argument("--seed", type=int, default=None, help="episode seed")
    parser.add_argument("--timing", action="store_true",
                        help="Log real-time factor and stage mean/max timings every 5 seconds")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="sim seconds to run from the first step, then exit "
                             "(default 0: run until closed; G1 only)")
    add_robot_args(parser)


def parse_args(parser: argparse.ArgumentParser, argv: Sequence[str], robot: str) -> argparse.Namespace:
    """Parse argv for the runner of `robot` ('h12' or 'g1'), with a usage error (exit 2) for:
    - a robot flag naming the other robot. h12_mujoco.py hands whole --h12/--g1 tokens to
      the matching runner, so only an abbreviation (--g for --g1) gets here;
    - an option this robot's loop does not implement yet (--duration for the H1-2,
      --timing for the G1)."""
    args = parser.parse_args(argv)
    if args.robot not in (None, robot):
        parser.error(f"the robot flag is abbreviated and reads as --{args.robot} here, "
                     f"but this runs {ROBOT_LABELS[robot]}; spell the robot flag out: --{args.robot}")
    flag = _UNSUPPORTED[robot]
    if getattr(args, flag[2:]):
        parser.error(f"{flag} is not supported for {ROBOT_LABELS[robot]} yet")
    return args


def refuse_kitchen_args(parser: argparse.ArgumentParser, args: argparse.Namespace,
                        task: str, no_kitchen: Sequence[str]) -> None:
    """Usage error (exit 2) when --layout or --style is given for a task in
    `no_kitchen` (resolved names): those options choose a RoboCasa kitchen, and
    such a task builds none."""
    if task in no_kitchen:
        given = [flag for flag in ("--layout", "--style") if getattr(args, flag[2:]) is not None]
        if given:
            verb = "choose" if len(given) > 1 else "chooses"
            parser.error(f"{' and '.join(given)} {verb} a RoboCasa kitchen; "
                         f"task {task!r} has none")


def runnable_tasks() -> tuple[dict[str, type], set[str]]:
    """Return (env_classes, runnable).

    `env_classes` maps every robosuite-registered env name -> its class. This is
    the same REGISTERED_ENVS that robosuite.make() resolves against, and it DOES
    include RoboCasa's abstract bases (OpenDoor/CloseDoor/ManipulateDoor/
    PickPlace) — RoboCasa's metaclass deliberately omits those five from its own
    REGISTERED_KITCHEN_ENVS, but the robosuite parent metaclass still registers
    them. We need them here so an abstract base name can be looked up and expanded
    to its concrete subclasses.

    `runnable` is the set of concrete benchmark task names from RoboCasa's curated
    ATOMIC_/COMPOSITE_TASK_DATASETS (intersected with what's registered, to
    tolerate version skew). Those bases can't be instantiated directly — they
    require a `fixture_id` only a concrete subclass supplies — so they are never
    in `runnable`. Falls back to REGISTERED_KITCHEN_ENVS (which already excludes
    the five bases) if the curated lists are unavailable.
    """
    import robocasa  # noqa: F401  -> registers all kitchen envs on import
    from robosuite.environments.base import REGISTERED_ENVS
    env_classes = dict(REGISTERED_ENVS)
    try:
        from robocasa.utils.dataset_registry import (
            ATOMIC_TASK_DATASETS,
            COMPOSITE_TASK_DATASETS,
        )
        runnable = (set(ATOMIC_TASK_DATASETS) | set(COMPOSITE_TASK_DATASETS)) & set(env_classes)
    except Exception:
        from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
        runnable = set(REGISTERED_KITCHEN_ENVS) & set(env_classes)
    return env_classes, runnable


def random_task(seed: int | None = None) -> str:
    """Pick a random *runnable* (concrete) RoboCasa kitchen env name."""
    _, runnable = runnable_tasks()
    if seed is not None:
        random.seed(seed)
    return random.choice(sorted(runnable))


def resolve_task(name: str, seed: int | None = None, *, tag: str,
                 robot_task: Callable[[str], str | None] = lambda name: None) -> str:
    """Resolve a user-supplied --task name to a concrete runnable task.

    `robot_task(name)` returns the canonical name of a task the calling robot runs
    outside RoboCasa (blank_env's Blank, and the H1-2's battery workcell, matched
    in any case), else None; a name it returns is used as is. A name in
    H12_ONLY_TASKS that it does not claim exits with an error. If `name` is
    already a concrete benchmark task, return it unchanged.
    If it's an abstract base (e.g. OpenDoor, ManipulateDoor) — registered but not
    directly instantiable — randomly pick one of its concrete runnable subclasses
    instead of crashing. Messages are prefixed `[tag]`.
    """
    claimed = robot_task(name)
    if claimed is not None:
        return claimed
    if name.lower() in (task.lower() for task in H12_ONLY_TASKS):
        raise SystemExit(f"[{tag}] task {name!r} is H1-2 only")
    env_classes, runnable = runnable_tasks()
    if name in runnable:
        return name
    if name not in env_classes:
        raise SystemExit(f"[{tag}] unknown RoboCasa task {name!r}")
    base = env_classes[name]
    concrete = sorted(
        n for n in runnable
        if isinstance(env_classes.get(n), type)
        and issubclass(env_classes[n], base) and env_classes[n] is not base
    )
    if not concrete:
        raise SystemExit(f"[{tag}] {name!r} is abstract with no runnable concrete tasks")
    if seed is not None:
        random.seed(seed)
    choice = random.choice(concrete)
    print(f"[{tag}] {name!r} is an abstract task base; "
          f"randomly selected concrete subclass {choice!r} "
          f"(from {len(concrete)} options)")
    return choice


def frame_viewer_on_robot(handle, data, body_id: int) -> None:
    """Set the passive viewer free camera's spawn pose as a function of the robot
    base body's world pose. lookat tracks the body position; the orbit azimuth is
    offset from the body's yaw, so framing is invariant to spawn orientation."""
    import mujoco
    pos = data.xpos[body_id]
    w, x, y, z = (float(v) for v in data.xquat[body_id])
    yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    cam = handle.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (float(pos[0]), float(pos[1]), float(pos[2]) + VIEW_CAM_LOOKAT_DZ)
    cam.distance = VIEW_CAM_DISTANCE
    cam.azimuth = yaw_deg + VIEW_CAM_AZIMUTH
    cam.elevation = VIEW_CAM_ELEVATION
