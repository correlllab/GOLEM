"""Select the robot, parse the command line, resolve tasks and frame the viewer for both sims.

It imports only the standard library at module level (mujoco and robocasa
inside the functions that need them): h12_mujoco.py picks the robot with it
before it imports either robot's modules.
"""
import argparse
import math
import os
import random
import sys
from collections.abc import Callable, Mapping, Sequence

ROBOTS = ("h12", "g1")
ROBOT_LABELS = {"h12": "the H1-2", "g1": "the G1"}

# The option each robot's loop does not implement; parse_args refuses it.
_UNSUPPORTED = {"h12": "--duration", "g1": "--timing"}

# The passive viewer's opening camera: straight down onto the robot wherever
# RoboCasa and the spawn placement put it, rotated with the robot's yaw so the
# robot faces the same way in the image at any spawn orientation.
VIEW_CAM_DISTANCE  = 3.0     # m above the lookat point
VIEW_CAM_AZIMUTH   = 90.0    # deg, added to the robot's yaw
VIEW_CAM_ELEVATION = -90.0   # deg, straight down
VIEW_CAM_LOOKAT_DZ = 0.0     # m, lookat offset above the robot body


def add_robot_args(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive --h12 / --g1 flags, into dest 'robot' (None when neither)."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--h12", dest="robot", action="store_const", const="h12",
                       help="run the Unitree H1-2 (the default)")
    group.add_argument("--g1", dest="robot", action="store_const", const="g1",
                       help="run the Unitree G1 (GOLEM_ROBOT=g1 selects it too)")


def select_robot(argv: Sequence[str], env: Mapping[str, str] = os.environ) -> str:
    """Return the robot to run: --h12 or --g1 in argv, else GOLEM_ROBOT, else 'h12'.

    GOLEM_ROBOT is h12 or g1 in any case; empty counts as unset. Exits with an
    error when argv has both flags or GOLEM_ROBOT names another robot. Only
    whole tokens match; parse_args refuses an abbreviated flag.
    """
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
    """Add the options both runners take, the robot flags included.

    task_help is the --task help of a runner with tasks of its own.
    """
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
    """Parse argv for the runner of robot ('h12' or 'g1').

    Exits with a usage error (2) for a robot flag that reads as the other robot
    (h12_mujoco.py dispatches whole --h12/--g1 tokens, so only an abbreviation
    such as --g gets here) and for the option in _UNSUPPORTED for this robot.
    """
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
    """Exit with a usage error (2) for --layout or --style with a task in no_kitchen.

    Those options choose a RoboCasa kitchen, and such a task builds none.
    """
    if task in no_kitchen:
        given = [flag for flag in ("--layout", "--style") if getattr(args, flag[2:]) is not None]
        if given:
            verb = "choose" if len(given) > 1 else "chooses"
            parser.error(f"{' and '.join(given)} {verb} a RoboCasa kitchen; "
                         f"task {task!r} has none")


def runnable_tasks() -> tuple[dict[str, type], set[str]]:
    """Return (env_classes, runnable): every registered env by name, and the runnable tasks.

    env_classes is robosuite's REGISTERED_ENVS, which robosuite.make() resolves
    against. It includes RoboCasa's abstract bases (OpenDoor, ManipulateDoor,
    PickPlace, ...), which RoboCasa leaves out of its own REGISTERED_KITCHEN_ENVS,
    so that resolve_task can expand a base to its concrete subclasses.

    runnable is RoboCasa's ATOMIC_ and COMPOSITE_TASK_DATASETS that are
    registered, else REGISTERED_KITCHEN_ENVS when those lists are not
    importable. Neither holds an abstract base: those need a fixture_id only a
    concrete subclass supplies.
    """
    import robocasa  # noqa: F401  # registers every kitchen env
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
    """Return a random runnable RoboCasa kitchen task, reproducible for a seed."""
    _, runnable = runnable_tasks()
    if seed is not None:
        random.seed(seed)
    return random.choice(sorted(runnable))


def resolve_task(name: str, seed: int | None = None, *, tag: str,
                 robot_task: Callable[[str], str | None] = lambda name: None) -> str:
    """Return the runnable task a --task name stands for.

    robot_task(name) returns the canonical name of a task the runner provides
    outside RoboCasa (Blank, Battery), else None; that name is used as is. A
    runnable RoboCasa task is returned unchanged, and an abstract base (OpenDoor,
    ...) becomes one of its runnable subclasses, chosen at random (seeded by
    seed). Exits for an unknown name. Messages are prefixed [tag].
    """
    claimed = robot_task(name)
    if claimed is not None:
        return claimed
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
    """Point the passive viewer's free camera down at body_id, turned with its yaw (VIEW_CAM_*)."""
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
