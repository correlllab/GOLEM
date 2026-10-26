"""What the H1-2 and G1 sims share: the robot selection.

Stdlib only at import time: h12_mujoco.py picks the robot with it before it
imports either robot's modules.
"""
import argparse
import os
import sys
from collections.abc import Mapping, Sequence

ROBOTS = ("h12", "g1")


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
    Exits with an error when argv has both flags or GOLEM_ROBOT names another robot."""
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
