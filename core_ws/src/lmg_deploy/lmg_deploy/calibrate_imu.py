"""Estimate the H1-2 torso IMU's orientation calibration (roll, pitch) from a ROS 2 bag of /lowstate.

Record the robot standing still with both feet flat on level ground (the pre-pose on the floor, lowered from the
hoist: lmg_real's waiting_confirm), then, offline:

    ros2 run lmg_deploy calibrate_imu <bag folder> [--urdf .../h1_2_magpie.urdf]

For every /lowstate sample (every --stride-th) where the legs and the torso gyro are still and both knees carry
weight (imu_calibration.StillThresholds), in runs of at least --min-still-s, forward kinematics from each sole
(ankle roll link, level at zero joint angles) to imu_link gives imu_link's tilt if that sole is level; the
quaternion's tilt against it is the calibration (imu_calibration.calibrate). Each foot gives its own estimate:
their disagreement is the quality check (a foot on its edge, or uneven ground, shows up there). Yaw is unobservable
this way and stays 0. The result goes in config/imu_calibration.yaml under the robot's name, with the date, the
method and the bags' numbers.

Reading the bag: rosbag2_py inside the ROS environment, or the rosbags library elsewhere (--msg-dir: unitree_hg's
msg folder). The kinematics needs pinocchio.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from lmg_deploy.imu_calibration import (
    FEET,
    IMU_FRAME,
    StillThresholds,
    calibrate,
    fk_imu_gravity,
    still_and_loaded,
    still_windows,
)

PKG = Path(__file__).resolve().parent.parent
NUM_JOINTS = 27
FEET_TOL_DEG = 0.5   # deg: the two feet's estimates further apart than this fail the quality check


def read_lowstate(bag: str, msg_dir: str | None = None, topic: str = "/lowstate") -> dict[str, np.ndarray]:
    """Return a bag's /lowstate as arrays: t (s), q, dq, tau (N, 27), quat (N, 4 wxyz), gyro (N, 3)."""
    rows = {k: [] for k in ("t", "q", "dq", "tau", "quat", "gyro")}

    def add(stamp_ns, msg):
        motors = msg.motor_state[:NUM_JOINTS]
        rows["t"].append(stamp_ns * 1e-9)
        rows["q"].append([m.q for m in motors])
        rows["dq"].append([m.dq for m in motors])
        rows["tau"].append([m.tau_est for m in motors])
        rows["quat"].append(list(msg.imu_state.quaternion))
        rows["gyro"].append(list(msg.imu_state.gyroscope))

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from unitree_hg.msg import LowState
    except ImportError:
        rosbag2_py = None
    if rosbag2_py is not None:
        storage = "mcap" if any(p.suffix == ".mcap" for p in Path(bag).iterdir()) else "sqlite3"
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
        reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
        while reader.has_next():
            _, raw, stamp = reader.read_next()
            add(stamp, deserialize_message(raw, LowState))
    else:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
        if not msg_dir:
            raise SystemExit("without rosbag2_py, --msg-dir must name unitree_hg's msg folder (for the rosbags library)")
        store = get_typestore(Stores.ROS2_HUMBLE)
        types = {}
        for path in Path(msg_dir).glob("*.msg"):
            types.update(get_types_from_msg(path.read_text(), f"unitree_hg/msg/{path.stem}"))
        store.register(types)
        with AnyReader([Path(bag)], default_typestore=store) as reader:
            for conn, stamp, raw in reader.messages(connections=[c for c in reader.connections if c.topic == topic]):
                add(stamp, reader.deserialize(raw, conn.msgtype))
    if not rows["t"]:
        raise SystemExit(f"no {topic} in {bag}")
    out = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    out["t"] -= out["t"][0]
    return out


def default_config() -> str:
    """Return the shipped H1-2 deploy config: the source tree's, else the installed package's."""
    path = PKG / "policies" / "h12" / "h12.yaml"
    if path.is_file():
        return str(path)
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("lmg_deploy"), "policies", "h12", "h12.yaml")


def kinematics(urdf: str, config: str):
    """Return ArmKinematics on the policy's joint order with the soles and imu_link among its frames."""
    from lmg_deploy.kinematics import ArmKinematics
    with open(config) as f:
        cfg = yaml.safe_load(f)
    names = cfg["joint_names"]
    return ArmKinematics(urdf, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                         extra_frames=[IMU_FRAME, *FEET])


def fk_gravity(kin, q: np.ndarray) -> np.ndarray:
    """Return (len(FEET), 3) imu_link's projected gravity at joint angles q by kinematics from each sole, level."""
    kin.update(q)
    return fk_imu_gravity(kin)


def run(data: dict, kin, th: StillThresholds, min_still_s: float, stride: int) -> dict:
    """Return the calibration from a /lowstate recording (read_lowstate), and the still windows used."""
    d = {k: v[::stride] for k, v in data.items()}
    keep = still_windows(d["t"], still_and_loaded(d["dq"], d["gyro"], d["tau"], th), min_s=min_still_s)
    if not keep.any():
        raise SystemExit("no still window with both feet loaded: nothing to calibrate from (thresholds: "
                         f"{th}, at least {min_still_s} s)")
    g = np.array([fk_gravity(kin, q) for q in d["q"][keep]])
    result = calibrate(g[:, 0], g[:, 1], d["quat"][keep])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
    result["windows_s"] = [[round(float(d["t"][a]), 2), round(float(d["t"][b - 1]), 2)]
                           for a, b in zip(edges[::2], edges[1::2])]
    result["stride"] = stride
    return result


def main(argv: list[str] | None = None) -> int:
    """Print each bag's calibration estimate (and, with --json, write them)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("bags", nargs="+", help="rosbag2 folders with /lowstate")
    p.add_argument("--urdf", default=os.environ.get("H12_URDF", "/home/code/CL_Assets/ros_assets/h1_2_magpie.urdf"))
    p.add_argument("--config", help="a deploy config, for the joint order and wrist frames (default: the shipped "
                   "policies/h12/h12.yaml)")
    p.add_argument("--msg-dir", help="unitree_hg's msg folder, for the rosbags library (outside ROS)")
    p.add_argument("--stride", type=int, default=5, help="use every n-th /lowstate sample (500 Hz)")
    p.add_argument("--min-still-s", type=float, default=1.0)
    p.add_argument("--max-leg-dq", type=float, default=StillThresholds.max_leg_dq)
    p.add_argument("--max-gyro", type=float, default=StillThresholds.max_gyro)
    p.add_argument("--knee-load-tau", type=float, default=StillThresholds.knee_load_tau)
    p.add_argument("--json", help="write the results here")
    args = p.parse_args(argv)
    th = StillThresholds(args.max_leg_dq, args.max_gyro, args.knee_load_tau)
    kin = kinematics(args.urdf, args.config or default_config())
    results = {}
    for bag in args.bags:
        r = run(read_lowstate(bag, args.msg_dir), kin, th, args.min_still_s, args.stride)
        results[os.path.basename(os.path.normpath(bag))] = r
        print(f"{os.path.basename(os.path.normpath(bag))}: roll {r['roll_deg']:+.3f} pitch {r['pitch_deg']:+.3f} deg "
              f"from {r['samples']} still samples (windows {r['windows_s']} s)\n"
              f"  left foot  roll {r['left'][0]:+.3f} pitch {r['left'][1]:+.3f}\n"
              f"  right foot roll {r['right'][0]:+.3f} pitch {r['right'][1]:+.3f}"
              f"  (disagreement roll {r['feet_disagreement'][0]:.3f}, pitch {r['feet_disagreement'][1]:.3f})\n"
              f"  imu_link tilt by kinematics roll {r['fk_tilt'][0]:+.3f} pitch {r['fk_tilt'][1]:+.3f}; "
              f"the quaternion reads roll {r['imu_tilt'][0]:+.3f} pitch {r['imu_tilt'][1]:+.3f}")
        if max(r["feet_disagreement"]) > FEET_TOL_DEG:
            print(f"  WARNING: the feet disagree by more than {FEET_TOL_DEG} deg: the soles are not both flat on level "
                  "ground by kinematics (a foot on its edge, uneven floor, the hoist carrying weight, or a joint zero "
                  "offset); don't use this estimate")
    if len(results) > 1:
        est = np.array([[r["roll_deg"], r["pitch_deg"]] for r in results.values()])
        spread = est.max(0) - est.min(0)
        print(f"across bags: roll spread {spread[0]:.3f}, pitch spread {spread[1]:.3f} deg; "
              f"mean roll {est[:, 0].mean():+.3f} pitch {est[:, 1].mean():+.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
