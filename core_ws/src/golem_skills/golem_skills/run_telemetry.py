#!/usr/bin/env python3
"""
Record per-run base-sway telemetry for the skills node.

A RunRecorder's background thread samples, at RATE_HZ wall clock:

  * the robot base pose: `pelvis` in the odometry world frame,
  * the grasping arm's gripper point: its GraspGenX frame in `pelvis`,
  * the task target: a fixed world point (the screw, the handle),

and appends one CSV row per tick. Each skill run gets its own directory,
`<package>/logs/runs/<skill>/<stamp>_<slug>/` (host-persistent through the
core_ws bind mount, like the model logs), holding `telemetry.csv` (COLUMNS)
and `result.json` (the run's outcome, arm, target and request).

The columns, rate and yaw formula match the fridge sway study's recorder, so
its analysis (sway_analysis.py, FS = 10) reads these files unchanged.

The base pose comes from odometry (the WORLD_FRAME -> pelvis TF), not from a
whole-body CoM or a sim ground-truth topic, so the same code runs in the sim
and on the real robot, where only IMU/odometry (FAST-LIO) measure the base.

Recording is best-effort: a sampling error leaves NaN in that row (the
analysis drops NaN rows) and a write error is logged; neither stops a skill.
"""

import csv
import json
import math
import os
import threading
import time
from datetime import datetime

from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time


# Hz; the fridge recorder's rate, which sway_analysis.py assumes
RATE_HZ = 10.0
# the fridge telemetry's columns, with its door_* named target_*
COLUMNS = ('t_wall', 'pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_yaw_deg',
           'grip_x', 'grip_y', 'grip_z', 'target_x', 'target_y', 'target_z')
_NAN = float('nan')


def _slug(text):
    """Return a filesystem-safe lowercase slug of `text` ('run' when empty)."""
    out = ''.join(c if c.isalnum() else '_' for c in (text or '')).strip('_').lower()
    return out or 'run'


def describe_goal(goal):
    """
    Flatten a skill goal's primitive fields into a JSON-friendly dict for result.json.

    Lists and nested messages are summarized (item count, type name), not dumped.
    """
    out = {}
    if goal is None:
        return out
    try:
        for name in goal.get_fields_and_field_types():
            v = getattr(goal, name)
            if isinstance(v, (bool, int, float, str)):
                out[name] = v
            elif isinstance(v, (list, tuple, bytes)):
                out[name] = f'[{len(v)} items]'
            else:
                out[name] = type(v).__name__
    except Exception:
        pass
    return out


def _yaw_deg(qx, qy, qz, qw):
    """Return the yaw about world +Z [deg] of a quaternion (the fridge recorder's formula)."""
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy),
                                   1.0 - 2.0 * (qy * qy + qz * qz)))


class RunRecorder:
    """
    Sample and store one skill run's telemetry.

    Construct, start(), call note_arm() / note_target() as the skill resolves
    them, then finish(), which stops the sampler and writes result.json.

    All ROS access goes through `node` (its tf_buffer, logger and
    gripper_aperture). The sampler never spins the executor, so it runs on its
    own daemon thread beside the node's MultiThreadedExecutor.
    """

    def __init__(self, node, skill, out_dir, world_frame, grip_frames,
                 arm=None, request=None):
        """Prepare a recorder writing into `out_dir`; nothing is sampled before start()."""
        self._node = node
        self._skill = skill
        self._dir = out_dir
        self._world_frame = world_frame
        # {'left': frame, 'right': frame}
        self._grip_frames = grip_frames
        self._request = request or {}
        # guards _arm, _grip_frame and _target, which the skill updates mid-run
        self._lock = threading.Lock()
        self._arm = arm if arm in grip_frames else None
        self._grip_frame = grip_frames.get(self._arm) if self._arm else None
        # (x, y, z) in the world frame
        self._target = None
        self._stop = threading.Event()
        self._thread = None
        self._csv_file = None
        self._csv = None
        self._rows = 0
        self._t0 = None

    def start(self):
        """Create the run directory, open telemetry.csv and start the sampler; return self."""
        try:
            os.makedirs(self._dir, exist_ok=True)
            self._csv_file = open(os.path.join(self._dir, 'telemetry.csv'),
                                  'w', newline='')
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(COLUMNS)
        except OSError as e:
            self._node.get_logger().warn(
                f'run telemetry: cannot open {self._dir}: {e}; not recording')
            self._csv = self._csv_file = None
            return self
        self._t0 = time.time()
        self._thread = threading.Thread(
            target=self._loop, name=f'run_telemetry_{self._skill}', daemon=True)
        self._thread.start()
        self._node.get_logger().info(
            f'run telemetry: recording {self._skill} at {RATE_HZ:.0f} Hz -> '
            f'{self._dir}')
        return self

    def note_arm(self, arm):
        """Point the gripper columns at `arm`'s GraspGenX frame once the skill has chosen it."""
        if arm not in self._grip_frames:
            return
        with self._lock:
            self._arm = arm
            self._grip_frame = self._grip_frames[arm]

    def note_target(self, xyz):
        """Record the task target, a fixed world point: a Pose (its position) or (x, y, z)."""
        pos = getattr(xyz, 'position', None)
        if pos is not None:
            xyz = (pos.x, pos.y, pos.z)
        try:
            t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        except (TypeError, ValueError, IndexError):
            return
        with self._lock:
            self._target = t

    def finish(self, success=None, message=None):
        """Stop the sampler, close the CSV and write result.json; call it once."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except OSError:
                pass
        with self._lock:
            arm, grip_frame, target = self._arm, self._grip_frame, self._target
        aperture = None
        if arm is not None:
            try:
                aperture = self._node.gripper_aperture(arm)
            except Exception:
                aperture = None
        record = {
            'skill': self._skill,
            'success': None if success is None else bool(success),
            'message': None if message is None else str(message),
            'arm': arm,
            'target_world': list(target) if target is not None else None,
            'grip_final_mm': aperture,
            'n_rows': self._rows,
            'rate_hz': RATE_HZ,
            'duration_s': (round(time.time() - self._t0, 3)
                           if self._t0 is not None else None),
            't_start_wall': self._t0,
            'world_frame': self._world_frame,
            'grip_frame': grip_frame,
            'request': self._request,
        }
        try:
            with open(os.path.join(self._dir, 'result.json'), 'w') as f:
                json.dump(record, f, indent=2, default=str)
        except OSError as e:
            self._node.get_logger().warn(
                f'run telemetry: cannot write result.json in {self._dir}: {e}')
        if self._csv is not None:
            self._node.get_logger().info(
                f'run telemetry: {self._skill} wrote {self._rows} rows -> '
                f'{self._dir}')

    def _loop(self):
        """Append one row every 1/RATE_HZ s until finish() stops the sampler."""
        period = 1.0 / RATE_HZ
        while not self._stop.is_set():
            self._sample_row()
            if self._stop.wait(period):
                break

    def _sample_row(self):
        """Look up the base and gripper poses and append one CSV row (NaN where TF fails)."""
        px = py = pz = yaw = _NAN
        gx = gy = gz = _NAN
        tx = ty = tz = _NAN
        with self._lock:
            grip_frame, target = self._grip_frame, self._target
        try:
            tf = self._node.tf_buffer.lookup_transform(
                self._world_frame, 'pelvis', Time(),
                timeout=RclpyDuration(seconds=0.05))
            t, q = tf.transform.translation, tf.transform.rotation
            px, py, pz = t.x, t.y, t.z
            yaw = _yaw_deg(q.x, q.y, q.z, q.w)
        except Exception:
            pass
        # pelvis frame, as the fridge recorder's pelvis -> right_graspgenx_frame lookup
        if grip_frame is not None:
            try:
                tf = self._node.tf_buffer.lookup_transform(
                    'pelvis', grip_frame, Time(),
                    timeout=RclpyDuration(seconds=0.05))
                gt = tf.transform.translation
                gx, gy, gz = gt.x, gt.y, gt.z
            except Exception:
                pass
        if target is not None:
            tx, ty, tz = target
        if self._csv is None:
            return
        try:
            self._csv.writerow([
                f'{time.time() - self._t0:.2f}',
                f'{px:.4f}', f'{py:.4f}', f'{pz:.4f}', f'{yaw:.2f}',
                f'{gx:.4f}', f'{gy:.4f}', f'{gz:.4f}',
                f'{tx:.4f}', f'{ty:.4f}', f'{tz:.4f}'])
            self._csv_file.flush()
            self._rows += 1
        except (OSError, ValueError):
            pass


def run_dir(log_root, skill, target_object):
    """Return the run's directory, `<log_root>/runs/<skill>/<YYYYmmdd_HHMMSS_mmm>_<slug>`."""
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    return os.path.join(log_root, 'runs', skill,
                        f'{stamp}_{_slug(target_object)}')
