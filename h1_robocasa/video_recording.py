"""Opt-in, file-only live-state recording. No ROS interfaces or actuation."""

from collections import deque
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import mujoco
import numpy as np


def recording_fps(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 60:
        raise ValueError("record-fps must be finite and in (0, 60]")
    return value


class RetentionTracker:
    """Physical evidence only; a command log is still needed to verify intent."""

    def __init__(self):
        self.history = deque()
        self.initial_hand_z = None
        self.was_seated = False
        self.weld_seen = False

    def observe(self, t, clearance, relative, hand_z, welded, gripping):
        self.was_seated |= clearance <= 0
        self.weld_seen |= welded
        self.initial_hand_z = min(hand_z, self.initial_hand_z) if self.initial_hand_z is not None else hand_z
        # Wrist frame to the Magpie distal pads is roughly 0.18 m.
        near = float(np.linalg.norm(relative)) < 0.25
        if (clearance < 0.01 or not near or self.weld_seen or not gripping or
                (self.history and t - self.history[-1][0] > 0.15)):
            self.history.clear()
        else:
            self.history.append((t, np.array(relative)))
            while len(self.history) > 1 and self.history[1][0] <= t - 1.0:
                self.history.popleft()
        duration = t - self.history[0][0] if self.history else 0.0
        drift = (float(np.max(np.linalg.norm(
            np.array([r for _, r in self.history]) - self.history[0][1], axis=1)))
                 if self.history else None)
        return {"near_hand": near, "hand_contact": gripping, "peg_weld_seen": self.weld_seen,
                "retained_sim_seconds": duration,
                "relative_position_drift_m": drift,
                "observed_hand_lift_m": hand_z - self.initial_hand_z,
                "physical_extraction_candidate": bool(
                    self.was_seated and duration >= 1.0 and drift <= 0.005
                    and hand_z - self.initial_hand_z >= 0.01)}


class VideoRecorder:
    """Bounded state copies on the sim thread; all GL/IO on a worker thread.

    Missing captures repeat the previous image, with a visible warning, to
    preserve simulation-time duration rather than silently speeding playback.
    """

    def __init__(self, model, env, path, fps=20):
        self.fps = recording_fps(fps)
        self.path = Path(path).expanduser().resolve()
        if self.path.suffix.lower() != ".webm":
            raise ValueError("--record must name a .webm file")
        if not self.path.parent.is_dir():
            raise ValueError(f"recording parent directory does not exist: {self.path.parent}")
        if self.path.exists() or self.path.with_suffix(".jsonl").exists():
            raise ValueError("recording or JSONL already exists; choose a fresh path")
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            raise ValueError("--record requires ffmpeg on PATH")
        self.model = model
        # Offscreen framebuffer capacity is visual-only, set before the recording
        # GL context is created; no physical model parameters change.
        model.vis.global_.offwidth = max(960, int(model.vis.global_.offwidth))
        model.vis.global_.offheight = max(720, int(model.vis.global_.offheight))
        self.config = getattr(env, "scene_config", None)
        self.peg = self.tip = self.entry = None
        self.hands = {}
        self.trackers = {}
        if self.config is not None and hasattr(self.config, "peg_variant"):
            self.peg = self._id("body", self.config.peg_prefix + self.config.peg_variant["body"])
            self.tip = self._id("site", self.config.peg_prefix + self.config.peg_variant["tip_site"])
            self.entry = self._id("site", "pegboard_" + self.config.source_hole_spec["entry_site"])
            for side in ("left", "right"):
                self.hands[side] = self._id("body", f"robot0_{side}_wrist_yaw_link")
                self.trackers[side] = RetentionTracker()
        self.free = queue.Queue()
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        for _ in range(3):
            self.free.put(np.empty(mujoco.mj_stateSize(model, self.state_spec)))
        self.pending = queue.Queue(maxsize=3)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.origin = self.next_t = None
        self.dropped = 0
        self.last_t = None
        self.thread = threading.Thread(target=self._run, name="video_recording", daemon=True)
        self.thread.start()
        if not self.ready.wait(30):
            self.stop.set()
            raise RuntimeError("recording renderer startup timed out")
        if self.error:
            raise RuntimeError(f"recording startup failed: {self.error}")

    def _id(self, kind, name):
        i = mujoco.mj_name2id(self.model, getattr(mujoco.mjtObj, "mjOBJ_" + kind.upper()), name)
        if i < 0:
            raise ValueError(f"recording cannot resolve {kind} {name!r}")
        return i

    def capture(self, data, band):
        """Caller holds sim_lock. Never waits for GL, encoding or disk IO."""
        if self.error:
            return
        t = float(data.time)
        self.last_t = t
        if self.origin is None:
            self.origin = self.next_t = t
        if t + 1e-9 < self.next_t:
            return
        self.next_t += (math.floor((t - self.next_t) * self.fps + 1e-8) + 1) / self.fps
        try:
            snapshot = self.free.get_nowait()
        except queue.Empty:
            self.dropped += 1
            return
        mujoco.mj_getState(self.model, data, snapshot, self.state_spec)
        self.pending.put_nowait((snapshot, bool(band and band.enabled), time.time()))

    def _metadata(self, data, supported, wall):
        row = {"sim_time": float(data.time), "wall_time_unix": wall,
               "support_band_enabled": supported, "dropped_captures": self.dropped,
               "commanded_lift_requires_external_command_log": True}
        if self.peg is None:
            return row
        def pose(body):
            return np.r_[data.xpos[body], data.xquat[body]].tolist()
        tip, entry = data.site_xpos[self.tip], data.site_xpos[self.entry]
        normal = data.site_xmat[self.entry].reshape(3, 3)[:, 2]
        clearance = float(np.dot(tip - entry, normal))
        contacts = []
        touching_bodies = set()
        for contact in data.contact[:data.ncon]:
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.peg in bodies:
                touching_bodies.update(int(b) for b in bodies if b != self.peg)
                contacts.append({"geoms": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                                           int(g)) for g in (contact.geom1, contact.geom2)],
                                 "distance_m": float(contact.dist)})
        welded = [int(i) for i in range(self.model.neq)
                  if data.eq_active[i] and self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD
                  and self.peg in (self.model.eq_obj1id[i], self.model.eq_obj2id[i])]
        row.update(peg_id=self.config.peg_id, source_hole=self.config.source_hole,
                   peg_tip_site=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, self.tip),
                   peg_pose_world_wxyz=pose(self.peg), peg_tip_world=tip.tolist(),
                   board_entry_world=entry.tolist(), tip_entry_clearance_m=clearance,
                   tip_entry_distance_m=float(np.linalg.norm(tip - entry)),
                   peg_contacts=contacts, active_peg_welds=welded, hands={})
        for side, hand in self.hands.items():
            def under_hand(body):
                while body > 0 and body != hand:
                    body = int(self.model.body_parentid[body])
                return body == hand
            gripping = any(under_hand(b) for b in touching_bodies)
            rotation = data.xmat[hand].reshape(3, 3)
            relative = rotation.T @ (data.xpos[self.peg] - data.xpos[hand])
            row["hands"][side] = {
                "pose_world_wxyz": pose(hand), "peg_relative_position_m": relative.tolist(),
                **self.trackers[side].observe(float(data.time), clearance, relative,
                                             float(data.xpos[hand, 2]), bool(welded), gripping)}
        return row

    def _run(self):
        process = None
        try:
            # Respect the model's existing framebuffer size without mutating it.
            width = min(960, int(self.model.vis.global_.offwidth))
            height = min(720, int(self.model.vis.global_.offheight))
            width -= width % 2
            height -= height % 2
            from PIL import Image, ImageDraw
            with mujoco.Renderer(self.model, width=width, height=height) as renderer, \
                    self.path.with_suffix(".jsonl").open("x", buffering=1) as metadata, \
                    self.path.with_suffix(".ffmpeg.log").open("x") as log:
                process = subprocess.Popen([
                    self.ffmpeg, "-n", "-loglevel", "warning", "-f", "rawvideo", "-vcodec", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(self.fps),
                    "-i", "pipe:0", "-an", "-c:v", "libvpx-vp9", "-deadline", "realtime",
                    "-cpu-used", "8", "-crf", "24", "-b:v", "0", "-pix_fmt", "yuv420p",
                    str(self.path)], stdin=subprocess.PIPE, stderr=log)
                camera = mujoco.MjvCamera()
                # This private data is never exposed to a bridge or controller.
                render_data = mujoco.MjData(self.model)
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.distance, camera.azimuth, camera.elevation = 0.68, 135, -30
                option = mujoco.MjvOption()
                option.geomgroup[:] = 0
                option.geomgroup[1:3] = 1
                option.sitegroup[:] = 0
                metadata.write(json.dumps({"event": "recording_start", "fps": self.fps,
                    "width": width, "height": height, "timebase": "simulation",
                    "evaluation_only": True, "camera": "third_person_board_closeup",
                    "state_source": "live integration snapshot; private mj_forward for geometry",
                    "retention_criteria": {"tip_clearance_m": 0.01, "duration_sim_s": 1.0,
                        "max_relative_drift_m": 0.005, "wrist_radius_m": 0.25,
                        "max_sample_gap_sim_s": 0.15, "hand_contact_required": True,
                        "no_peg_weld_ever": True, "observed_hand_lift_m": 0.01}}) + "\n")
                self.ready.set()
                index, previous = 0, None
                def write_frame(rgb, row, repeated=False):
                    nonlocal index
                    image = Image.fromarray(rgb)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, width, 46), fill="black")
                    support = "tether-supported" if row["support_band_enabled"] else "tether OFF"
                    draw.text((10, 6), f"1x simulation time; {support}; NOT wall real-time", fill="white")
                    draw.text((10, 24), f"sim {row['sim_time']:.3f}s" +
                              (" | CAPTURE GAP: repeated image" if repeated else ""), fill="white")
                    process.stdin.write(image.tobytes())
                    metadata.write(json.dumps({**row, "frame": index, "video_time": index / self.fps,
                                               "repeated_image": repeated}) + "\n")
                    index += 1
                while not self.stop.is_set() or not self.pending.empty():
                    try:
                        state, supported, wall = self.pending.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    try:
                        mujoco.mj_setState(self.model, render_data, state, self.state_spec)
                        mujoco.mj_forward(self.model, render_data)
                        data = render_data
                        row = self._metadata(data, supported, wall)
                        target = round((float(data.time) - self.origin) * self.fps)
                        while previous is not None and index < target:
                            write_frame(*previous, repeated=True)
                        if self.entry is not None:
                            camera.lookat[:] = data.site_xpos[self.entry] + np.array([0, 0, 0.07])
                        else:
                            camera.lookat[:] = data.xpos[1]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        rgb = renderer.render().copy()
                        write_frame(rgb, row)
                        previous = (rgb, row)
                    finally:
                        self.free.put_nowait(state)
                if previous is not None:
                    while index <= round((self.last_t - self.origin) * self.fps):
                        write_frame(*previous, repeated=True)
                process.stdin.close()
                if process.wait(timeout=30):
                    raise RuntimeError(f"ffmpeg failed; see {self.path.with_suffix('.ffmpeg.log')}")
                metadata.write(json.dumps({"event": "recording_end", "frames": index,
                                           "dropped_captures": self.dropped}) + "\n")
        except Exception as exc:
            self.error = str(exc)
            print(f"[recording] FAILED: {exc}", flush=True)
        finally:
            self.ready.set()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            print("[recording] writer did not finish within 60s", flush=True)
        elif not self.error:
            print(f"[recording] saved {self.path} and {self.path.with_suffix('.jsonl')}", flush=True)
