"""Test the H1-2's frontier exploration on lmg_deploy's policy in the sim, closed loop: the roles run in the containers.

golem_skills' frontier explorer walks the H1-2, on the loco-manipulation game's MARL policy
(lmg_sim.launch.py), through Nav2 from the spawn until it reports the RoboCasa kitchen
explored, or its time runs out. test_lmg_explore.sh runs the roles and has the usage. The
H1-2 counterpart of tests/g1/test_sim_explore.py, built on test_sim_stack.py.

  sim     test_sim_stack.py's sim role, recording besides its pelvis every joint of the
          scene (qpos) at each sample, and at the spawn the compiled model
          (<out>/model.mjb) for the video role. The stack test's goal (goal.json) is not used.
  drive   in the ros container, beside 'ros2 launch golem_bringup lmg_sim.launch.py
          use_rviz:=false use_skills:=true': waits until the policy node has released the
          sim's elastic band, Nav2 is active and /skill/frontier_explore is served, stands
          STAND_S, sends one SkillFrontierExplore goal (timeout EXPLORE_TIMEOUT_S, the
          explorer's defaults otherwise) and waits for its result; records what the stack
          test's drive records (drive.npz) plus every /map (maps.npz), map -> pelvis at
          MAP_POSE_PERIOD_S (drive.npz map_pelvis) and the explorer's feedback and result
          (events.json).
  report  anywhere with numpy and scipy: checks the run, prints the numbers and writes
          <out>/report.json; exit 1 on a failure.
  video   in the robocasa container (EGL): <out>/explore.mp4 from the recordings, a chase
          camera on the robot beside its head camera and the /map as slam_toolbox
          published it, with the robot's path and the explorer's goals. Optional
          (test_lmg_explore.sh's GOLEM_TEST_VIDEO).

Pass if: the stack comes up (the band released, Nav2 active and the explorer served before
the goal); from the band's release to the end the band stays off, the pelvis stays above
test_sim_stack's H1-2 floor (0.77 m) and tilts at most 15 deg, no robot geom penetrates
anything but the floor and the sim's fall logger logs no fall; golem_safety latches no estop
before the drive's end; safety_node is the only publisher of /lowcmd and nothing publishes
the sim's /odom; at least MIN_COVERAGE of the reachable floor (scene.npz) is free in the
final /map (placed in the world through map -> pelvis against the sim's pelvis, as the G1's
test does); and when the explorer reports the map explored, its goals_reached is above 0
and matches its log. Otherwise its outcome (done or out of time) and its goals are
reported, not checked.

The explorer's time and every check of the motion are in sim time; the waits for the stack
are on wall time with limits.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "g1"))
import test_sim_bringup as bringup_test  # noqa: E402  (log parsing, the recorder base, the sim hook)
import test_sim_explore as g1_explore  # noqa: E402  (the explorer's log, its outcome and the coverage check)
import test_sim_nav as nav_test  # noqa: E402  (geometry, TF helpers)
import test_sim_stack as stack_test  # noqa: E402  (the H1-2 sim role, the drive's recordings, the checks)

ROBOT = "h12"
# s of sim time: the explorer's budget, its goal's timeout. The H1-2 walks at most 0.3 m/s
# (golem_slam's config/h12/nav2.yaml), against the G1's 120 s budget for a walk at twice that.
EXPLORE_TIMEOUT_S = 240
RESULT_GRACE_S = g1_explore.RESULT_GRACE_S  # s of sim time past it to wait for the result
STAND_S = stack_test.STAND_S  # s of sim time standing before the goal, Nav2 up
SETTLE_S = stack_test.SETTLE_S  # s of sim time after the result
MIN_COVERAGE = g1_explore.MIN_COVERAGE  # the share of the reachable floor free in the final /map
MAP_POSE_PERIOD_S = 0.1  # s of wall time between samples of map -> pelvis
# The video: frames per second of sim time, and the panes' sizes (px).
VIDEO_FPS = 25
CHASE_SIZE = (960, 720)
SIDE_SIZE = (480, 360)
WALL_GROUP = 5  # the geom group the video's chase camera hides the walls in


class Recorder(stack_test.Recorder):
    """test_sim_stack's recorder plus the scene's qpos at each sample, for the video."""

    TAG = "test_lmg_explore"
    ROWS = stack_test.Recorder.ROWS + ("qpos",)

    def sample(self, iface, band):
        """Append the stack test's rows and the whole qpos."""
        super().sample(iface, band)
        self.rows["qpos"].append(iface.data.qpos.astype(np.float32))


def on_init(out, iface):
    """At the spawn: the stack test's scene map (scene.npz; its goal is unused) and the compiled model (model.mjb)."""
    import mujoco
    stack_test.map_scene(out, ROBOT, iface.model, iface.data, iface.resolver.body_id("pelvis"))
    mujoco.mj_saveModel(iface.model, os.path.join(out, "model.mjb"), None)


def run_sim(out, sim_args):
    """Run h12_mujoco.py --h12 with the recorder, the scene map and the saved model."""
    if os.environ.get("GOLEM_ROBOT", "").strip().lower() != ROBOT:
        raise SystemExit(f"GOLEM_ROBOT must be {ROBOT}")
    bringup_test.run_recorded_sim(out, ROBOT, sim_args, Recorder(out), "test_lmg_explore",
                                  on_init=lambda iface: on_init(out, iface))


class Driver(stack_test.Driver):
    """Drive the explore test: the stack test's recordings, every /map, map -> pelvis and one exploration."""

    def __init__(self, out):
        """Record as the stack test's drive does, plus the maps' contents and the global costmap."""
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        super().__init__(out)
        self.name = "test_lmg_explore"
        self.map = self.costmap = None
        self.map_grids = []  # (sim t, the OccupancyGrid) of every /map
        self.node.create_subscription(OccupancyGrid, "/map", self._keep_map, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.node.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                      lambda msg: setattr(self, "costmap", msg), QoSProfile(
                                          depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                          durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.map_pelvis = []  # wall, sim, then _tf_parts of map -> pelvis (NaN: none)
        self._next_pose = 0.0
        self.explore_feedback = []  # sim t, phase, frontiers_remaining, at each change

    def _keep_map(self, msg):
        """Keep every /map, with the sim time it arrived."""
        self.map = msg
        self.map_grids.append((self._sim(), msg))

    def samples(self, now):
        """The stack test's samples, and map -> pelvis when due."""
        super().samples(now)
        if now >= self._next_pose:
            self._next_pose = now + MAP_POSE_PERIOD_S
            tf = nav_test._tf_parts(self.lookup("map", "pelvis"))
            self.map_pelvis.append([now, self._sim()] + ([math.nan] * 8 if tf is None else tf))

    def wait_nav(self, timeout=180.0):
        """Wait until Nav2 is ready; return whether it became ready within `timeout` s of wall time.

        Ready: Nav2's lifecycle nodes active, /navigate_to_pose served, map -> base_footprint in
        TF, and a /map and a /converted_scan heard. The game runs no frame_task_server, so
        /named_config is not waited for.
        """
        from lifecycle_msgs.msg import State
        from lifecycle_msgs.srv import GetState
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        navigate = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        clients = {name: self.node.create_client(GetState, f"/{name}/get_state") for name in stack_test.NAV2_NODES}
        end = time.time() + timeout
        states = {}
        while time.time() < end:
            for name, client in clients.items():
                if states.get(name) == "active" or not client.service_is_ready():
                    continue
                response = self.wait_future(client.call_async(GetState.Request()), 5.0)
                if response is not None:
                    states[name] = ("active" if response.current_state.id == State.PRIMARY_STATE_ACTIVE
                                    else response.current_state.label)
            if (all(states.get(n) == "active" for n in stack_test.NAV2_NODES) and navigate.server_is_ready()
                    and self.lookup("map", "base_footprint") is not None and self.maps and self.scans):
                self.mark("nav_active", states=states)
                return True
            for _ in range(20):
                self.spin()
        self.mark("nav_timeout", states=states, action_server=navigate.server_is_ready(),
                  map_base=self.lookup("map", "base_footprint") is not None, maps=len(self.maps),
                  scans=len(self.scans))
        return False

    def wait_explorer(self, timeout=60.0):
        """Wait for /skill/frontier_explore to be served; return whether it was within `timeout` s."""
        from custom_ros_messages.action import SkillFrontierExplore
        from rclpy.action import ActionClient
        self.explorer = ActionClient(self.node, SkillFrontierExplore, "/skill/frontier_explore")
        end = time.time() + timeout
        while time.time() < end:
            if self.explorer.server_is_ready():
                self.mark("explorer_up")
                return True
            for _ in range(20):
                self.spin()
        self.mark("explorer_timeout")
        return False

    def _on_explore_feedback(self, msg):
        """Record the explorer's feedback when its phase or frontier count changes."""
        f = msg.feedback
        row = [self.clock, f.phase, f.frontiers_remaining]
        if not self.explore_feedback or self.explore_feedback[-1][1:] != row[1:]:
            self.explore_feedback.append(row)

    def explore(self):
        """Run one exploration to its result; return True if it ended with a result.

        After EXPLORE_TIMEOUT_S + RESULT_GRACE_S of sim time the goal is canceled.
        """
        from action_msgs.msg import GoalStatus
        from custom_ros_messages.action import SkillFrontierExplore
        goal = SkillFrontierExplore.Goal()
        goal.timeout.sec = EXPLORE_TIMEOUT_S
        self.mark("explore_sent", timeout_s=EXPLORE_TIMEOUT_S,
                  map_pelvis=nav_test._tf_parts(self.lookup("map", "pelvis")))
        handle = self.wait_future(self.explorer.send_goal_async(
            goal, feedback_callback=self._on_explore_feedback), 10.0)
        if handle is None or not handle.accepted:
            self.mark("explore_result", accepted=False, status=None, success=False)
            return False
        result = handle.get_result_async()
        start = self.clock
        give_up = time.time() + 4 * (EXPLORE_TIMEOUT_S + RESULT_GRACE_S)
        while not result.done():
            if self.clock - start > EXPLORE_TIMEOUT_S + RESULT_GRACE_S or time.time() > give_up:
                self.mark("explore_timeout", sim_seconds=self.clock - start)
                self.wait_future(handle.cancel_goal_async(), 10.0)
                self.wait_future(result, 10.0)
                break
            self.spin()
        response = result.result() if result.done() else None
        r = response.result if response is not None else None
        # The /map and the costmap the explorer last saw.
        for name, msg in (("map_done.npz", self.map), ("costmap_done.npz", self.costmap)):
            if msg is not None:
                g1_explore.save_grid(os.path.join(self.out, name), msg)
        self.mark("explore_result", accepted=True, status=response.status if response else None,
                  success=bool(r and r.success), message=r.message if r else None,
                  goals_reached=r.goals_reached if r else None, sim_seconds=self.clock - start,
                  feedback=self.explore_feedback, map_pelvis=nav_test._tf_parts(self.lookup("map", "pelvis")))
        return response is not None and response.status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED)

    def finish(self):
        """Save the stack test's recordings with map -> pelvis, the final /map (map.npz) and every /map (maps.npz)."""
        super().finish()
        with np.load(os.path.join(self.out, "drive.npz")) as drive:
            arrays = dict(drive)
        arrays["map_pelvis"] = np.array(self.map_pelvis, float).reshape(-1, 10)
        np.savez(os.path.join(self.out, "drive.npz"), **arrays)
        if self.map is not None:
            g1_explore.save_grid(os.path.join(self.out, "map.npz"), self.map)
        if self.map_grids:
            grids = {}
            for i, (t, msg) in enumerate(self.map_grids):
                o = msg.info.origin
                grids[f"data_{i}"] = np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width)
                grids[f"meta_{i}"] = np.array([t, msg.info.resolution, o.position.x, o.position.y, nav_test._yaw_xyzw(
                    o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w)])
            np.savez_compressed(os.path.join(self.out, "maps.npz"), count=len(self.map_grids), **grids)


def run_drive(out):
    """Run the drive role; return 0 when the exploration ended with a result."""
    d = Driver(out)
    ok = False
    try:
        d.mark("drive_start")
        if (d.wait_band_released(timeout=300.0, refusals=bringup_test.LAUNCH_REFUSALS) and d.wait_nav()
                and d.wait_explorer()):
            d.spin_for(STAND_S)
            ok = d.explore()
            d.spin_for(SETTLE_S)
        return 0 if ok else 1
    finally:
        try:
            d.finish()  # what was recorded, whatever happened
        finally:
            d.close()


class Report(stack_test.Report):
    """Check an explore run: the stack test's whole-run checks plus the exploration's own."""

    # The G1 explore test's checks of the explorer and the coverage, and the nav test's placing of the map.
    explorer_outcome = g1_explore.Report.explorer_outcome
    coverage = g1_explore.Report.coverage
    world_from = nav_test.Report.world_from

    def stack_up(self):
        """Check the band was released, Nav2 came up and the explorer was served before the goal; return the release."""
        release = self.release()
        missing = [n for n in ("band_released", "nav_active", "explorer_up", "explore_sent") if n not in self.events]
        if release is None or missing:
            self.check("stack up", False, ("the sim's elastic band was never released; " if release is None else "")
                       + (f"no {missing} mark: see drive.log and launch.log" if missing else ""))
            return None
        first = self.events["drive_start"]["wall"]
        self.numbers["band_release_t"] = release[1]
        self.check("stack up", True,
                   f"band released at sim t {release[1]:.2f} s; Nav2 active at sim t "
                   f"{self.events['nav_active']['sim']:.1f} s, the explorer at "
                   f"{self.events['explorer_up']['sim']:.1f} s ({self.events['explorer_up']['wall'] - first:.0f} s "
                   f"wall after the drive started)")
        return release

    def run(self, out):
        """Run every check, write report.json and print the verdict; return 0 if all passed."""
        t, wall = self.s["t"], self.s["wall"]
        self.numbers["real_time_factor"] = float((t[-1] - t[0]) / (wall[-1] - wall[0]))
        print(f"     {self.robot}: {len(t)} samples, sim t {t[0]:.1f}-{t[-1]:.1f} s, real-time factor "
              f"{self.numbers['real_time_factor']:.2f}")
        bringup_test.record_sim_settings(self.numbers, out)
        g = self.goal
        self.numbers["spawn"] = {"pelvis": g["spawn"], "yaw_deg": math.degrees(g["spawn_yaw"]),
                                 "clearance": g["spawn_clearance"]}
        release = self.stack_up()
        if release is not None:
            self.standing(release)
            self.explorer_outcome()
            sent = self.events["explore_sent"]
            if sent.get("map_pelvis") is None:
                self.check("coverage", False, "no map -> pelvis when the goal was sent")
            else:
                self.coverage(self.world_from(sent["map_pelvis"]))
            self.contacts(release)
        self.fall_log()
        self.writers()
        self.no_estop()
        ok = all(c["pass"] for c in self.checks)
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump({"pass": ok, "checks": self.checks, "numbers": self.numbers}, f, indent=1)
        print(f"{'PASSED' if ok else 'FAILED'}: {sum(c['pass'] for c in self.checks)}/{len(self.checks)} checks")
        return 0 if ok else 1


def _map_image(data, size, res, origin, path_xy, robot, goals):
    """Return a BGR image of an occupancy grid (rows y) fitted into size (w, h), with the path, robot and goals.

    path_xy (n, 2) and goals [(x, y, outcome)] are in the map frame; robot is (x, y, yaw) or None.
    """
    import cv2
    h, w = data.shape
    img = np.full((h, w, 3), 128, np.uint8)  # unknown
    img[(data >= 0) & (data < 50)] = (235, 235, 235)
    img[data >= 50] = (40, 40, 40)
    img = img[::-1]  # rows up
    scale = min(size[0] / w, size[1] / h)
    img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size[1], size[0], 3), 128, np.uint8)
    ox, oy = (size[0] - img.shape[1]) // 2, (size[1] - img.shape[0]) // 2
    canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img

    def px(x, y):
        """Return the canvas pixel of map point (x, y)."""
        return (int(ox + (x - origin[0]) / res * scale), int(oy + img.shape[0] - (y - origin[1]) / res * scale))
    if len(path_xy) > 1:
        cv2.polylines(canvas, [np.array([px(x, y) for x, y in path_xy], np.int32)], False, (200, 120, 0), 2)
    colors = {"reached": (0, 160, 0), "seen on the way": (0, 160, 0), "failed": (0, 0, 220), None: (0, 140, 255)}
    for gx, gy, outcome in goals:
        cv2.drawMarker(canvas, px(gx, gy), colors.get(outcome, (0, 140, 255)), cv2.MARKER_TILTED_CROSS, 14, 2)
    if robot is not None:
        x, y, yaw = robot
        p = px(x, y)
        cv2.circle(canvas, p, 6, (0, 0, 200), -1)
        cv2.line(canvas, p, px(x + 0.4 * math.cos(yaw), y + 0.4 * math.sin(yaw)), (0, 0, 200), 2)
    return canvas


def run_video(out):
    """Write <out>/explore.mp4: the chase camera, the head camera and the /map, at VIDEO_FPS of sim time."""
    import cv2
    import mujoco
    model = mujoco.MjModel.from_binary_path(os.path.join(out, "model.mjb"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, CHASE_SIZE[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, CHASE_SIZE[1])
    data = mujoco.MjData(model)
    sim = np.load(os.path.join(out, "sim.npz"))
    t, qpos = sim["t"], sim["qpos"]
    with open(os.path.join(out, "events.json")) as f:
        ev = {e["name"]: e for e in json.load(f)["events"]}
    drive = np.load(os.path.join(out, "drive.npz"))
    poses = drive["map_pelvis"]
    poses = poses[np.isfinite(poses[:, 3])]
    maps = np.load(os.path.join(out, "maps.npz"))
    map_meta = [maps[f"meta_{i}"] for i in range(int(maps["count"]))]
    map_t = np.array([m[0] for m in map_meta])
    goals = g1_explore.explorer_goals(os.path.join(out, "launch.log"))
    feedback = ev.get("explore_result", {}).get("feedback") or []
    # From the band's release (or the start) to the end.
    t0 = ev.get("band_released", {}).get("sim") or float(t[0])
    frames = np.arange(max(t0 - 2.0, t[0]), t[-1], 1.0 / VIDEO_FPS)
    head = next(model.camera(i).name for i in range(model.ncam) if model.camera(i).name.endswith("head_cam"))
    # The sim prefixes the robot's names (robot0_pelvis on the H1-2).
    pelvis = next(i for i in range(model.nbody) if model.body(i).name.endswith("pelvis"))
    # The kitchen's walls, hidden from the chase camera (geom group WALL_GROUP, off in its scene option), which would
    # otherwise often sit behind one; the head camera sees them.
    for g in range(model.ngeom):
        if model.body(model.geom_bodyid[g]).name.startswith("wall_"):
            model.geom_group[g] = WALL_GROUP
    chase_option = mujoco.MjvOption()
    chase_option.geomgroup[WALL_GROUP] = 0
    chase = mujoco.Renderer(model, CHASE_SIZE[1], CHASE_SIZE[0])
    side = mujoco.Renderer(model, SIDE_SIZE[1], SIDE_SIZE[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = pelvis
    cam.distance, cam.elevation = 3.4, -30.0
    path = os.path.join(out, "explore.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS,
                             (CHASE_SIZE[0] + SIDE_SIZE[0], CHASE_SIZE[1]))
    azimuth = None
    sent_t = ev.get("explore_sent", {}).get("sim")
    for ft in frames:
        i = min(int(np.searchsorted(t, ft)), len(t) - 1)
        data.qpos[:] = qpos[i]  # float32 in the recording
        mujoco.mj_forward(model, data)
        # The camera behind the robot, turning slowly with it.
        yaw = math.degrees(nav_test.yaw_of(data.xquat[pelvis][None])[0])
        target = yaw + 180.0 - 30.0
        azimuth = target if azimuth is None else azimuth + ((target - azimuth + 180.0) % 360.0 - 180.0) * 0.03
        cam.azimuth = azimuth
        chase.update_scene(data, cam, scene_option=chase_option)
        left = cv2.cvtColor(chase.render(), cv2.COLOR_RGB2BGR)
        side.update_scene(data, head)
        top = cv2.cvtColor(side.render(), cv2.COLOR_RGB2BGR)
        k = int(np.searchsorted(map_t, ft, side="right")) - 1
        seen = poses[poses[:, 1] <= ft]
        robot = None
        if len(seen):
            p = seen[-1]
            robot = (p[3], p[4], nav_test._yaw_xyzw(*p[6:10]))
        if k >= 0:
            _, res, mx, my, _ = map_meta[k]
            shown = goals if sent_t is not None and ft >= sent_t else []
            bottom = _map_image(maps[f"data_{k}"], SIDE_SIZE, float(res), (mx, my), seen[:, 3:5], robot,
                                [(g["x"], g["y"], g["outcome"]) for g in shown])
        else:
            bottom = np.full((SIDE_SIZE[1], SIDE_SIZE[0], 3), 128, np.uint8)
        frame = np.concatenate([left, np.concatenate([top, bottom], axis=0)], axis=1)
        phase = next((row[1] for row in reversed(feedback) if row[0] is not None and row[0] <= ft), "")
        label = f"sim t {ft:6.1f} s"
        if sent_t is not None and ft >= sent_t:
            label += f"   exploring +{ft - sent_t:5.1f} s  {phase}"
        for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thick + 1, cv2.LINE_AA)
            cv2.putText(frame, "head camera", (CHASE_SIZE[0] + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,
                        thick, cv2.LINE_AA)
            cv2.putText(frame, "/map (slam_toolbox)", (CHASE_SIZE[0] + 10, SIDE_SIZE[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thick, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"[test_lmg_explore] wrote {path}: {len(frames)} frames, {len(frames) / VIDEO_FPS:.1f} s", flush=True)
    return 0


def main():
    """Run the role the command line names; return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("sim", "drive", "report", "video"))
    parser.add_argument("--out", required=True, help="the shared output directory")
    parser.epilog = "sim: h12_mujoco.py's arguments follow a --"
    argv = sys.argv[1:]
    sim_args = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:len(argv) - len(sim_args) - ("--" in argv)])
    if args.role == "sim":
        run_sim(args.out, sim_args)
        return 0
    if args.role == "drive":
        return run_drive(args.out)
    if args.role == "video":
        return run_video(args.out)
    return Report(args.out).run(args.out)


if __name__ == "__main__":
    sys.exit(main())
