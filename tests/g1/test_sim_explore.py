"""Closed-loop frontier-exploration test of the G1 sim: g1_navigation's frontier
explorer walks the G1 through Nav2 from the spawn until it reports the RoboCasa
kitchen explored, or its time runs out. The parts that run in the containers;
test_sim_explore.sh runs them and has the usage. Built on test_sim_nav.py.

  sim     test_sim_nav.py's sim role as it is: h12_mujoco.py --g1, recording the robot's
          state (<out>/sim.npz) and any robot-environment contact (<out>/contacts.json),
          and at the spawn the kitchen's occupancy from the sim's own geometry
          (<out>/scene.npz), among it the floor the robot can reach: the cells of a 5 cm
          grid, 10 x 10 m around the spawn, connected to the spawn through cells at
          least PATH_CLEARANCE (0.35 m) from anything the robot can touch. The nav
          test's goal (<out>/goal.json) is not used
  drive   in the ros container, beside 'ros2 launch g1_bringup g1_sim_bringup.launch.py
          use_locomotion:=true use_nav:=true frontier_explore:=true': waits until the
          walking node walks, Nav2 is active and /skill/frontier_explore is served,
          stands STAND_S, sends one SkillFrontierExplore goal (timeout EXPLORE_TIMEOUT_S,
          the explorer's defaults otherwise) and waits for its result; records as the nav
          test's drive does (<out>/drive.npz, map.npz) and the explorer's feedback and
          result (<out>/events.json)
  report  anywhere with numpy and scipy: checks the run, prints the numbers and writes
          <out>/report.json; exit 1 on a failure

Pass if: over the whole recording the pelvis stays above MIN_PELVIS_Z and tilts at
most MAX_TILT_DEG, and the sim's fall logger logs no fall (never fell); no robot geom
penetrates anything but the floor (touches nothing); every /cmd_vel is inside the speed
caps and the velocity smoother is its only publisher; at least MIN_COVERAGE of the
reachable floor (scene.npz) is free in the final /map (placed in the world through
map -> pelvis against the sim's pelvis, as the nav test does); and every occupied cell
of the final /map is within MAP_TOL of something the sim has in the scan's band (no
phantom cells). The stack has to come up, and the sim has to publish /odom exactly
when the test runs sim_gt. When the explorer reports the map explored, its result's
goals_reached has to be above 0 and match its log: the goals Nav2 reached plus those
seen on the way (frontier_explore counts both). Otherwise its outcome (done or out of
time), its goals and their failures are reported, not checked, and so are the nav
test's other checks (the scan path, the scan's phantoms, the localisation), whose
limits are for one goal 2-8 m from the spawn.

The explorer's time and every check of the motion are in sim time; the waits for the
stack are on wall time with limits.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_sim_nav as nav  # noqa: E402  (the sim role, the drive's recorder, the report's checks)

# the explorer's budget, its goal's timeout (sim s). In 11 runs at the explorer's current
# settings it was done after 20.7-28.7 s, and in 4 since slam_toolbox's and Nav2's base frame
# moved to the floor after 23.6-29.8 s; before it canceled goals seen on the way, after
# 71.6 s (53 s of it on one goal Nav2 could not reach). Four times the longest
EXPLORE_TIMEOUT_S = 120
RESULT_GRACE_S = 30.0      # sim s past it to wait for the result (a Nav2 goal is canceled at the end)
STAND_S = nav.STAND_S      # sim s standing before the goal, Nav2 up
SETTLE_S = nav.SETTLE_S    # sim s after the result
# pass limit: the share of the reachable floor free in the final /map. Each of those 11 runs
# ended at 100% (6166 cells, 15.4 m^2); one run before the explorer waited for a costmap
# newer than its /map stopped at 88.7%, the corridor behind the counter peninsula (11% of
# the floor) unseen. This catches a skipped corridor and leaves room for a few unseen cells
MIN_COVERAGE = 0.95
MAP_OCCUPIED = nav.MAP_OCCUPIED
# the explorer's log lines (g1_navigation/frontier_explore.py)
GOAL_SENT = re.compile(r"\[frontier_explore\]: goal (\d+): \(([-\d.]+), ([-\d.]+)\) yaw ([-\d.]+) deg, "
                       r"(\d+) frontier cells, ([\d.]+) m from the robot")
GOAL_END = re.compile(r"\[frontier_explore\]: goal (\d+): (reached|failed|seen on the way)(?: \(([^)]*)\))? "
                      r"after ([\d.]+) s")


# --------------------------------------------------------------------------- #
# drive: the exploration against the running bringup
# --------------------------------------------------------------------------- #
def save_grid(path, msg):
    """A nav_msgs/OccupancyGrid as the nav test's map.npz"""
    o = msg.info.origin
    q = o.orientation
    np.savez(path, data=np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width),
             res=msg.info.resolution, frame_id=msg.header.frame_id,
             origin=[o.position.x, o.position.y, nav._yaw_xyzw(q.x, q.y, q.z, q.w)],
             stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)


class Driver(nav.Driver):
    def __init__(self, out, odom_source):
        super().__init__(out, odom_source)
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        self.explore_feedback = []   # sim t, phase, frontiers_remaining, at each change
        # the global costmap, which the explorer walks on
        self.costmap = None
        self.node.create_subscription(OccupancyGrid, "/global_costmap/costmap",
                                      lambda msg: setattr(self, "costmap", msg), QoSProfile(
                                          depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                          durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def wait_explorer(self, timeout=60.0):
        """/skill/frontier_explore served"""
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
        f = msg.feedback
        row = [self.clock, f.phase, f.frontiers_remaining]
        if not self.explore_feedback or self.explore_feedback[-1][1:] != row[1:]:
            self.explore_feedback.append(row)

    def explore(self):
        """One exploration, to its result (or EXPLORE_TIMEOUT_S + RESULT_GRACE_S of sim
        time, then canceled); True if it ended with a result"""
        from action_msgs.msg import GoalStatus
        from custom_ros_messages.action import SkillFrontierExplore
        goal = SkillFrontierExplore.Goal()
        goal.timeout.sec = EXPLORE_TIMEOUT_S
        self.mark("explore_sent", timeout_s=EXPLORE_TIMEOUT_S, map_pelvis=nav._tf_parts(self.lookup("map", "pelvis")),
                  map_odom=nav._tf_parts(self.lookup("map", "odom")))
        handle = self.wait_future(self.explorer.send_goal_async(
            goal, feedback_callback=self._on_explore_feedback), 10.0)
        if handle is None or not handle.accepted:
            self.mark("explore_result", accepted=False, status=None, success=False)
            return False
        result = handle.get_result_async()
        start = self.clock
        give_up = time.time() + 4 * (EXPLORE_TIMEOUT_S + RESULT_GRACE_S)
        while not result.done():
            if (self.clock - start > EXPLORE_TIMEOUT_S + RESULT_GRACE_S or time.time() > give_up
                    or self._stall_dumped):
                self.mark("explore_timeout" if not self._stall_dumped else "explore_abandoned_on_stall",
                          sim_seconds=self.clock - start)
                self.wait_future(handle.cancel_goal_async(), 10.0)
                self.wait_future(result, 10.0)
                break
            self.spin()
        response = result.result() if result.done() else None
        r = response.result if response is not None else None
        # the /map and the costmap the explorer last saw
        for name, msg in (("map_done.npz", self.map), ("costmap_done.npz", self.costmap)):
            if msg is not None:
                save_grid(os.path.join(self.out, name), msg)
        self.mark("explore_result", accepted=True, status=response.status if response else None,
                  success=bool(r and r.success), message=r.message if r else None,
                  goals_reached=r.goals_reached if r else None, sim_seconds=self.clock - start,
                  feedback=self.explore_feedback, map_pelvis=nav._tf_parts(self.lookup("map", "pelvis")))
        return response is not None and response.status in (GoalStatus.STATUS_SUCCEEDED,
                                                             GoalStatus.STATUS_ABORTED)


def run_drive(out, odom_source):
    d = Driver(out, odom_source)
    ok = False
    try:
        d.mark("drive_start")
        if d.wait_walking() and d.wait_nav() and d.wait_explorer():
            d.spin_for(STAND_S)
            ok = d.explore()
            d.spin_for(SETTLE_S)
        return 0 if ok else 1
    finally:
        try:
            d.finish()   # what was recorded, whatever happened
        finally:
            d.close()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def explorer_goals(launch_log):
    """The explorer's goals from launch.log: [{n, x, y (map), yaw_deg, cells, distance, outcome,
    why, seconds}]"""
    goals = {}
    try:
        with open(launch_log, errors="replace") as f:
            for line in f:
                m = GOAL_SENT.search(line)
                if m:
                    n = int(m.group(1))
                    goals[n] = {"n": n, "x": float(m.group(2)), "y": float(m.group(3)),
                                "yaw_deg": float(m.group(4)), "cells": int(m.group(5)),
                                "distance": float(m.group(6)), "outcome": None}
                m = GOAL_END.search(line)
                if m and int(m.group(1)) in goals:
                    goals[int(m.group(1))].update(outcome=m.group(2), why=m.group(3),
                                                  seconds=float(m.group(4)))
    except OSError:
        pass
    return [goals[n] for n in sorted(goals)]


class Report(nav.Report):
    def record_only(self, method, *args):
        """Run one of the nav test's checks for its numbers: its verdict is printed and kept
        (report.json's "notes"), not counted"""
        def note(name, ok, text):
            self.notes.append({"check": name, "within_nav_limit": bool(ok), "detail": text})
            print(f"NOTE {name} ({'within' if ok else 'outside'} the nav test's limit): {text}")
        self.check = note
        try:
            method(*args)
        finally:
            del self.check

    def nav_test(self):
        """The exploration's checks, in place of the nav test's goal"""
        self.notes = []
        s, ev, g = self.s, self.events, self.goal
        self.numbers["spawn"] = {"pelvis": g["spawn"], "yaw_deg": math.degrees(g["spawn_yaw"]),
                                 "clearance": g["spawn_clearance"]}
        print(f"     spawn ({g['spawn'][0]:.3f}, {g['spawn'][1]:.3f}), yaw {math.degrees(g['spawn_yaw']):.1f} deg, "
              f"{g['spawn_clearance']:.2f} m of room")
        odom_pubs = self.info.get("odom_publishers")
        want_odom = ["mujoco_sensors"] if self.info["odom_source"] == "sim_gt" else []
        self.check("odom source", odom_pubs == want_odom,
                   f"odom_source {self.info['odom_source']}: /odom publishers {odom_pubs} (expected {want_odom})")
        released = np.nonzero(s["released"])[0]
        if not released.size:
            self.check("stack up", False, "the stand-in was never released (no valid rt/lowcmd)")
            return
        r = int(released[0])
        for name in ("walking", "nav_active", "explorer_up", "explore_sent"):
            if name not in ev:
                self.check("stack up", False, f"no '{name}' mark: see drive.log and launch.log")
                return
        first = ev["drive_start"]["wall"]
        self.check("stack up", True,
                   f"stand-in released at sim t {s['t'][r]:.2f} s; the drive saw the walking node walk at sim t "
                   f"{ev['walking']['sim']:.1f} s, Nav2 active at {ev['nav_active']['sim']:.1f} s and the explorer "
                   f"at {ev['explorer_up']['sim']:.1f} s ({ev['explorer_up']['wall'] - first:.0f} s wall after it "
                   f"started)")
        self.explorer_outcome()
        sent = ev["explore_sent"]
        last = max(ev.values(), key=lambda e: e["wall"])["name"]
        i_end = self.at("explore_result") if "explore_result" in ev else self.at(last)
        a = self.at("explore_sent")
        self.phases([("stand-in", 0, r), ("standing, walking node", r, a), ("exploring", a, i_end + 1),
                     ("after the result", i_end + 1, len(s["t"]))])
        self.cmd_vel()
        world_map = self.world_from(sent["map_pelvis"])
        self.map_check(world_map)
        self.coverage(world_map)
        t0, t1 = sent["sim"], s["t"][i_end]
        self.record_only(self.scan_path, t0, t1)
        self.record_only(self.scan_phantoms)
        self.record_only(self.drift, t0, t1)

    def explorer_outcome(self):
        """What the explorer did: its result, its goals (launch.log) and the time, for the record"""
        ev, s = self.events, self.s
        result = ev.get("explore_result", {})
        goals = explorer_goals(os.path.join(self.out, "launch.log"))
        failed = [g for g in goals if g["outcome"] == "failed"]
        seen = [g for g in goals if g["outcome"] == "seen on the way"]
        by_nav2 = [g for g in goals if g["outcome"] == "reached"]
        wall = result["wall"] - ev["explore_sent"]["wall"] if result else math.nan
        sim_s = result.get("sim_seconds", math.nan)
        if "explore_result" in ev and "explore_sent" in ev:
            a, b = self.at("explore_sent"), self.at("explore_result")
            if b > a:
                self.numbers["real_time_factor_exploring"] = float((s["t"][b] - s["t"][a]) /
                                                                   (s["wall"][b] - s["wall"][a]))
        status = result.get("status")
        outcome = {4: "done", 6: "aborted", 5: "canceled"}.get(status, f"no result (status {status})")
        self.numbers["explorer"] = {"outcome": outcome, "status": status, "message": result.get("message"),
                                    "goals_reached": result.get("goals_reached"), "goals": goals,
                                    "goals_sent": len(goals), "goals_failed": len(failed),
                                    "goals_reached_by_nav2": len(by_nav2),
                                    "goals_seen_on_the_way": len(seen), "sim_s": sim_s,
                                    "wall_s": wall, "feedback": result.get("feedback")}
        print(f"     explorer: {outcome} ({result.get('message')!r}) after {sim_s:.1f} s of sim time ({wall:.1f} s "
              f"wall; real-time factor {self.numbers.get('real_time_factor_exploring', math.nan):.2f}), "
              f"{len(goals)} goals sent, {result.get('goals_reached')} reached ({len(by_nav2)} by Nav2, {len(seen)} "
              f"seen on the way), {len(failed)} failed")
        for g in goals:
            print(f"       goal {g['n']}: ({g['x']:.2f}, {g['y']:.2f}) in map, {g['cells']} frontier cells, "
                  f"{g['distance']:.2f} m away: {g['outcome']}"
                  + (f" ({g['why']})" if g.get("why") else "")
                  + (f" after {g['seconds']:.1f} s" if g.get("seconds") is not None else ""))
        # a finished exploration reached its goals: by Nav2, or seen on the way
        reached = result.get("goals_reached")
        if outcome == "done":
            self.check("goals reached", reached is not None and reached > 0 and reached == len(by_nav2) + len(seen),
                       f"the result's goals_reached {reached} (must be > 0); launch.log: {len(by_nav2)} reached by "
                       f"Nav2 + {len(seen)} seen on the way = {len(by_nav2) + len(seen)}, of {len(goals)} sent")
        else:
            print(f"     goals reached: not checked, the explorer did not finish ({outcome})")

    def coverage(self, world_map):
        """The share of the reachable floor (scene.npz) that the final /map has free"""
        path = os.path.join(self.out, "map.npz")
        if not os.path.exists(path):
            self.check("coverage", False, "no /map")
            return
        m = np.load(path)
        scene = np.load(os.path.join(self.out, "scene.npz"))
        reachable, origin, gres = scene["reachable"], scene["origin"], float(scene["res"])
        rows, cols = np.nonzero(reachable)
        world = np.stack([origin[0] + (cols + 0.5) * gres, origin[1] + (rows + 0.5) * gres, np.ones(len(rows))])
        local = np.linalg.inv(world_map @ nav.se2(*m["origin"])) @ world
        res, data = float(m["res"]), m["data"]
        ix, iy = np.floor(local[0] / res).astype(int), np.floor(local[1] / res).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix < data.shape[1]) & (iy < data.shape[0])
        value = np.full(len(rows), -1)
        value[inside] = data[iy[inside], ix[inside]]
        free = (value >= 0) & (value < MAP_OCCUPIED)
        share = float(free.mean()) if len(rows) else 0.0
        # the grid must hold all of the reachable floor, or the share is of a part of it
        edge = int(reachable[[0, -1], :].sum() + reachable[:, [0, -1]].sum())
        unknown = ~free & (value < 0)
        self.numbers["coverage"] = {"reachable_cells": int(len(rows)), "reachable_m2": len(rows) * gres * gres,
                                    "free": int(free.sum()), "share": share, "unknown": int(unknown.sum()),
                                    "occupied": int((value >= MAP_OCCUPIED).sum()), "reachable_on_grid_edge": edge}
        self.check("coverage", share >= MIN_COVERAGE and edge == 0,
                   f"{free.sum()} of the {len(rows)} cells of reachable floor ({len(rows) * gres * gres:.1f} m^2) are "
                   f"free in the final /map: {share:.1%} (limit {MIN_COVERAGE:.0%}); {unknown.sum()} unknown, "
                   f"{(value >= MAP_OCCUPIED).sum()} occupied" + (f"; {edge} reachable cells on the edge of the sim's "
                                                                  "grid: it does not hold all of the floor" if edge
                                                                  else ""))

    def run(self, out):
        code = super().run(out)
        # the notes too
        with open(os.path.join(out, "report.json")) as f:
            report = json.load(f)
        report["notes"] = getattr(self, "notes", [])
        with open(os.path.join(out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("role", choices=("sim", "drive", "report"))
    parser.add_argument("--out", required=True, help="the shared output directory")
    parser.add_argument("--odom-source", choices=("fastlio", "sim_gt"), default="fastlio",
                        help="drive: the bringup's odom_source")
    parser.epilog = "sim: the G1 sim's arguments (h12_mujoco.py --g1) follow a --"
    argv = sys.argv[1:]
    sim_args = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:len(argv) - len(sim_args) - ("--" in argv)])
    if args.role == "sim":
        nav.run_sim(args.out, sim_args)
        return 0
    if args.role == "drive":
        return run_drive(args.out, args.odom_source)
    return Report(args.out).run(args.out)


if __name__ == "__main__":
    sys.exit(main())
