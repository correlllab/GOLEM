"""Check the shared MID-360 model (golem_robocasa/mid360.py): pattern, CPU caster and scanner.

The scenes are small MJCFs built here (closed box rooms, and a room with a stand-in robot
for the caster), so no RoboCasa env is built. Needs MuJoCo and NumPy only; run it with
ROS_DOMAIN_ID=231 tests/robocasa/run_in_container.sh -p test_mid360.py
"""
import unittest

import mujoco
import numpy as np

import mid360

FRAME_POINTS = 20_000                 # POINTS_PER_SECOND / FRAME_RATE_HZ
STEP_S = 0.01                         # s of sim time between update() calls (CpuCaster.batch_s)
VISUAL = (0, 1, 0, 0, 0, 0)           # geomgroup: visual geoms only, as the G1 casts
UPSIDE_DOWN = np.diag([1.0, -1.0, -1.0])   # sensor -> world, rpy pi 0 0, as both robots hang it


def _cube_room(half):
    """Return an MJCF of a closed cube room, inner half-size half (m), centred on the origin.

    The walls are 0.1 m thick visual boxes (group 1); the sensor sits at the origin.
    """
    walls = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = sign * (half + 0.05)
            size = [half + 0.1] * 3
            size[axis] = 0.05
            walls.append(f'<geom type="box" group="1" pos="{pos[0]} {pos[1]} {pos[2]}" '
                         f'size="{size[0]} {size[1]} {size[2]}"/>')
    return f'<mujoco><worldbody>{"".join(walls)}</worldbody></mujoco>'


# A 4 x 4 m room with a floor and ceiling, furniture of several geom types (a tilted box, a
# cylinder, a sphere, a mesh) and a stand-in robot: the sensor sits in a head shell on
# "torso" (the excluded body), with an arm beside it and a pelvis below, all one subtree.
# Seen from the sensor, each wall's bottom edge straight ahead is lower than its corners, the
# geometry that makes mj_multiRay's culling miss it.
_KITCHEN = """
<mujoco>
  <asset>
    <mesh name="tetra" vertex="0 0 0  0.6 0 0  0 0.6 0  0 0 0.6"/>
  </asset>
  <worldbody>
    <geom type="box" group="1" pos="0 0 -0.05" size="2.1 2.1 0.05"/>
    <geom type="box" group="1" pos="0 0 2.55" size="2.1 2.1 0.05"/>
    <geom type="box" group="1" pos="2.05 0 1.25" size="0.05 2.1 1.3"/>
    <geom type="box" group="1" pos="-2.05 0 1.25" size="0.05 2.1 1.3"/>
    <geom type="box" group="1" pos="0 2.05 1.25" size="2.1 0.05 1.3"/>
    <geom type="box" group="1" pos="0 -2.05 1.25" size="2.1 0.05 1.3"/>
    <geom type="box" group="1" pos="1.2 0.8 0.45" size="0.4 0.3 0.45" euler="0 0 30"/>
    <geom type="cylinder" group="1" pos="-1.0 1.1 0.5" size="0.25 0.5"/>
    <geom type="sphere" group="1" pos="-1.2 -1.0 0.3" size="0.3"/>
    <geom type="mesh" group="1" mesh="tetra" pos="0.9 -1.2 0.0" euler="0 0 20"/>
    <geom type="box" group="0" pos="1.5 -0.5 0.3" size="0.2 0.2 0.3"/>
    <body name="pelvis" pos="0 0 0.9">
      <freejoint/>
      <geom type="box" group="1" size="0.12 0.15 0.08"/>
      <body name="torso" pos="0 0 0.35">
        <geom type="sphere" group="1" size="0.1"/>
        <site name="lidar" pos="0 0 0" euler="180 0 0"/>
        <body name="arm" pos="0 -0.22 -0.05">
          <geom type="capsule" group="1" fromto="0 0 0 0.15 0 -0.35" size="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _scene(xml):
    """Compile xml and return (model, data) after mj_forward."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _cube_snapshot(half, rotation=np.eye(3)):
    """Return (caster, snapshot) for a sensor centred in a closed cube room, half-size half (m)."""
    model, data = _scene(_cube_room(half))
    caster = mid360.CpuCaster(model, exclude_body=-1, geomgroup=VISUAL, threads=2)
    return caster, mid360.Snapshot(np.zeros(3), rotation, data)


def _run(scanner, snapshot, times):
    """Call scanner.update at each time and return the frames it completed."""
    frames = []
    for t in times:
        frame = scanner.update(snapshot, t)
        if frame is not None:
            frames.append(frame)
    return frames


def _steps(start, stop):
    """Return the sim times start, start + STEP_S, ..., stop (s), without accumulated rounding."""
    count = int(round((stop - start) / STEP_S))
    return [start + i * STEP_S for i in range(count + 1)]


class PatternTests(unittest.TestCase):
    """Check the vendored MuJoCo-LiDAR pattern and the directions built from it."""

    def test_pattern_shape_and_range(self):
        """The pattern holds 800,000 (azimuth, elevation) pairs inside the MID-360's view."""
        pattern = mid360.load_pattern()
        self.assertEqual(pattern.shape, (800_000, 2))
        self.assertEqual(pattern.dtype, np.float64)
        self.assertGreaterEqual(pattern[:, 0].min(), 0.0)
        self.assertLessEqual(pattern[:, 0].max(), 2.0 * np.pi + 1e-6)   # float32 2 pi rounds up
        self.assertGreaterEqual(pattern[:, 1].min(), -0.127)
        self.assertLessEqual(pattern[:, 1].max(), 0.911)

    def test_directions_wrap_around_the_pattern(self):
        """directions() gives unit vectors, elevation up, continuing at row 0 past the last row."""
        pattern = mid360.load_pattern()
        tail = mid360.directions(pattern, len(pattern) - 3, 6)
        np.testing.assert_allclose(np.linalg.norm(tail, axis=1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(tail[3:], mid360.directions(pattern, 0, 3))
        np.testing.assert_allclose(np.arcsin(tail[:3, 2]), pattern[-3:, 1], atol=1e-12)


class CpuCasterTests(unittest.TestCase):
    """Compare CpuCaster with MuJoCo's exact per-ray mj_ray."""

    def test_cpu_caster_matches_mj_ray(self):
        """2,000 rays on the furnished room equal per-ray mj_ray within 1e-9 m and on hit or miss.

        A ray whose first hit is the robot (other than the excluded torso) has no return.
        The scene must catch mj_multiRay's culling: with the BVH left in place some rays differ.
        """
        model, data = _scene(_KITCHEN)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar")
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        origin = data.site_xpos[site].copy()
        rotation = data.site_xmat[site].reshape(3, 3).copy()
        local = mid360.directions(mid360.load_pattern(), 123_456, 2000)
        world = local @ rotation.T
        geomgroup = np.array(VISUAL, dtype=np.uint8).reshape(6, 1)
        robot = model.body_rootid[model.geom_bodyid] == model.body_rootid[torso]

        expected = np.empty(len(world))
        robot_hits = 0
        geomid = np.empty((1, 1), dtype=np.int32)
        for i, vec in enumerate(world):
            dist = mujoco.mj_ray(model, data, origin.reshape(3, 1), vec.reshape(3, 1), geomgroup,
                                 1, torso, geomid)
            hit_robot = geomid[0, 0] >= 0 and robot[geomid[0, 0]]
            robot_hits += hit_robot
            miss = geomid[0, 0] < 0 or hit_robot or dist > mid360.MAX_RANGE
            expected[i] = np.nan if miss else dist
        self.assertGreater(robot_hits, 0, "no ray hits the robot")
        self.assertGreater(np.isfinite(expected).sum(), 1500)

        # The culling this caster works around misses some of these rays.
        culled_id = np.empty(len(world), dtype=np.int32)
        culled = np.empty(len(world))
        mujoco.mj_multiRay(model, data, origin.reshape(3, 1),
                           np.ascontiguousarray(world).reshape(-1, 1), geomgroup, 1, torso,
                           culled_id.reshape(-1, 1), culled.reshape(-1, 1), len(world),
                           mid360.MAX_RANGE)
        culled[(culled_id < 0) | robot[np.where(culled_id >= 0, culled_id, 0)]] = np.nan
        differ = (np.isnan(culled) != np.isnan(expected)) | (np.abs(culled - expected) > 1e-6)
        self.assertGreater(differ.sum(), 0, "the scene does not trigger mj_multiRay's culling")

        bvhadr = model.body_bvhadr.copy()
        caster = mid360.CpuCaster(model, exclude_body=torso, geomgroup=VISUAL, threads=4)
        try:
            got = caster.cast(origin, rotation, local, data)
        finally:
            caster.close()
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
        hit = np.isfinite(expected)
        np.testing.assert_allclose(got[hit], expected[hit], rtol=0.0, atol=1e-9)
        np.testing.assert_array_equal(model.body_bvhadr, bvhadr)

    def test_bvh_restored_when_a_cast_raises(self):
        """A cast that raises leaves model.body_bvhadr as it was."""
        model, _ = _scene(_KITCHEN)
        bvhadr = model.body_bvhadr.copy()
        caster = mid360.CpuCaster(model, exclude_body=-1, geomgroup=VISUAL, threads=2)
        local = mid360.directions(mid360.load_pattern(), 0, 4000)
        try:
            with self.assertRaises(Exception):
                caster.cast(np.zeros(3), np.eye(3), local, None)   # no MjData to cast in
        finally:
            caster.close()
        np.testing.assert_array_equal(model.body_bvhadr, bvhadr)


class ScannerTests(unittest.TestCase):
    """Check Mid360Scanner's frames, timing, blind zone and noise."""

    def test_frames_are_20000_points_at_10hz(self):
        """1 s of updates every 10 ms in a closed room gives 10 frames of 20,000 rays.

        Offsets are non-decreasing and at most 100 ms, line is the point's index mod 4, and the
        pattern continues from one frame to the next.
        """
        caster, snapshot = _cube_snapshot(3.0)
        scanner = mid360.Mid360Scanner(caster)
        pattern = mid360.load_pattern()
        frames = _run(scanner, snapshot, _steps(0.0, 1.0))
        caster.close()
        self.assertEqual(len(frames), 10)
        for k, frame in enumerate(frames):
            self.assertAlmostEqual(frame.timebase, 0.1 * k, places=9)
            self.assertEqual(frame.rays, FRAME_POINTS)
            self.assertEqual(len(frame.points), FRAME_POINTS)   # every ray hits a wall
            self.assertEqual(frame.offset_time.dtype, np.uint32)
            self.assertEqual(frame.points.dtype, np.float32)
            self.assertEqual(frame.line.dtype, np.uint8)
            self.assertTrue(np.all(np.diff(frame.offset_time.astype(np.int64)) >= 0))
            # The frame's last rays are cast by the call at its end, labelled with that time.
            self.assertLessEqual(int(frame.offset_time.max()), 100_000_000)
            np.testing.assert_array_equal(frame.line, np.arange(FRAME_POINTS) % mid360.LINES)
            unit = frame.points / np.linalg.norm(frame.points, axis=1, keepdims=True)
            np.testing.assert_allclose(
                unit, mid360.directions(pattern, k * FRAME_POINTS, FRAME_POINTS), atol=1e-5)

    def test_blind_zone_and_noise(self):
        """Walls at 0.05 m give no points; at 2 m, ranges are unbiased with 2 cm of noise.

        The same seed gives the same frame.
        """
        caster, snapshot = _cube_snapshot(0.05)
        frames = _run(mid360.Mid360Scanner(caster), snapshot, _steps(0.0, 0.1))
        caster.close()
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].rays, FRAME_POINTS)
        self.assertEqual(len(frames[0].points), 0)

        caster, snapshot = _cube_snapshot(2.0, UPSIDE_DOWN)
        first = _run(mid360.Mid360Scanner(caster, rng_seed=7), snapshot, _steps(0.0, 0.1))
        again = _run(mid360.Mid360Scanner(caster, rng_seed=7), snapshot, _steps(0.0, 0.1))
        caster.close()
        points = first[0].points.astype(np.float64)
        self.assertEqual(len(points), FRAME_POINTS)
        measured = np.linalg.norm(points, axis=1)
        world = (points / measured[:, None]) @ UPSIDE_DOWN.T
        error = measured - 2.0 / np.abs(world).max(axis=1)
        self.assertLess(abs(error.mean()), 0.003)
        self.assertGreater(error.std(), 0.017)
        self.assertLess(error.std(), 0.023)
        for field in ("offset_time", "points", "line"):
            np.testing.assert_array_equal(getattr(first[0], field), getattr(again[0], field))

    def test_scanner_restarts_after_time_goes_back(self):
        """Time going from 5.0 s back to 0.0 starts a new frame at 0.0; no frame spans the jump."""
        caster, snapshot = _cube_snapshot(3.0)
        scanner = mid360.Mid360Scanner(caster)
        before = _run(scanner, snapshot, _steps(4.5, 5.0))
        self.assertTrue(before)
        self.assertIsNone(scanner.update(snapshot, 0.0))
        after = _run(scanner, snapshot, _steps(0.01, 0.2))
        caster.close()
        self.assertEqual([round(f.timebase, 9) for f in after], [0.0, 0.1])
        for frame in after:
            self.assertEqual(frame.rays, FRAME_POINTS)
            self.assertLessEqual(int(frame.offset_time.max()), 100_000_000)


if __name__ == "__main__":
    unittest.main()
