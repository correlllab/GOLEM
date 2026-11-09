"""Check the shared MID-360 model (golem_robocasa/mid360.py): pattern, scanner and scan ring.

The scanner and the ring cast through casters that need no GPU: the tests' exact reference
(exact_rays.ExactCaster, MuJoCo's mj_ray) and fakes. The scenes are small MJCFs (closed box
rooms), so no RoboCasa env is built. Needs MuJoCo, NumPy and sim_timing only; run it with
ROS_DOMAIN_ID=231 tests/robocasa/run_in_container.sh -p test_mid360.py
"""
import threading
import time
import unittest

import mujoco
import numpy as np

import mid360
from exact_rays import ExactCaster
from sim_timing import LoopTimings

FRAME_POINTS = 20_000                 # POINTS_PER_SECOND / FRAME_RATE_HZ
STEP_S = 0.01                         # s of sim time between update() calls (ExactCaster.batch_s)
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


def _scene(xml):
    """Compile xml and return (model, data) after mj_forward."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _cube_snapshot(half, rotation=np.eye(3)):
    """Return (caster, snapshot) for a sensor centred in a closed cube room, half-size half (m)."""
    model, data = _scene(_cube_room(half))
    caster = ExactCaster(model, exclude_body=-1, geomgroup=VISUAL)
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


class _RoomCaster:
    """Return each ray's distance to the walls of a closed cube room of half-size half (m) around the sensor.

    It reads no MjData, so the ring's snapshots, which hold only geom poses, cast the same.
    """

    def __init__(self, half):
        """Cast in a room of inner half-size half (m)."""
        self.half = half

    def cast(self, origin, rotation, directions, data, geom_alpha=None, mat_alpha=None):
        """Return half / the largest world component of each of directions (m)."""
        world = np.asarray(directions) @ np.asarray(rotation).T
        return self.half / np.abs(world).max(axis=1)

    def close(self):
        """Do nothing: the caster holds nothing."""


class _FakeCaster:
    """Return the snapshot's data, a distance in m, for every ray."""

    def cast(self, origin, rotation, directions, data, geom_alpha=None, mat_alpha=None):
        """Return data (m) for each of directions."""
        return np.full(len(directions), float(data))


class _PatternedCaster:
    """Return, by ray index in the call, no hit, a hit in the blind zone, past MAX_RANGE or at 2 m.

    Keeps every distance it returned, in order, in self.cast_dist.
    """

    def __init__(self):
        """Start with no casts."""
        self.cast_dist = []

    def cast(self, origin, rotation, directions, data, geom_alpha=None, mat_alpha=None):
        """Return NaN, 0.05 m, 50 m, 2 m and 2 m in turn for each of directions."""
        dist = np.resize([np.nan, 0.05, 50.0, 2.0, 2.0], len(directions))
        self.cast_dist.append(dist)
        return dist


class ScannerTests(unittest.TestCase):
    """Check Mid360Scanner's frames, timing, blind zone, noise and rays without a return."""

    def test_rays_without_a_return_are_zero_points(self):
        """Every ray fired is a point; one without a return is (0, 0, 0), as the driver publishes it.

        No hit, a hit in the blind zone and one past MAX_RANGE are all no return; each such
        point keeps its offset_time and line, so a frame is the pattern's 20,000 rays in order.
        """
        caster = _PatternedCaster()
        scanner = mid360.Mid360Scanner(caster)
        snapshot = mid360.Snapshot(np.zeros(3), np.eye(3), None)
        frames = _run(scanner, snapshot, _steps(0.0, 0.1))
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame.rays, FRAME_POINTS)
        self.assertEqual(len(frame.points), FRAME_POINTS)
        self.assertEqual(len(frame.offset_time), FRAME_POINTS)
        np.testing.assert_array_equal(frame.line, np.arange(FRAME_POINTS) % mid360.LINES)
        self.assertTrue(np.all(np.diff(frame.offset_time.astype(np.int64)) >= 0))
        ranges = np.linalg.norm(frame.points.astype(np.float64), axis=1)
        zero = np.all(frame.points == 0.0, axis=1)
        # The rays in the order they were cast; the call that ends the frame also casts the
        # next frame's first ray.
        cast = np.concatenate(caster.cast_dist)
        self.assertEqual(len(cast), FRAME_POINTS + 1)
        np.testing.assert_array_equal(zero, cast[:FRAME_POINTS] != 2.0)
        self.assertGreater(zero.sum(), 0)
        self.assertLess(np.abs(ranges[~zero] - 2.0).max(), 0.2)   # 10 sigma of the noise

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
        """Walls at 0.05 m give only zero points; at 2 m, ranges are unbiased with 2 cm of noise.

        The same seed gives the same frame.
        """
        caster, snapshot = _cube_snapshot(0.05)
        frames = _run(mid360.Mid360Scanner(caster), snapshot, _steps(0.0, 0.1))
        caster.close()
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].rays, FRAME_POINTS)
        self.assertEqual(len(frames[0].points), FRAME_POINTS)
        self.assertFalse(frames[0].points.any())

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

    def test_scanner_restarts_after_a_reset_inside_the_first_frame(self):
        """Time going back to the frame's own start drops the frame; no frame mixes both scenes.

        Before the reset every ray returns 2 m, after it 3 m.
        """
        scanner = mid360.Mid360Scanner(_FakeCaster())
        before = mid360.Snapshot(np.zeros(3), np.eye(3), 2.0)
        after = mid360.Snapshot(np.zeros(3), np.eye(3), 3.0)
        self.assertEqual(_run(scanner, before, _steps(0.0, 0.08)), [])
        frames = _run(scanner, after, _steps(0.0, 0.2))
        self.assertEqual([round(f.timebase, 9) for f in frames], [0.0, 0.1])
        for frame in frames:
            self.assertEqual(frame.rays, FRAME_POINTS)
            self.assertEqual(len(frame.points), FRAME_POINTS)
            ranges = np.linalg.norm(frame.points, axis=1)
            self.assertLess(np.abs(ranges - 3.0).max(), 0.2)   # 10 sigma of the noise


class ScanRingTests(unittest.TestCase):
    """Check ScanRing's depth, backpressure and frame lag."""

    def ring(self, cast_s=0.0, interval=STEP_S, timings=None):
        """Return a started ScanRing in a 3 m cube room whose casts take at least cast_s wall s.

        Frames go to self.frames, each with the ring's sim time when it was published.
        """
        model, data = _scene(_cube_room(3.0))
        caster = _RoomCaster(3.0)
        cast = caster.cast

        def slow(*args, **kwargs):
            """Cast, then sleep out the rest of cast_s."""
            start = time.perf_counter()
            dist = cast(*args, **kwargs)
            time.sleep(max(0.0, cast_s - (time.perf_counter() - start)))
            return dist

        caster.cast = slow
        self.frames = []
        ring = mid360.ScanRing(mid360.Mid360Scanner(caster), model,
                               lambda d: (np.zeros(3), UPSIDE_DOWN),
                               lambda frame: self.frames.append((frame, ring.now)), interval,
                               timings=timings)
        self.addCleanup(ring.close)
        self.data = data
        return ring

    def test_ring_holds_one_frame_period_of_casts(self):
        """The ring has as many slots as whole intervals fit in a frame period, at least one."""
        for interval, slots in ((0.01, 10), (0.02, 5), (0.1, 1), (0.03, 3), (0.15, 1)):
            with self.subTest(interval=interval):
                self.assertEqual(self.ring(interval=interval).slots, slots)

    def test_a_slow_caster_makes_tick_wait_and_no_frame_lags_a_frame_period(self):
        """With casts slower than real time, tick() waits for the thread instead of dropping rays.

        Casts of 25 ms every 10 ms of sim time: the physics loop below, which would run far
        faster than real time, is held to the casts' pace. Every frame still fires all 20,000
        rays (each hits the room), and at its publication the sim is at most one frame period
        past the frame's end. Each wait is timed as the lidar_wait stage.
        """
        timings = LoopTimings(enabled=True)
        ring = self.ring(cast_s=0.025, timings=timings)
        start = time.perf_counter()
        for k in range(301):   # 0.6 s of sim time in 2 ms steps
            ring.tick(self.data, k * 0.002)
        elapsed = time.perf_counter() - start
        ring.close()
        self.assertEqual(ring.casts, 61)
        self.assertGreater(ring.waits, 40)
        # 61 casts of 25 ms; at most one frame of casts (10) is still queued when the loop ends.
        self.assertGreater(elapsed, (61 - 10) * 0.025 * 0.9)
        self.assertEqual(len(self.frames), 6)
        for frame, now in self.frames:
            self.assertEqual(len(frame.points), FRAME_POINTS)
            end = frame.timebase + 0.1
            self.assertLessEqual(now - end, 0.1 + 1e-9)
        self.assertGreater(ring.max_lag, 0.05)
        self.assertLessEqual(ring.max_lag, 0.1 + 1e-9)
        self.assertIn(f"6 lidar frames ({ring.waits} of 61 casts waited", ring.stats())
        self.assertEqual(timings.stages["lidar_wait"][1], ring.waits)
        self.assertEqual(timings.stages["lidar_capture"][1], 61)

    def test_a_fast_caster_never_waits(self):
        """tick() does not wait, nor count a wait, while the casts keep up with the loop."""
        ring = self.ring()
        for k in range(101):
            while ring._free.empty():   # the loop outruns real time
                time.sleep(0.0005)
            ring.tick(self.data, k * 0.002)
        ring.close()
        self.assertEqual((ring.waits, ring.casts, len(self.frames)), (0, 21, 2))

    def test_tick_raises_when_the_scan_thread_has_stopped(self):
        """A full ring whose thread has stopped raises at once, on every later cast too."""
        ring = self.ring(interval=0.1)   # one slot
        release = threading.Event()
        ring.scanner.update = lambda snapshot, t: release.wait(30.0) and None
        ring.tick(self.data, 0.0)
        ring._work.put(None)   # the thread stops after the cast in progress
        release.set()
        ring._thread.join(10.0)
        ring.tick(self.data, 0.1)   # takes the freed slot; the stopped thread never frees it
        for k in (2, 3):
            start = time.perf_counter()
            with self.assertRaisesRegex(RuntimeError, "scan thread has stopped"):
                ring.tick(self.data, 0.1 * k)
            # well under mid360._SLOT_POLL_S: the wait for a slot is skipped
            self.assertLess(time.perf_counter() - start, 0.2)


if __name__ == "__main__":
    unittest.main()
