"""Model the Livox MID-360's scans in MuJoCo for both robots: the real scan pattern, cast exactly.

Geometry only (numpy + mujoco, no ROS): each robot's sensor code takes the sensor's pose
from its model, calls Mid360Scanner.update() and packs the LidarFrame it returns into the
messages the Livox driver publishes. The MID-360's IMU is not here: g1/g1_lidar.py (G1) and
h12_sensors.py (H1-2) model it.

Frames. Directions and points are in the sensor's own frame (livox_frame, the frame the
driver publishes in with MID360_config.json's extrinsic at zero); the caller gives the
sensor's pose in the world, origin and rotation (sensor -> world). Both robots hang the
sensor upside down, so that frame's +z points at the floor.

Pattern. third_party/mujoco_lidar/mid360.npy is the MID-360's non-repetitive pattern from
MuJoCo-LiDAR (its README there): 800,000 (azimuth, elevation) pairs in rad, in firing order.
Azimuth is measured from +x towards +y; elevation is positive towards +z, the sensor's own
up, and spans -7 to +52 deg. 500 consecutive points already sweep ~155 deg of azimuth and
20,000, one 10 Hz frame at 200,000 points/s, cover the whole field of view. The scanner
walks the pattern without restarting it at each frame, so consecutive frames fill each
other's gaps as the real sensor's do; the file repeats after 40 frames.

Timing. update() is called from the sensor's loop with the time of the snapshot's
kinematics. Each call casts the rays that came due since the last call, from that call's
pose, and labels them with its time, so offset_time always matches the pose a point was
measured from (FAST_LIO's motion compensation relies on it). Point k of a frame is due at
k / points_per_frame of the frame period. A frame's timebase is its first point's time, as
in the driver (lddc.cpp InitCustomMsg). The caller decides how often to call: every
caster's batch_s of sim time.

Returns. A ray gives a point when its first hit lies between MIN_RANGE (the blind zone)
and MAX_RANGE (the range at 10% reflectivity). The measured range is the true range plus
Gaussian noise of RANGE_NOISE (1 sigma; the MID-360's 2 cm range precision); the blind zone
and the maximum range apply to the true range. Rays without a return give no point.

Self returns. The sensor sits inside a shell of the body it is mounted on, so rays pass
through that body's geoms (MuJoCo's bodyexclude). A ray whose first hit is any other geom
of the robot gives no point, so the scan holds no returns from the robot's own limbs while
the limbs still occlude what is behind them. Only geoms in the caster's geomgroup are hit;
MuJoCo's ray functions also skip geoms with alpha 0 (RoboCasa's placement regions and
hidden collision boxes).
"""
import os
from concurrent.futures import ThreadPoolExecutor, wait
from typing import NamedTuple

import mujoco
import numpy as np

POINTS_PER_SECOND = 200_000          # the MID-360's point rate
FRAME_RATE_HZ = 10.0                 # the driver's publish_freq (msg_MID360_launch.py)
LINES = 4                            # kLineNumberMid360: the driver numbers points i % 4
MIN_RANGE = 0.1                      # m, the blind zone
MAX_RANGE = 40.0                     # m, the range at 10% reflectivity
RANGE_NOISE = 0.02                   # m, 1 sigma

PATTERN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "third_party", "mujoco_lidar", "mid360.npy")
# Worker threads of a CpuCaster: a quarter of the cores, at most 4, so the casts leave the
# physics and the cameras their cores.
DEFAULT_THREADS = max(1, min(4, (os.cpu_count() or 1) // 4))
# Rays below which another thread is not used: handing a chunk to a thread costs tens of us,
# the work of about 10 rays in a kitchen.
_MIN_RAYS_PER_THREAD = 250
_EPS_T = 1e-9                        # s, tolerance on accumulated sim time


class LidarFrame(NamedTuple):
    """Hold one published scan, the points in the order they were measured."""
    timebase: float            # s, time of the first ray (the CustomMsg's timebase and stamp)
    offset_time: np.ndarray    # uint32 ns after timebase, non-decreasing
    points: np.ndarray         # float32 (n, 3), m, in the sensor's own frame
    line: np.ndarray           # uint8, 0..LINES-1
    rays: int                  # rays fired in the frame, with a return or not


class Snapshot(NamedTuple):
    """Hold what one cast reads: the sensor's pose and the MjData of the scene's geom poses."""
    origin: np.ndarray         # (3,) m, the sensor's position in the world
    rotation: np.ndarray       # (3, 3), sensor -> world
    data: mujoco.MjData        # its geom_xpos and geom_xmat are what the rays hit


def load_pattern():
    """Return the MID-360's scan pattern: (800,000, 2) float64, azimuth and elevation in rad."""
    return np.load(PATTERN_FILE).astype(np.float64)


def directions(pattern, start, count):
    """Return the unit directions (count, 3) of pattern rows start.., wrapping past its end.

    The directions are in the sensor's own frame (see the module docstring).
    """
    rows = pattern[(start + np.arange(count)) % len(pattern)]
    azimuth, elevation = rows[:, 0], rows[:, 1]
    cos_el = np.cos(elevation)
    return np.stack([cos_el * np.cos(azimuth), cos_el * np.sin(azimuth), np.sin(elevation)],
                    axis=-1)


class CpuCaster:
    """Cast rays exactly with mj_multiRay, split over threads.

    Threads. cast() clears model.body_bvhadr for its duration, and mj_step's collision
    reads it: give the caster a model no other thread steps or casts on while it runs (as
    g1_sensors' scan thread casts on its own copy of the model), and call cast() from one
    thread at a time. The extra threads cast on scratch MjData of their own, since
    mj_multiRay allocates on the MjData it is given.
    """

    batch_s = 0.01             # s of sim time between the scanner's update() calls

    def __init__(self, model, exclude_body, geomgroup, threads=DEFAULT_THREADS,
                 robot_geoms=None):
        """Cast on model, through the geoms of body id exclude_body (-1: none).

        geomgroup is the 6 flags of the geom groups the rays hit. threads is how many
        threads cast at once, the calling one included. robot_geoms is a bool mask
        (ngeom,) of geoms whose first hit gives no return; by default every geom of
        exclude_body's root subtree (the robot the sensor is on), none with exclude_body -1.
        """
        self.model = model
        self.exclude_body = int(exclude_body)
        self._geomgroup = np.array(geomgroup, dtype=np.uint8).reshape(6, 1)
        if robot_geoms is not None:
            self.robot_geoms = np.asarray(robot_geoms, dtype=bool)
        elif self.exclude_body >= 0:
            root = model.body_rootid[self.exclude_body]
            self.robot_geoms = model.body_rootid[model.geom_bodyid] == root
        else:
            self.robot_geoms = np.zeros(model.ngeom, dtype=bool)
        # The scratch MjData carry geom poses only, so a model with flexes casts on the
        # caller's MjData alone.
        self.threads = 1 if model.nflex else max(1, int(threads))
        self._scratch = [mujoco.MjData(model) for _ in range(self.threads - 1)]
        self._pool = (ThreadPoolExecutor(self.threads - 1, thread_name_prefix="mid360_cast")
                      if self.threads > 1 else None)

    def close(self):
        """Stop the worker threads."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _cast_chunk(self, d, source, origin, world, geomid, dist, lo, hi):
        """Cast rays lo..hi on d, after copying source's geom poses into it unless source is d."""
        if source is not d:
            np.copyto(d.geom_xpos, source.geom_xpos)
            np.copyto(d.geom_xmat, source.geom_xmat)
        mujoco.mj_multiRay(self.model, d, origin, world[lo:hi].reshape(-1, 1), self._geomgroup,
                           1, self.exclude_body, geomid[lo:hi].reshape(-1, 1),
                           dist[lo:hi].reshape(-1, 1), hi - lo, MAX_RANGE)

    def cast(self, origin, rotation, directions, data):
        """Return the distance (m) along each unit direction (n, 3) to its first hit, NaN for none.

        origin (3,) m and rotation (3, 3) sensor -> world are the sensor's pose; directions
        are in its own frame; data holds the geom poses. NaN where nothing is hit within
        MAX_RANGE or the robot is hit first.
        """
        n = len(directions)
        dist = np.empty(n)
        geomid = np.empty(n, dtype=np.int32)
        if n == 0:
            return dist
        world = np.ascontiguousarray(np.asarray(directions) @ np.asarray(rotation).T,
                                     dtype=np.float64)
        origin = np.array(origin, dtype=np.float64).reshape(3, 1)
        chunks = min(self.threads, max(1, n // _MIN_RAYS_PER_THREAD))
        bounds = np.linspace(0, n, chunks + 1).astype(int)
        # mj_multiRay culls each geom by the azimuth and elevation of its bounding box's 8
        # corners as seen from the sensor, a range that can be narrower than the geom's: a
        # wall's bottom edge is seen lower than its corners, so rays in between pass through
        # the wall (3 to 5% of the rays in a RoboCasa kitchen, up to 1.6 m too far or with no
        # return). MuJoCo skips the culling for bodies without a BVH, so for this call none
        # has one; the result is then mj_ray's, ray for ray. Cleared once around every
        # thread's cast, and restored only once they have all finished.
        bvhadr = self.model.body_bvhadr.copy()
        self.model.body_bvhadr[:] = -1
        futures = []
        try:
            for k, scratch in enumerate(self._scratch[:chunks - 1], start=1):
                futures.append(self._pool.submit(self._cast_chunk, scratch, data, origin, world,
                                                 geomid, dist, bounds[k], bounds[k + 1]))
            self._cast_chunk(data, data, origin, world, geomid, dist, bounds[0], bounds[1])
            for future in futures:
                future.result()
        finally:
            wait(futures)
            self.model.body_bvhadr[:] = bvhadr
        hit = geomid >= 0
        robot = hit & self.robot_geoms[np.where(hit, geomid, 0)]
        dist[~hit | robot] = np.nan
        return dist


class Mid360Scanner:
    """Fire the MID-360's pattern through a caster and assemble its 10 Hz frames."""

    def __init__(self, caster, rng_seed=0, points_per_second=POINTS_PER_SECOND,
                 rate_hz=FRAME_RATE_HZ):
        """Scan with caster (its cast() as CpuCaster's); rates are per sim second.

        rng_seed seeds the range noise, so the same seed and calls give the same frames.
        """
        self.caster = caster
        self.pattern = load_pattern()
        self.points_per_frame = int(round(points_per_second / rate_hz))
        self.period = 1.0 / rate_hz
        self._rng_seed = rng_seed
        n = self.points_per_frame
        self._dist = np.empty(n)
        self._when = np.empty(n)
        self._line = (np.arange(n) % LINES).astype(np.uint8)
        self.reset()

    def reset(self):
        """Drop the frame in progress; the next update() starts a frame at the pattern's start.

        The range noise restarts from rng_seed.
        """
        self._start = None
        self._pattern_start = 0
        self._rng = np.random.default_rng(self._rng_seed)

    def _begin(self, t):
        """Start the next frame at sim time t, at the pattern row after the last frame's."""
        self._start = t
        self._timebase = None
        self._next = 0
        self._directions = directions(self.pattern, self._pattern_start, self.points_per_frame)
        self._pattern_start = (self._pattern_start + self.points_per_frame) % len(self.pattern)

    def _cast_until(self, snapshot, t, count):
        """Cast the frame's rays up to index count from snapshot's pose, labelled with time t."""
        first, last = self._next, count
        if last <= first:
            return
        if self._timebase is None:
            self._timebase = t
        self._dist[first:last] = self.caster.cast(snapshot.origin, snapshot.rotation,
                                                  self._directions[first:last], snapshot.data)
        self._when[first:last] = t
        self._next = last

    def _finish(self):
        """Return the frame's returns, with range noise, as a LidarFrame."""
        with np.errstate(invalid="ignore"):   # NaN, no hit, compares False
            valid = (self._dist >= MIN_RANGE) & (self._dist <= MAX_RANGE)
        measured = self._dist[valid] + self._rng.normal(0.0, RANGE_NOISE, int(valid.sum()))
        offset = np.round((self._when[valid] - self._timebase) * 1e9)
        points = self._directions[valid] * measured[:, None]
        return LidarFrame(self._timebase, offset.astype(np.uint32), points.astype(np.float32),
                          self._line[valid].copy(), self.points_per_frame)

    def update(self, snapshot, t):
        """Cast the rays due by sim time t from snapshot; return the frame t completed, else None.

        t (s) is the time of the snapshot's kinematics. Time going back (a sim reset)
        drops the frame in progress and starts again as after reset().
        """
        frame = None
        if self._start is None or t < self._start - _EPS_T:   # first call, or the sim was reset
            self.reset()
            self._begin(t)
        elif t - self._start >= self.period - _EPS_T:
            self._cast_until(snapshot, t, self.points_per_frame)
            frame = self._finish()
            following = self._start + self.period
            self._begin(following if t - following < self.period - _EPS_T else t)
        due = int(np.floor((t - self._start) / self.period * self.points_per_frame + 1e-6)) + 1
        self._cast_until(snapshot, t, min(due, self.points_per_frame))
        return frame
