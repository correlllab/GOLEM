"""The G1's Livox MID-360 in MuJoCo: ray-cast scans and the sensor's built-in IMU.

Geometry only (numpy + mujoco, no ROS): g1_sensor_msgs.py packs what this module
returns into the messages the Livox driver publishes (/livox/lidar CustomMsg,
/livox/imu Imu, both in livox_frame).

Frames. Everything is in the sensor's own frame, livox_frame, which is the
URDF's mid360_link (g1_robosuite puts a site of that name on torso_link at the
URDF pose: rpy pi 0.0511 0, so the sensor hangs upside down). The driver runs
with MID360_config.json's extrinsic at zero (as golem_slam's G1 FAST_LIO sim
config assumes), so the real points arrive in the same raw frame.

Scan. The MID-360 covers 360 deg around and -7 to +52 deg in elevation in its
own frame (upside down: from 7 deg above the horizon to 52 deg below), at about
200,000 points/s, published as 10 Hz frames. The real scan pattern is Livox's
non-repetitive one; this module approximates it: within a frame the azimuth
sweeps once around at a constant rate, the elevation of point k follows the
golden-ratio sequence (uniform in solid angle, a Fibonacci lattice per frame),
and both continue across frames so consecutive frames fill each other's gaps.
Point k of a frame is due at k / points_per_frame of the frame period.

Point count. An exact ray costs 4 to 6 us in a RoboCasa kitchen (cast() tests
every visible geom, see there), so the real 20,000 points per frame would take
1.2 s of compute per sim second. POINTS_PER_FRAME is 5,000 (50,000 points/s):
about 0.3 s per sim second, and about what FAST_LIO keeps of a real frame (its
MID-360 config's point_filter_num 3 of ~20,000). The cost is linear in it.

Timing. update() is called from the sim loop with the time of the current
kinematics. Each call casts the rays that came due since the last call, from
the sensor pose at that moment, and labels them with that moment, so
offset_time always matches the pose a point was measured from (FAST_LIO's
motion compensation relies on it). Called every physics step, the labels step
with the physics; called less often, points are grouped coarser in time but
still labelled with the pose they were cast from. A frame's timebase is its
first point's time, as in the driver (lddc.cpp InitCustomMsg).

Self returns. The sensor sits inside the head mesh, which belongs to torso_link,
so rays pass through torso_link's geoms (MuJoCo's bodyexclude); in the FOV that
is only the head shell, since torso_link is rigid with the sensor. A ray whose
first hit is any other robot geom (shoulders, wrists, hands: 3.3% of the rays
with every joint at 0) gives no point, so the scan has no returns from the
robot's own body and the body still occludes what is behind it, as on the real
robot. Only visual geoms (group 1) are hit, the surfaces the camera renders;
MuJoCo's ray functions also skip geoms with alpha 0, which in RoboCasa are
placement regions and hidden collision boxes. Rays with no hit within MAX_RANGE
give no point (the real driver may instead publish them as zero points; not
checked).

IMU. imu() gives the angular velocity (rad/s) and the specific force (in g, the
unit the driver publishes; lidar_imu_data_queue.h) of a point rigidly attached
at the IMU's place in the sensor (IMU_IN_LIDAR), in livox_frame axes (the IMU's
axes are the lidar's: FAST_LIO's extrinsic_R is identity). It reads MuJoCo's
cvel/cacc, which mj_forward/mj_step fill only when the model has an
acceleration-stage sensor; the G1 MJCF's accelerometers ensure that, and the
constructor checks it. No noise or bias is modelled.

Threads. cast() briefly clears model.body_bvhadr (see there): call update() and
cast() from the thread that steps the physics, between steps, as the sim loop
calls its bridge.

State time. mj_step computes the kinematics, velocities and accelerations
(everything this module reads) for the state it starts from, then integrates,
so after mj_step they describe data.time - model.opt.timestep, not data.time.
The caller passes that time to update() and stamps imu() with it; after
mj_forward they describe data.time.
"""
from typing import NamedTuple

import mujoco
import numpy as np

SITE = "mid360_link"
TORSO_BODY = "torso_link"

FRAME_RATE_HZ = 10.0                 # the driver's publish_freq (msg_MID360_launch.py)
POINTS_PER_FRAME = 5000              # the MID-360 has ~20,000 (see the docstring)
ELEVATION_DEG = (-7.0, 52.0)         # in the sensor's own frame
MIN_RANGE = 0.1                      # m, the MID-360's minimum range
MAX_RANGE = 40.0                     # m, its range at 10% reflectivity
LINES = 4                            # kLineNumberMid360: the driver numbers points i % 4
GEOM_GROUPS = (0, 1, 0, 0, 0, 0)     # visual geoms only

# The IMU's position in livox_frame, m. FAST_LIO's MID-360 extrinsic_T
# (-0.011, -0.02329, 0.04412) is the lidar's position in the IMU's frame
# (p_imu = R p_lidar + T, R identity), so the IMU sits at -T in the lidar's.
IMU_IN_LIDAR = np.array([0.011, 0.02329, -0.04412])
IMU_RATE_HZ = 200.0
STANDARD_GRAVITY = 9.80665           # m/s^2 per g

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # 1 / golden ratio
_EPS_T = 1e-9                         # s, tolerance on accumulated sim time
# sensors for which mj_forward runs mj_rnePostConstraint, filling cacc
_CACC_SENSORS = [int(getattr(mujoco.mjtSensor, f"mjSENS_{name}")) for name in
                 ("ACCELEROMETER", "FORCE", "TORQUE", "FRAMELINACC", "FRAMEANGACC")]


class LidarFrame(NamedTuple):
    """One published scan; the points in the order they were measured."""
    timebase: float            # s, time of the first point (the CustomMsg's timebase and stamp)
    offset_time: np.ndarray    # uint32 ns after timebase, non-decreasing
    points: np.ndarray         # float32 (n, 3), m, in livox_frame
    line: np.ndarray           # uint8, 0..LINES-1


def scan_pattern(frame_index, points_per_frame=POINTS_PER_FRAME, elevation_deg=ELEVATION_DEG):
    """(unit directions (n, 3) in livox_frame, line numbers) of one frame's rays, in firing order"""
    k = np.arange(points_per_frame)
    shift = (frame_index * _GOLDEN) % 1.0
    azimuth = 2.0 * np.pi * (k + shift) / points_per_frame
    g = frame_index * points_per_frame + k   # the elevation sequence runs on across frames
    low, high = np.sin(np.radians(elevation_deg))
    elevation = np.arcsin(low + (high - low) * ((g * _GOLDEN) % 1.0))
    cos_el = np.cos(elevation)
    directions = np.stack([cos_el * np.cos(azimuth), cos_el * np.sin(azimuth), np.sin(elevation)],
                          axis=-1)
    return directions, (k % LINES).astype(np.uint8)


class Mid360:
    """The MID-360 on a G1 MjModel: frames from update(), IMU samples from imu().

    prefix is the robot's name prefix in the model ('robot0_' in a RoboCasa env,
    '' in the standalone G1 MJCF in CL_Assets/g1).
    """

    def __init__(self, model, prefix="robot0_", points_per_frame=POINTS_PER_FRAME,
                 rate_hz=FRAME_RATE_HZ, min_range=MIN_RANGE, max_range=MAX_RANGE):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + SITE)
        if self.site_id < 0:
            raise KeyError(f"site not found in model: {prefix + SITE}")
        self.torso_id = int(model.site_bodyid[self.site_id])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, self.torso_id) != prefix + TORSO_BODY:
            raise ValueError(f"{prefix + SITE} is not on {prefix + TORSO_BODY}")
        # the robot is the subtree of the free-floating pelvis the sensor hangs off
        root = model.body_rootid[self.torso_id]
        self.robot_geom = model.body_rootid[model.geom_bodyid] == root
        if (not np.isin(model.sensor_type, _CACC_SENSORS).any()
                or model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR):
            raise ValueError("model computes no accelerometer-type sensor, so MuJoCo leaves "
                             "cacc stale and the IMU cannot be read")
        self.points_per_frame = int(points_per_frame)
        self.period = 1.0 / rate_hz
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self._geomgroup = np.array(GEOM_GROUPS, dtype=np.uint8).reshape(6, 1)
        n = self.points_per_frame
        self._dist = np.empty(n)
        self._geomid = np.empty(n, dtype=np.int32)
        self._when = np.empty(n)
        self._vel = np.zeros(6)
        self._acc = np.zeros(6)
        self.reset()

    def reset(self):
        """Drop the frame in progress; the next update() starts frame 0."""
        self._frame_index = -1
        self._start = None

    # ----------------------------------------------------------------- geometry
    def pose(self, data):
        """(position (3,), rotation (3, 3) livox_frame -> world) of the sensor"""
        return (data.site_xpos[self.site_id].copy(),
                data.site_xmat[self.site_id].reshape(3, 3).copy())

    def cast(self, data, directions, dist=None, geomid=None):
        """Distance (m) along each unit direction (n, 3) in livox_frame to the first
        surface, from the current pose; NaN where there is no point (no hit within
        max_range, closer than min_range, or the robot's own body in the way)."""
        n = len(directions)
        dist = np.empty(n) if dist is None else dist
        geomid = np.empty(n, dtype=np.int32) if geomid is None else geomid
        if n == 0:
            return dist
        position, rotation = self.pose(data)
        world = np.ascontiguousarray(directions @ rotation.T, dtype=np.float64)
        # mj_multiRay culls each geom by the azimuth and elevation of its bounding box's
        # 8 corners as seen from the sensor, a range that can be narrower than the geom's:
        # a wall's bottom edge is seen lower than its corners, so rays in between pass
        # through the wall (3 to 5% of the rays in a RoboCasa kitchen, up to 1.6 m too far
        # or with no return). MuJoCo skips the culling for bodies without a BVH, so for
        # this call none has one; the result is then mj_ray's, ray for ray.
        bvhadr = self.model.body_bvhadr.copy()
        self.model.body_bvhadr[:] = -1
        try:
            mujoco.mj_multiRay(self.model, data, position.reshape(3, 1), world.reshape(-1, 1),
                               self._geomgroup, 1, self.torso_id,
                               geomid.reshape(-1, 1), dist.reshape(-1, 1), n, self.max_range)
        finally:
            self.model.body_bvhadr[:] = bvhadr
        hit = geomid >= 0
        robot = hit & self.robot_geom[np.where(hit, geomid, 0)]
        dist[~hit | robot | (dist < self.min_range) | (dist > self.max_range)] = np.nan
        return dist

    # -------------------------------------------------------------------- scans
    def _begin(self, t):
        self._frame_index += 1
        self._start = t
        self._timebase = None
        self._next = 0
        self._directions, self._line = scan_pattern(self._frame_index, self.points_per_frame)

    def _cast_until(self, data, t, count):
        first, last = self._next, count
        if last <= first:
            return
        if self._timebase is None:
            self._timebase = t
        self.cast(data, self._directions[first:last], self._dist[first:last],
                  self._geomid[first:last])
        self._when[first:last] = t
        self._next = last

    def _finish(self):
        valid = np.isfinite(self._dist)
        offset = np.round((self._when[valid] - self._timebase) * 1e9)
        points = self._directions[valid] * self._dist[valid, None]
        return LidarFrame(self._timebase, offset.astype(np.uint32), points.astype(np.float32),
                          self._line[valid].copy())

    def update(self, data, t):
        """Cast the rays due by time t (the time of data's kinematics, see the
        module docstring). Returns the frame that t completed, else None."""
        frame = None
        if self._start is None or t < self._start - _EPS_T:   # first call, or the sim was reset
            self.reset()
            self._begin(t)
        elif t - self._start >= self.period - _EPS_T:
            self._cast_until(data, t, self.points_per_frame)
            frame = self._finish()
            following = self._start + self.period
            self._begin(following if t - following < self.period - _EPS_T else t)
        due = int(np.floor((t - self._start) / self.period * self.points_per_frame + 1e-6)) + 1
        self._cast_until(data, t, min(due, self.points_per_frame))
        return frame

    # ---------------------------------------------------------------------- IMU
    def imu(self, data, point=IMU_IN_LIDAR):
        """(angular velocity rad/s, specific force in g) at point (m, livox_frame) on the
        sensor, in livox_frame axes; by default the MID-360's IMU"""
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.site_id,
                                 self._vel, 1)
        mujoco.mj_objectAcceleration(self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.site_id,
                                     self._acc, 1)
        omega, alpha, accel = self._vel[:3], self._acc[:3], self._acc[3:]
        # the site's specific force carried to the point on the same rigid body
        accel = accel + np.cross(alpha, point) + np.cross(omega, np.cross(omega, point))
        return omega.copy(), accel / STANDARD_GRAVITY
