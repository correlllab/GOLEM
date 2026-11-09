"""Model the G1's Livox MID-360 in MuJoCo: its mount, its scans and its built-in IMU.

Geometry only (numpy + mujoco, no ROS): g1_sensor_msgs.py packs what this module
returns into the messages the Livox driver publishes (/livox/lidar CustomMsg,
/livox/imu Imu, both in livox_frame).

Frames. Everything is in the sensor's own frame, livox_frame, which is the
URDF's mid360_link (g1_robosuite puts a site of that name on torso_link at the
URDF pose: rpy pi 0.0511 0, so the sensor hangs upside down). The driver runs
with MID360_config.json's extrinsic at zero (as golem_slam's G1 FAST_LIO sim
config assumes), so the real points arrive in the same raw frame.

Scans. Both robots' MID-360 is the shared model in golem_robocasa/mid360.py: the
real non-repetitive scan pattern, the blind zone, the maximum range and the
range noise are there, with its GPU caster and its frame timing. This module gives
it the G1's mount: make_caster() casts from the mid360_link site and
make_scanner() fires POINTS_PER_FRAME rays per frame, the real sensor's 20,000
(200,000 points/s at 10 Hz). A frame costs a few ms of render time on the GPU.

Self returns. The sensor sits inside the head mesh, which belongs to torso_link,
so rays pass through torso_link's geoms (MuJoCo's bodyexclude); in the FOV that
is only the head shell, since torso_link is rigid with the sensor. A ray whose
first hit is any other robot geom (shoulders, wrists, hands: 3% of a frame's
rays with every joint at 0) gives no return (a zero point), so the scan has no
returns from the robot's own body and the body still occludes what is behind
it, as on the real robot. Only visual geoms (group 1) are hit, the surfaces the camera
renders; geoms with alpha 0, which in RoboCasa are placement regions and hidden
collision boxes, are not.

IMU. imu() gives the angular velocity (rad/s) and the specific force (in g, the
unit the driver publishes; lidar_imu_data_queue.h) of a point rigidly attached
at the IMU's place in the sensor (IMU_IN_LIDAR), in livox_frame axes (the IMU's
axes are the lidar's: FAST_LIO's extrinsic_R is identity). It reads MuJoCo's
cvel/cacc, which mj_forward/mj_step fill only when the model has an
acceleration-stage sensor; the G1 MJCF's accelerometers ensure that, and the
constructor checks it. No noise or bias is modelled.

State time. mj_step computes the kinematics, velocities and accelerations
(everything this module reads) for the state it starts from, then integrates,
so after mj_step they describe data.time - model.opt.timestep, not data.time.
The caller passes that time to the scanner's update() and stamps imu() with it;
after mj_forward they describe data.time.
"""
import mujoco
import numpy as np

import mid360

SITE = "mid360_link"
TORSO_BODY = "torso_link"

FRAME_RATE_HZ = mid360.FRAME_RATE_HZ
POINTS_PER_FRAME = int(mid360.POINTS_PER_SECOND / FRAME_RATE_HZ)   # the real sensor's 20,000
GEOM_GROUPS = (0, 1, 0, 0, 0, 0)     # visual geoms only

# The IMU's position in livox_frame, m. FAST_LIO's MID-360 extrinsic_T
# (-0.011, -0.02329, 0.04412) is the lidar's position in the IMU's frame
# (p_imu = R p_lidar + T, R identity), so the IMU sits at -T in the lidar's.
IMU_IN_LIDAR = np.array([0.011, 0.02329, -0.04412])
IMU_RATE_HZ = 200.0
STANDARD_GRAVITY = 9.80665           # m/s^2 per g

# Sensor types for which mj_forward runs mj_rnePostConstraint, filling cacc.
_CACC_SENSORS = [int(getattr(mujoco.mjtSensor, f"mjSENS_{name}")) for name in
                 ("ACCELEROMETER", "FORCE", "TORQUE", "FRAMELINACC", "FRAMEANGACC")]


def make_scanner(caster, points_per_frame=POINTS_PER_FRAME, rate_hz=FRAME_RATE_HZ):
    """Return a mid360.Mid360Scanner firing points_per_frame rays per frame, rate_hz frames per sim second."""
    return mid360.Mid360Scanner(caster, points_per_second=points_per_frame * rate_hz,
                                rate_hz=rate_hz)


class Mid360:
    """Locate the MID-360 on a G1 MjModel: its pose and snapshots, a caster for its scans, its IMU."""

    def __init__(self, model, prefix="robot0_"):
        """Find the sensor in model.

        prefix is the robot's name prefix ('robot0_' in a RoboCasa env, '' in
        the standalone G1 MJCF). Raises if the site is missing or not on
        torso_link, or if the model computes no acceleration-stage sensor.
        """
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + SITE)
        if self.site_id < 0:
            raise KeyError(f"site not found in model: {prefix + SITE}")
        self.torso_id = int(model.site_bodyid[self.site_id])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, self.torso_id) != prefix + TORSO_BODY:
            raise ValueError(f"{prefix + SITE} is not on {prefix + TORSO_BODY}")
        # The robot is the subtree of the free-floating pelvis the sensor hangs off.
        root = model.body_rootid[self.torso_id]
        self.robot_geom = model.body_rootid[model.geom_bodyid] == root
        if (not np.isin(model.sensor_type, _CACC_SENSORS).any()
                or model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR):
            raise ValueError("model computes no accelerometer-type sensor, so MuJoCo leaves "
                             "cacc stale and the IMU cannot be read")
        self._vel = np.zeros(6)
        self._acc = np.zeros(6)

    def pose(self, data):
        """Return the sensor's (position (3,) m, rotation (3, 3) livox_frame -> world)."""
        return (data.site_xpos[self.site_id].copy(),
                data.site_xmat[self.site_id].reshape(3, 3).copy())

    def snapshot(self, data, geom_alpha=None, mat_alpha=None):
        """Return the mid360.Snapshot a scan of data's kinematics reads.

        geom_alpha and mat_alpha are the model's live alpha (Snapshot's fields); data is
        referenced, not copied.
        """
        position, rotation = self.pose(data)
        return mid360.Snapshot(position, rotation, data, geom_alpha, mat_alpha)

    def make_caster(self):
        """Return a mid360.GpuCaster through torso_link, visual geoms only.

        Rays whose first hit is another robot geom give no return. The caster casts on its own
        copy of the model; close() it when done. Raises RuntimeError when EGL fails to render.
        """
        return mid360.GpuCaster(self.model, self.torso_id, GEOM_GROUPS, robot_geoms=self.robot_geom)

    def imu(self, data, point=IMU_IN_LIDAR):
        """Return (angular velocity rad/s, specific force g) at point, in livox_frame axes.

        point is m in livox_frame, by default the MID-360's IMU.
        """
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.site_id,
                                 self._vel, 1)
        mujoco.mj_objectAcceleration(self.model, data, mujoco.mjtObj.mjOBJ_SITE, self.site_id,
                                     self._acc, 1)
        omega, alpha, accel = self._vel[:3], self._acc[:3], self._acc[3:]
        # The site's specific force carried to the point on the same rigid body.
        accel = accel + np.cross(alpha, point) + np.cross(omega, np.cross(omega, point))
        return omega.copy(), accel / STANDARD_GRAVITY
