"""Define the H1-2 for golem_sim.sim_loop: its spawn, band, sensors, grippers and estop box."""
import os
import threading

import mujoco
import numpy as np

import h1_2_robosuite  # registers H1_2 and the MAGPIE grippers
import mid360
import realsense_profile
import robosuite_common
import sim_common
import battery_env  # registers the Battery env
import blank_env  # registers the Blank env
from h12_sensors import H12Sensors
from magpie_hand_bridge import MagpieHandBridge
from mujoco_env import ElasticBand, FallLogger
from estop_server import start_estop_server
from sim_names import NameResolver
from unitree_interface import SimInterface

NUM_MOTORS = 27
# The spawn pose (rad) in sim_names.ROS_MOTOR_ORDER: bent-knee legs, then the
# torso and both arms at 0.
INIT_LEG_POS = np.array([
    0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
    0.0, -0.16, 0.0, 0.36, -0.2, 0.0,
], dtype=float)
INIT_ARM_POS = np.array([
    0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=float)


def _initial_motor_qpos():
    """Return the 27-motor spawn pose (rad), without reading any controller config."""
    qpos = np.concatenate([INIT_LEG_POS, INIT_ARM_POS])
    if qpos.shape != (NUM_MOTORS,):
        raise ValueError(f"initial motor pose must contain {NUM_MOTORS} entries")
    return qpos


# Slots in ROS_MOTOR_ORDER: 12 torso, 13-19 left arm, 20-26 right arm, each arm
# shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw.
_TORSO, _ARMS = 12, (13, 20)


def _limp_upper_body_poses(model, motor_qpos, spawn):
    """Return the poses the unpowered upper body sags into, as (qpos indices, values) pairs.

    The spawn clearance must hold in them too, since every motor is off until
    the first command. Measured in RoboCasa: the torso twists up to ~0.7 rad
    either way, the shoulders pitch back ~0.25 rad, the elbows bend ~0.45 rad
    and the wrists drop onto their pitch stops, swinging the grippers forward.
    """
    poses = []
    for twist in (0.75, -0.75):
        q = np.array(spawn, dtype=float)
        q[_TORSO] = twist
        for base in _ARMS:
            q[base + 0] = 0.25    # shoulder pitch (rad, + is backward)
            q[base + 2] = -0.2    # shoulder yaw
            q[base + 3] = 0.45    # elbow
            wrist_pitch = int(np.flatnonzero(model.jnt_qposadr == motor_qpos[base + 5])[0])
            q[base + 5] = model.jnt_range[wrist_pitch, 0]
        poses.append((motor_qpos, q))
    return poses


# The arm postures (rad, left arm then right, each shoulder pitch/roll/yaw, elbow, wrist
# roll/pitch/yaw) golem_upper_ik's frame_task_server moves through at start, while the band
# still holds the robot at its spawn: golem_upper_ik/config/h12/robot.yaml's
# startup_sequence and named_configs, copied because the sim reads no controller config
# (tests/docker/test_sim_startup_poses.py keeps them equal).
STARTUP_ARM_STAGES = (
    ("init_1", (0.0, 0.35, 0.0, 1.3, 0.0, 0.0, 0.0, 0.0, -0.35, 0.0, 1.3, 0.0, 0.0, 0.0)),
    ("init_2", (0.7, 0.35, 0.0, 1.3, 0.0, 0.0, 0.0, 0.7, -0.35, 0.0, 1.3, 0.0, 0.0, 0.0)),
    ("init_3", (0.7, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, -0.35, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("home", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
)

def spawn(env, resolver, backoff=0.0, margin=None, offset=(0.0, 0.0)):
    """Put the robot at rest in its spawn pose, clear of the fixtures by margin, in the sagged poses too.

    Call after env.reset(), whose zero-action settling folds the limp robot.
    The robot is placed at RoboCasa's spawn shifted by offset ((dx, dy) m,
    world), its lowest geom 2 cm above the floor, and backed off (its -x) until
    every robot geom is margin (m; None: the env's spawn_margin, else 0.20)
    from any non-floor geom, in the spawn pose, the poses the unpowered upper
    body sags into and along the controller's start-up arm moves
    (STARTUP_ARM_STAGES); where backing straight off does not reach it, the
    search shifts sideways too (robosuite_common.place_robot_collision_free).
    backoff (m) then backs it up to that much further into open floor. Zeroes
    every velocity in the scene.
    """
    data = env.sim.data._data
    init_qpos = _initial_motor_qpos()
    data.qpos[resolver.motor_qpos] = init_qpos
    data.qvel[:] = 0.0
    pos = list(env.init_robot_base_pos)
    pos[0] += offset[0]
    pos[1] += offset[1]
    h1_2_robosuite.place_robot_collision_free(
        env, pos, robosuite_common.euler_to_wxyz(getattr(env, "init_robot_base_ori", None)),
        extra_backoff=backoff,
        margin=getattr(env, "spawn_margin", 0.20) if margin is None else margin,
        poses=(_limp_upper_body_poses(env.sim.model._model, resolver.motor_qpos, init_qpos)
               + robosuite_common.startup_arm_poses(resolver.motor_qpos, init_qpos, STARTUP_ARM_STAGES,
                                                  _ARMS[0])),
    )


# The elastic band on the torso holds the robot until SPACE or /elastic_band/toggle releases
# it. Its anchor is 4 cm below the torso's spawn: the feet carry part of the weight.
# Stiffness 1e4 N/m holds the limp robot near upright through the ~20 s the leg policy waits
# under the band for the arms' start-up (static tilt ~ gravity torque / stiffness; at 1e3 it
# pitched forward ~60 deg). Damping 2e3 N*s/m settles a sideways shove well within a second.
BAND_BODY = "torso_link"
BAND_ANCHOR_DZ = -0.04   # m
BAND_STIFFNESS = 1e4     # N/m
BAND_DAMPING = 2e3       # N*s/m


def make_band(data, resolver):
    """Return (band, torso body id): the band on the torso, anchored BAND_ANCHOR_DZ from it now.

    Call it at the spawn. Raises KeyError when the model has no torso_link.
    """
    body = resolver.body_id(BAND_BODY)
    if body < 0:
        raise KeyError(f"no body {BAND_BODY} for the elastic band")
    anchor = data.xpos[body].copy()
    anchor[2] += BAND_ANCHOR_DZ
    return ElasticBand(anchor, length=0, stiffness=BAND_STIFFNESS, damping=BAND_DAMPING), body


class H12Robot:
    """The H1-2 with its Magpie grippers, as golem_sim.SimRobot."""

    name = "h12"
    label = "the H1-2"
    own_tasks = battery_env.TASKS + blank_env.TASKS
    no_kitchen = blank_env.TASKS
    gil_switch_s = None

    def __init__(self):
        """Start with no extras running."""
        self._hands = ()
        self._estop = None

    def make_env(self, task, layout, style, seed):
        """Return the H1-2's robosuite env for task, not yet reset."""
        return h1_2_robosuite.make_kitchen_env(task, layout_ids=layout, style_ids=style, seed=seed)

    def make_resolver(self, env):
        """Return the H1-2's sim_names.NameResolver."""
        return NameResolver(env.sim.model._model)

    def spawn(self, env, resolver):
        """Spawn the H1-2 (spawn above) with the spawn knobs.

        GOLEM_SPAWN_BACKOFF (m, default 0: at the counter) backs the robot further into open
        floor; Nav2 needs a free start cell to plan (~1.0). GOLEM_SPAWN_MARGIN (m, unset: the
        env's spawn_margin, else 0.20) is the clearance to the fixtures.
        GOLEM_SPAWN_OFFSET="dx,dy" (m, world) shifts the task's spawn before placement; unset
        keeps the task's own.
        """
        margin = os.environ.get("GOLEM_SPAWN_MARGIN", "").strip()
        offset = os.environ.get("GOLEM_SPAWN_OFFSET", "").strip()
        dx, dy = (float(v) for v in offset.split(",")) if offset else (0.0, 0.0)
        if offset:
            print(f"[h12_robot] spawn offset ({dx:+.3f}, {dy:+.3f}) m")
        spawn(env, resolver, backoff=float(os.environ.get("GOLEM_SPAWN_BACKOFF", "0") or 0),
              margin=float(margin) if margin else None, offset=(dx, dy))
        print("[h12_robot] initial stance from baked-in sim defaults")

    def make_band(self, data, resolver):
        """Return (band, torso body id): the band on the torso, anchored BAND_ANCHOR_DZ from it now."""
        return make_band(data, resolver)

    def odom_body(self, env, resolver):
        """Return the body of the pelvis free joint."""
        model = env.sim.model._model
        pfx = env.robots[0].robot_model.naming_prefix
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{pfx}{robosuite_common.FREE_JOINT_NAME}")
        return int(model.jnt_bodyid[joint])

    def make_interface(self, model, data, resolver):
        """Return the H1-2's Unitree DDS interface; the gripper actuators are the hand bridges'."""
        return SimInterface(model, data, resolver=resolver)

    def make_lock(self, viewer):
        """Return one lock for both: every MjData access off the loop's thread (sensors, grippers, viewer) holds it."""
        lock = threading.Lock()
        return lock, lock

    def make_sensors(self, node, env, model, data, lock):
        """Return the H1-2's sensor suite: the three RealSense cameras and the MID-360.

        The cameras publish what the real RealSense drivers do (realsense_profile.yaml).
        Rendering is the heaviest per-step CPU cost: GOLEM_CAMERAS=0 drops every camera.
        GOLEM_CAMERA_SCALE renders the head camera at that fraction of its real 1280x720 (same
        projection, scaled camera_info). The hand cameras keep their gripper-MJCF fovy at
        GOLEM_HAND_CAMERA_RES=<W>x<H> (default 256x256; the D405's colour stream is 640x480).
        GOLEM_CAMERA_DEPTH=0 drops depth/image_rect_raw and GOLEM_CAMERA_POINTCLOUD=0 drops
        depth/color/points; with both off the depth render is skipped.
        """
        pfx = env.robots[0].robot_model.naming_prefix
        cameras_on = sim_common.env_flag('GOLEM_CAMERAS', True)
        camera_scale = float(os.environ.get('GOLEM_CAMERA_SCALE', '').strip() or 0.5)
        profile = realsense_profile.load_profile()
        camera_names = [
            ("head",       f"{pfx}{h1_2_robosuite.HEAD_CAMERA}"),
            ("left_hand",  "gripper0_left_hand_cam"),
            ("right_hand", "gripper0_right_hand_cam"),
        ] if cameras_on else []
        hand_res = tuple(int(v) for v in
                         (os.environ.get('GOLEM_HAND_CAMERA_RES', '').strip() or '256x256').lower().split('x'))
        cameras = realsense_profile.camera_specs(model, camera_names, profile, camera_scale, hand_res)
        # GOLEM_LIDAR_POINTS: points per 10 Hz frame, default the real sensor's 20,000.
        lidar_points = mid360.points_from_env()
        print(f"[h12_robot] MID-360 lidar: {lidar_points} points a frame")
        depth_on = sim_common.env_flag('GOLEM_CAMERA_DEPTH', True)
        cloud_on = sim_common.env_flag('GOLEM_CAMERA_POINTCLOUD', True)
        print(f"[h12_robot] RGBD cameras {'ON' if cameras_on else 'OFF (GOLEM_CAMERAS=0)'}"
              + (f": head {cameras[0].color.width}x{cameras[0].color.height} "
                 f"(GOLEM_CAMERA_SCALE={camera_scale:g}), hands {hand_res[0]}x{hand_res[1]}, "
                 f"depth stream {'on' if depth_on else 'off'}, "
                 f"point cloud {'on' if cloud_on else 'off'}" if cameras else ""))
        return H12Sensors(
            node, model, data,
            cameras=cameras,
            camera_tf=realsense_profile.tf_edges(profile, [ns for ns, _ in camera_names]),
            camera_tf_rate_hz=float(profile["tf_rate_hz"]),
            depth_stream=depth_on,
            pointcloud=cloud_on,
            # The real MID-360's IMU rate. FAST-LIO propagates the pose with the IMU between
            # scans, and at 100 Hz the gait's accelerations alias into drift.
            imu_rate_hz=200.0,
            lidar_body=f"{pfx}livox_link",
            lidar_exclude_body=f"{pfx}torso_link",
            imu_quat_sensor=f"{pfx}livox_imu_quat",
            imu_gyro_sensor=f"{pfx}livox_imu_gyro",
            imu_acc_sensor=f"{pfx}livox_imu_acc",
            # Drop the robot's own limbs from the scan: their returns map an obstacle under
            # the robot and Nav2 refuses the start. GOLEM_LIDAR_SELF_FILTER=0 keeps them.
            lidar_self_prefixes=(
                (pfx, "gripper0_") if sim_common.env_flag("GOLEM_LIDAR_SELF_FILTER", True) else ()
            ),
            lidar_points=lidar_points,
            sim_lock=lock,
        )

    def start_extras(self, env, model, data, lock):
        """Start the Magpie gripper bridges and the estop box; return (the gripper nodes, the estop's key callback).

        SAFETY: the estop box is the one golem_bringup's estop_node reads (estop_server);
        golem_safety latches on its ENTER toggle, and on its silence once the sim stops.
        """
        self._hands = (
            MagpieHandBridge(model, data, side="right", sim_lock=lock, name_prefix="gripper0_right_"),
            MagpieHandBridge(model, data, side="left", sim_lock=lock, name_prefix="gripper0_left_"),
        )
        self._estop = start_estop_server("golem_sim")
        return list(self._hands), (self._estop.key_callback,)

    def tick_extras(self, sim_t):
        """Publish the grippers' state at its rate of sim time."""
        for hand in self._hands:
            hand.tick(sim_t)

    def stop_extras(self):
        """End the grippers' DeliGrasp settles, which wait on the sim clock, and close the estop box."""
        for hand in self._hands:
            hand.stop()
        if self._estop is not None:
            self._estop.shutdown()

    def make_fall_logger(self, resolver, band_body):
        """Return the logger of the band's release and the first time the torso tips or drops."""
        return FallLogger(band_body, "torso", upright_min=0.5, height_min=0.5)

    def describe(self, model, resolver):
        """Return how many motors the resolver mapped."""
        return f"resolver mapped {len(resolver.motor_ctrl)} motors"

    def surface(self):
        """Return the H1-2 sim's topics."""
        return ("rt/lowstate+rt/lowcmd, /clock, RGBD /realsense/{head,left_hand,right_hand}, "
                "livox lidar+imu, /{left,right}/gripper/*")
