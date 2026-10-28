"""Tests for g1_lidar (MID-360 scans and IMU) and g1_head_camera (D435), no ROS.

The G1 is g1_robosuite's robot MJCF (names unprefixed), in small scenes of
known geometry. Expected values come from the URDF and the scenes' analytic
geometry, not from MuJoCo's kinematics. Run in the robocasa image (robosuite
for g1_robosuite, EGL for the camera) with golem_robocasa on PYTHONPATH, as
tests/robocasa/g1/run_in_container.sh does.
"""
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from g1 import g1_head_camera as cam
from g1 import g1_lidar as lidar
from g1 import g1_robosuite

PELVIS_Z = 0.793    # m, the MJCF's pelvis height with every joint at 0
# the head camera sees the robot's own forearms and hands (every joint at 0) at
# 350-541 mm, on 12% of the image at the D435 color FOV
ROBOT_IN_VIEW_MM = 600
# the room, m: the floor at z = 0 and four 3 m walls' inner faces
FRONT, BACK, LEFT, RIGHT = 2.5, -2.5, 2.0, -2.0
ROOM = f"""
<geom name="floor" type="plane" size="0 0 0.1" group="1" rgba="0.5 0.5 0.5 1"/>
<geom name="front" type="box" pos="{FRONT + 0.05} 0 1.5" size="0.05 2.2 1.5" group="1"/>
<geom name="back" type="box" pos="{BACK - 0.05} 0 1.5" size="0.05 2.2 1.5" group="1"/>
<geom name="left" type="box" pos="0 {LEFT + 0.05} 1.5" size="2.7 0.05 1.5" group="1"/>
<geom name="right" type="box" pos="0 {RIGHT - 0.05} 1.5" size="2.7 0.05 1.5" group="1"/>
"""
LIGHT = '<light pos="0 0 4" dir="0 0 -1" directional="true"/>'


def _model(world_xml):
    root = g1_robosuite.build_robot_xml()
    world = root.find("worldbody")
    for elem in ET.fromstring(f"<w>{LIGHT}{world_xml}</w>"):
        world.append(elem)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _place(model, data, xyz=(0.0, 0.0, PELVIS_Z), yaw=0.0, pitch=0.0):
    """The pelvis at xyz, rotated by yaw then pitch; every joint at 0, at rest"""
    data.qpos[:] = 0.0
    data.qpos[:3] = xyz
    w, x, y, z = _wxyz(Rotation.from_euler("ZY", [yaw, pitch]))
    data.qpos[3:7] = (w, x, y, z)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)   # mj_kinematics alone leaves the cameras where they were


def _wxyz(rotation):
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


def _urdf_pose(tip, root="pelvis"):
    """T_root_tip (4x4) through the URDF, every joint at 0"""
    joints = {j.find("child").get("link"): j
              for j in ET.parse(g1_robosuite.SOURCE_URDF).getroot().findall("joint")}
    t = np.eye(4)
    link = tip
    while link != root:
        joint = joints[link]
        origin = joint.find("origin")
        step = np.eye(4)
        step[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
        step[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        t = step @ t
        link = joint.find("parent").get("link")
    return t


def _pelvis(xyz=(0.0, 0.0, PELVIS_Z), yaw=0.0, pitch=0.0):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("ZY", [yaw, pitch]).as_matrix()
    t[:3, 3] = xyz
    return t


def _room_range(origin, directions):
    """Distance along each unit direction from origin to the room's first surface"""
    with np.errstate(divide="ignore", invalid="ignore"):
        hits = [np.where(directions[:, 2] < 0, -origin[2] / directions[:, 2], np.inf)]
        for axis, low, high in ((0, BACK, FRONT), (1, RIGHT, LEFT)):
            d = directions[:, axis]
            hits.append(np.where(d > 0, (high - origin[axis]) / d,
                                 np.where(d < 0, (low - origin[axis]) / d, np.inf)))
    return np.min(hits, axis=0)


def _on_room(world):
    """Distance (m) from each world point to the nearest room surface"""
    x, y, z = world.T
    return np.min(np.abs([z, x - FRONT, x - BACK, y - LEFT, y - RIGHT]), axis=0)


def _frame_at_rest(sensor, data):
    """One frame from a robot that does not move: update() at the start and the end"""
    sensor.reset()
    assert sensor.update(data, 0.0) is None
    frame = sensor.update(data, sensor.period)
    assert frame is not None
    return frame


ROOM_MODEL = _model(ROOM)
T_PELVIS_LIDAR = _urdf_pose("mid360_link")


# ------------------------------------------------------------------ MID-360
def test_lidar_site_is_the_urdf_mount():
    data = mujoco.MjData(ROOM_MODEL)
    _place(ROOM_MODEL, data)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="")
    position, rotation = sensor.pose(data)
    expected = _pelvis() @ T_PELVIS_LIDAR
    assert np.allclose(position, expected[:3, 3], atol=1e-6), (position, expected[:3, 3])
    assert np.allclose(rotation, expected[:3, :3], atol=1e-6)
    assert rotation[2, 2] < -0.99   # upside down: the sensor's z points at the floor


def test_wall_at_known_range_and_direction():
    """Straight ahead in the sensor frame is the robot's front; its +y is the robot's right"""
    data = mujoco.MjData(ROOM_MODEL)
    _place(ROOM_MODEL, data)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="")
    position, rotation = (_pelvis() @ T_PELVIS_LIDAR)[:3, 3], (_pelvis() @ T_PELVIS_LIDAR)[:3, :3]
    rays = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    dist = sensor.cast(data, rays)
    pitch = 0.05112069379091391   # mid360_joint rpy: x leans down by it, so ahead hits the front wall
    expected = [(FRONT - position[0]) / np.cos(pitch), position[1] - RIGHT, LEFT - position[1],
                (position[0] - BACK) / np.cos(pitch)]
    assert np.allclose(dist, expected, atol=1e-6), (dist, expected)
    world = position + (rays * dist[:, None]) @ rotation.T
    assert abs(world[0, 0] - FRONT) < 1e-6 and abs(world[1, 1] - RIGHT) < 1e-6
    assert abs(world[2, 1] - LEFT) < 1e-6 and abs(world[3, 0] - BACK) < 1e-6


def test_frame_matches_the_room_and_the_mount():
    """Every ray of a frame: the range the URDF mount predicts, or blocked by the robot"""
    data = mujoco.MjData(ROOM_MODEL)
    _place(ROOM_MODEL, data)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="")
    frame = _frame_at_rest(sensor, data)
    n = sensor.points_per_frame
    assert 0.9 * n < len(frame.points) <= n, len(frame.points)
    pose = _pelvis() @ T_PELVIS_LIDAR
    rng = np.linalg.norm(frame.points, axis=1)
    directions = frame.points / rng[:, None]
    world_dirs = directions @ pose[:3, :3].T
    expected = _room_range(pose[:3, 3], world_dirs)
    assert np.abs(rng - expected).max() < 1e-4, np.abs(rng - expected).max()
    world = pose[:3, 3] + frame.points @ pose[:3, :3].T
    assert _on_room(world).max() < 1e-4
    # elevation within -7..52 deg of the sensor frame
    elevation = np.degrees(np.arcsin(directions[:, 2]))
    assert elevation.min() > -7.0 - 1e-4 and elevation.max() < 52.0 + 1e-4
    # upside down: the floor is at the sensor's +z, the left wall at its -y
    floor = np.abs(world[:, 2]) < 1e-4
    left = np.abs(world[:, 1] - LEFT) < 1e-4
    assert floor.sum() > 0.3 * n and left.sum() > 0.05 * n
    assert (frame.points[floor, 2] > 0).all() and (frame.points[left, 1] < 0).all()
    # floor points: each where the mount predicts, at range h / -dz
    assert np.allclose(rng[floor], -pose[2, 3] / world_dirs[floor, 2], atol=1e-4)
    assert set(np.unique(frame.line)) == {0, 1, 2, 3}


def _mj_ray(model, data, position, world_dirs, bodyexclude):
    """(distance, geom id) of each ray by mj_ray, one at a time (exact, slow)"""
    group = np.array(lidar.GEOM_GROUPS, np.uint8).reshape(6, 1)
    geomid = np.zeros((1, 1), np.int32)
    dist, ids = np.empty(len(world_dirs)), np.empty(len(world_dirs), int)
    for i, v in enumerate(world_dirs):
        dist[i] = mujoco.mj_ray(model, data, position.reshape(3, 1), v.reshape(3, 1).copy(), group,
                                1, bodyexclude, geomid)
        ids[i] = geomid[0, 0]
    return dist, ids


def test_no_returns_from_the_robot():
    """The rays that hit the robot (not torso_link, which the sensor sits in) give no point;
    every other ray gives mj_ray's distance"""
    data = mujoco.MjData(ROOM_MODEL)
    _place(ROOM_MODEL, data)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="")
    directions, _ = lidar.scan_pattern(0)
    position, rotation = sensor.pose(data)
    n = len(directions)
    world = directions @ rotation.T
    torso = sensor.torso_id
    dist, geomid = _mj_ray(ROOM_MODEL, data, position, world, -1)
    assert (geomid >= 0).all() and (ROOM_MODEL.geom_bodyid[geomid] == torso).all()
    print(f"  the head shell is {dist.min():.3f} to {dist.max():.3f} m from the sensor")
    dist, geomid = _mj_ray(ROOM_MODEL, data, position, world, torso)
    assert (geomid >= 0).all()   # a closed room
    blocked = sensor.robot_geom[geomid]
    bodies = {mujoco.mj_id2name(ROOM_MODEL, mujoco.mjtObj.mjOBJ_BODY, b)
              for b in ROOM_MODEL.geom_bodyid[geomid[blocked]]}
    print(f"  {blocked.sum()} of {n} rays blocked by {sorted(bodies)}")
    assert 0 < blocked.sum() < 0.1 * n, blocked.sum()   # shoulders, wrists and hands are in the FOV
    measured = sensor.cast(data, directions)
    assert np.isnan(measured[blocked]).all()
    assert np.isfinite(measured[~blocked]).all()
    assert np.array_equal(measured[~blocked], dist[~blocked])
    world_points = position + (directions * measured[:, None])[~blocked] @ rotation.T
    assert _on_room(world_points).max() < 1e-4


def test_cast_is_exact_where_multiray_culls():
    """A ray just below a wall's corners but above its bottom edge: mj_multiRay's own
    culling sends it through the wall to the floor behind; cast() hits the wall"""
    data = mujoco.MjData(ROOM_MODEL)
    _place(ROOM_MODEL, data)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="")
    position, rotation = sensor.pose(data)
    # at the left wall 0.3 m above the floor, mid-length: steeper than any of its corners
    target = np.array([0.0, LEFT, 0.3])
    v = (target - position) / np.linalg.norm(target - position)
    geomid, dist = np.zeros((1, 1), np.int32), np.zeros((1, 1))
    group = np.array(lidar.GEOM_GROUPS, np.uint8).reshape(6, 1)
    mujoco.mj_multiRay(ROOM_MODEL, data, position.reshape(3, 1), v.reshape(3, 1).copy(), group, 1,
                       sensor.torso_id, geomid, dist, 1, lidar.MAX_RANGE)
    floor = mujoco.mj_name2id(ROOM_MODEL, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert geomid[0, 0] == floor   # MuJoCo 3.3.1's culling; if this fails, MuJoCo fixed it
    bvhadr = ROOM_MODEL.body_bvhadr.copy()
    measured = sensor.cast(data, (rotation.T @ v)[None])
    assert abs(measured[0] - np.linalg.norm(target - position)) < 1e-9
    assert np.array_equal(ROOM_MODEL.body_bvhadr, bvhadr) and (bvhadr != -1).any()   # restored


def test_offset_time_labels_the_pose_each_point_was_cast_from():
    """A robot that walks and turns: offset_time is monotonic and, with the true pose at
    timebase + offset_time, every point lands on the room"""
    data = mujoco.MjData(ROOM_MODEL)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="")
    dt, speed, turn = 0.002, 0.5, 2.0   # s, m/s, rad/s

    def pose(t):
        return _pelvis((speed * t, 0.0, PELVIS_Z), yaw=turn * t) @ T_PELVIS_LIDAR

    frames = []
    for k in range(126):   # 0 to 0.25 s
        t = k * dt
        _place(ROOM_MODEL, data, (speed * t, 0.0, PELVIS_Z), yaw=turn * t)
        frame = sensor.update(data, t)
        if frame is not None:
            frames.append(frame)
    assert len(frames) == 2
    assert abs(frames[0].timebase) < 1e-9 and abs(frames[1].timebase - 0.1) < 1e-9
    for frame in frames:
        offset = frame.offset_time.astype(np.int64)
        assert offset[0] == 0 and (np.diff(offset) >= 0).all()
        assert offset[-1] == 100_000_000   # the last rays are cast at the frame's end
        assert (offset % 2_000_000 == 0).all()   # one group per physics step
        times = frame.timebase + offset * 1e-9
        world = np.empty_like(frame.points, dtype=float)
        for t in np.unique(times):
            at = times == t
            p = pose(t)
            world[at] = p[:3, 3] + frame.points[at] @ p[:3, :3].T
        assert _on_room(world).max() < 1e-4, _on_room(world).max()
        # the labels matter: every point at the timebase pose misses the room by far
        p = pose(frame.timebase)
        assert _on_room(p[:3, 3] + frame.points @ p[:3, :3].T).max() > 0.05


def test_frames_are_back_to_back_at_the_rate():
    data = mujoco.MjData(ROOM_MODEL)
    _place(ROOM_MODEL, data)
    sensor = lidar.Mid360(ROOM_MODEL, prefix="", points_per_frame=2000)
    stamps = [f.timebase for k in range(1001) if (f := sensor.update(data, k * 0.003)) is not None]
    # a 3 ms loop over 3 s: frame i is due over [0.1 i, 0.1 (i + 1)) and ends on the
    # first step at or past its end, so the stamps keep 10 Hz without drifting
    assert len(stamps) == 30, len(stamps)
    assert all(0.097 < d < 0.103 for d in np.diff(stamps)), np.diff(stamps)
    assert abs(stamps[-1] - 2.901) < 1e-9
    # a sim reset (time back to 0) starts over
    assert sensor.update(data, 0.0) is None and sensor.update(data, 0.1).timebase == 0.0


def test_scan_pattern_is_non_repetitive_and_uniform():
    a, line = lidar.scan_pattern(0)
    b, _ = lidar.scan_pattern(1)
    assert a.shape == (lidar.POINTS_PER_FRAME, 3) and np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.abs(a - b).max() > 0.1 and (line == np.arange(len(a)) % 4).all()
    azimuth = np.unwrap(np.arctan2(a[:, 1], a[:, 0]))
    assert (np.diff(azimuth) > 0).all() and azimuth[-1] - azimuth[0] < 2 * np.pi
    # uniform in solid angle: sin(elevation) uniform over the band
    s = np.sort(a[:, 2])
    low, high = np.sin(np.radians(lidar.ELEVATION_DEG))
    assert np.abs(s - np.linspace(low, high, len(s))).max() < 0.01


# ---------------------------------------------------------------------- IMU
def test_imu_reads_gravity_upside_down():
    """Held at rest (no acceleration): +1 g up, which is about -z for the upside-down sensor"""
    model = ROOM_MODEL
    data = mujoco.MjData(model)
    _place(model, data)
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0
    mujoco.mj_rnePostConstraint(model, data)
    gyro, acc = lidar.Mid360(model, prefix="").imu(data)
    up = np.array([0.0, 0.0, -model.opt.gravity[2]]) / lidar.STANDARD_GRAVITY
    expected = (_pelvis() @ T_PELVIS_LIDAR)[:3, :3].T @ up
    pitch = 0.05112069379091391
    assert np.allclose(expected, np.array([-np.sin(pitch), 0.0, -np.cos(pitch)]) * up[2], atol=1e-9)
    assert np.allclose(acc, expected, atol=1e-9), (acc, expected)
    assert np.allclose(gyro, 0.0, atol=1e-12)
    # tilted: pitch the pelvis forward 20 deg
    _place(model, data, pitch=np.radians(20.0))
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0
    mujoco.mj_rnePostConstraint(model, data)
    _, acc = lidar.Mid360(model, prefix="").imu(data)
    expected = (_pelvis(pitch=np.radians(20.0)) @ T_PELVIS_LIDAR)[:3, :3].T @ up
    assert np.allclose(acc, expected, atol=1e-9)


def test_imu_in_free_fall_reads_zero():
    model = _model(ROOM)
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)
    _place(model, data)
    mujoco.mj_forward(model, data)
    _, acc = lidar.Mid360(model, prefix="").imu(data)
    assert np.abs(acc).max() < 1e-6, acc


def test_imu_lever_arm_matches_mujocos_accelerometer():
    """imu() at the torso IMU's point, rotated into its frame, is MuJoCo's own torso
    gyro and accelerometer for a robot moving every joint"""
    model = ROOM_MODEL
    data = mujoco.MjData(model)
    _place(model, data)
    rng = np.random.default_rng(1)
    data.qvel[:] = rng.uniform(-3.0, 3.0, model.nv)
    mujoco.mj_forward(model, data)
    sensor = lidar.Mid360(model, prefix="")
    lidar_pose = _urdf_pose("mid360_link", "torso_link")
    torso_imu = _urdf_pose("imu_in_torso", "torso_link")
    point = np.linalg.inv(lidar_pose)[:3, :] @ np.append(torso_imu[:3, 3], 1.0)
    gyro, acc = sensor.imu(data, point)
    to_torso_imu = torso_imu[:3, :3].T @ lidar_pose[:3, :3]

    def sensordata(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return data.sensordata[model.sensor_adr[sid]:][:3]

    expected_acc = sensordata("imu-torso-linear-acceleration")
    assert np.allclose(to_torso_imu @ gyro, sensordata("imu-torso-angular-velocity"), atol=1e-9)
    assert np.allclose(to_torso_imu @ acc * lidar.STANDARD_GRAVITY, expected_acc, atol=1e-9)
    # the 0.28 m lever arm from the sensor's site to that point matters here
    _, at_site = sensor.imu(data, np.zeros(3))
    lever = np.linalg.norm(acc - at_site) * lidar.STANDARD_GRAVITY
    assert lever > 1.0, lever
    print(f"  |w| {np.linalg.norm(gyro):.2f} rad/s; the lever arm adds {lever:.2f} m/s^2")


def test_imu_point_is_fastlios_extrinsic():
    """FAST_LIO's mid360 extrinsic_T is the lidar in the IMU frame: the IMU is at -T"""
    import os
    import yaml
    config = os.path.join(g1_robosuite._REPO, "core_ws", "src", "g1", "g1_navigation", "config",
                          "g1_fastlio_sim.yaml")
    if not os.path.isdir(os.path.join(g1_robosuite._REPO, "core_ws", "src")):   # core_ws is not mounted
        print(f"  (skipped: no core_ws/src under {g1_robosuite._REPO})")
        return
    with open(config) as f:   # core_ws is there: a missing config is a failure, not a skip
        mapping = yaml.safe_load(f)["/**"]["ros__parameters"]["mapping"]
    assert np.allclose(mapping["extrinsic_R"], np.eye(3).ravel())
    assert np.allclose(lidar.IMU_IN_LIDAR, -np.array(mapping["extrinsic_T"]))


# ------------------------------------------------------------------- camera
def _camera_world_pose(pelvis=None):
    """T_world_head_color_optical_frame from the URDF and OPTICAL_IN_D435"""
    optical = np.eye(4)
    optical[:3, :3] = Rotation.from_quat(cam.OPTICAL_IN_D435_XYZW).as_matrix()
    return (_pelvis() if pelvis is None else pelvis) @ _urdf_pose("d435_link") @ optical


def test_camera_is_at_d435_link_with_the_optical_constant():
    model = ROOM_MODEL
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "d435_link")
    assert np.allclose(model.cam_pos[cid], model.site_pos[sid])
    r_d435 = Rotation.from_quat(np.roll(model.site_quat[sid], -1)).as_matrix()
    r_cam = Rotation.from_quat(np.roll(model.cam_quat[cid], -1)).as_matrix()
    # a MuJoCo camera looks along its -z with +y up; the optical frame is z ahead, y down
    optical = r_d435.T @ r_cam @ np.diag([1.0, -1.0, -1.0])
    assert np.allclose(optical, Rotation.from_quat(cam.OPTICAL_IN_D435_XYZW).as_matrix(), atol=1e-9)


def test_intrinsics_match_fovy():
    camera = cam.HeadCamera(ROOM_MODEL, prefix="")
    assert camera.width * 9 == camera.height * 16
    cid = mujoco.mj_name2id(ROOM_MODEL, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    assert camera.fovy == ROOM_MODEL.cam_fovy[cid] == g1_robosuite.HEAD_CAM_FOVY
    assert np.isclose(2 * np.degrees(np.arctan(camera.height / 2 / camera.fy)), camera.fovy)
    assert camera.fx == camera.fy
    assert (camera.cx, camera.cy) == ((camera.width - 1) / 2, (camera.height - 1) / 2)


def test_known_objects_land_at_the_predicted_pixels():
    """Spheres placed in the URDF's optical frame show at K's projection, in rgb order"""
    # optical frame (x right, y down, z ahead), clear of the forearms, which point
    # ahead with every joint at 0 and fill the lower middle of the image
    spheres = {(1, 0, 0): (0.0, 0.0, 1.2), (0, 1, 0): (0.9, -0.35, 1.45),
               (0, 0, 1): (-0.45, -0.3, 1.6), (1, 1, 1): (-0.75, 0.3, 1.3)}
    radius = 0.04
    world_cam = _camera_world_pose()
    xml = ""
    for color, p in spheres.items():
        w = world_cam[:3, :3] @ p + world_cam[:3, 3]
        xml += (f'<geom type="sphere" size="{radius}" pos="{w[0]} {w[1]} {w[2]}" group="1" '
                f'rgba="{color[0]} {color[1]} {color[2]} 1"/>')
    model = _model(xml)   # no floor: behind the spheres there is nothing (depth 0)
    data = mujoco.MjData(model)
    _place(model, data)
    camera = cam.HeadCamera(model, prefix="")
    rgb, depth = camera.render(data)
    camera.close()
    assert rgb.shape == (camera.height, camera.width, 3) and rgb.dtype == np.uint8
    v, u = np.mgrid[:camera.height, :camera.width]
    for color, (x, y, z) in spheres.items():
        # each sphere's pixels by depth (the lighting shades the colour unevenly)
        mask = np.abs(depth.astype(float) - 1000 * z) < 1000 * radius + 5
        assert mask.sum() > 300, (color, mask.sum())
        # an off-axis sphere's outline is an ellipse centred a little outward of its
        # centre's projection: the middle of the rays at theta -+ alpha from the axis
        theta, alpha = np.arccos(z / np.linalg.norm((x, y, z))), np.arcsin(radius / np.linalg.norm((x, y, z)))
        radial = (np.tan(theta + alpha) + np.tan(theta - alpha)) / 2
        direction = np.array([x, y]) / (np.hypot(x, y) or 1.0)
        expected = (camera.fx * radial * direction[0] + camera.cx,
                    camera.fy * radial * direction[1] + camera.cy)
        found = (u[mask].mean(), v[mask].mean())
        error = np.hypot(found[0] - expected[0], found[1] - expected[1])
        assert error < 0.1, (color, found, expected)
        print(f"  sphere at ({x}, {y}, {z}) m: {mask.sum()} px, centroid {error:.3f} px "
              f"from K's projection of its outline ({camera.fx * x / z + camera.cx:.2f}, "
              f"{camera.fy * y / z + camera.cy:.2f} for its centre)")
        # rgb8: a red, green or blue sphere's own channel is the brightest
        mean = rgb[mask].mean(axis=0)
        if sum(color) == 1:
            assert np.argmax(mean) == np.argmax(color) and mean.max() > 2 * np.sort(mean)[1], mean
        # depth is z in the optical frame: the sphere's least z is its centre's minus
        # the radius (inside its outline: a pixel on the outline can read a few mm off)
        inside = mask.copy()
        inside[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        assert abs(depth[inside].min() - 1000 * (z - radius)) < 2, (color, depth[inside].min())


def test_depth_is_z_in_mm():
    """A wall square to the optical axis at Z reads Z in mm at every pixel (not the range)"""
    world_cam = _camera_world_pose()
    quat = _wxyz(Rotation.from_matrix(world_cam[:3, :3]))
    for distance in (0.35, 1.0):
        centre = world_cam[:3, :3] @ (0.0, 0.0, distance + 0.005) + world_cam[:3, 3]
        model = _model(f'<geom type="plane" size="0 0 0.1" group="1" rgba="0.5 0.5 0.5 1"/>'
                       f'<geom type="box" size="5 5 0.005" pos="{centre[0]} {centre[1]} {centre[2]}" '
                       f'quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" group="1"/>')
        data = mujoco.MjData(model)
        _place(model, data)
        camera = cam.HeadCamera(model, prefix="")
        _, depth = camera.render(data)
        camera.close()
        assert depth.dtype == np.uint16 and depth.shape == (camera.height, camera.width)
        expected = round(distance * 1000)
        exact = np.abs(depth.astype(int) - expected) <= 1
        robot = depth.astype(int) < min(expected - 1, ROBOT_IN_VIEW_MM)
        # every pixel reads the wall's Z but those on the robot's own arms and hands
        assert np.all(exact | robot) and robot.mean() < 0.2, (distance, exact.mean(), robot.mean())
        assert depth.min() > 0 and depth.max() <= expected + 1   # only the robot is nearer
        print(f"  wall at {distance} m: {exact.mean():.1%} of pixels within 1 mm, "
              f"the rest the robot's own body")


def test_no_surface_reads_zero():
    model = _model("")
    data = mujoco.MjData(model)
    _place(model, data, pitch=np.radians(-100.0))   # on its back, the camera at the sky
    camera = cam.HeadCamera(model, prefix="")
    _, depth = camera.render(data)
    camera.close()
    # no depth but on the robot's own arms and hands
    assert (depth == 0).mean() > 0.8 and depth.max() < ROBOT_IN_VIEW_MM, ((depth == 0).mean(), depth.max())


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
