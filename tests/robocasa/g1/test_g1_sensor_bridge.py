"""Tests for g1.g1_sensors without DDS: its env knobs (SensorConfig), the
MID-360's scan thread (_LidarScans), whose frames go to a list instead of a
publisher, and the head camera's render thread (_HeadCamera), whose messages go
to a stand-in node's lists. No ROS node is created. The camera needs GL
(MUJOCO_GL=egl with the GPU, or Mesa's).

The G1 is g1_robosuite's robot MJCF (names unprefixed) in a room with two
screens: one coloured by its geom's rgba, one by its material's, so that both
alphas MuJoCo's rays read can be changed at runtime, as RoboCasa tasks do. Run
in the robocasa image with ROS 2 Humble and the Livox messages sourced and
h1_robocasa on PYTHONPATH, as tests/robocasa/g1/run_in_container.sh does.
"""
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from livox_ros_driver2.msg import CustomMsg
from rclpy.serialization import deserialize_message

from g1 import g1_head_camera
from g1 import g1_lidar as lidar
from g1 import g1_robosuite
from g1 import g1_sensor_msgs as sm
from g1 import g1_sensors as bridge

PELVIS_Z = 0.793    # m, the MJCF's pelvis height with every joint at 0
SCREEN_X = 1.0      # m: the screens' faces towards the robot, ahead (geom alpha) and behind (material alpha)
ROOM = f"""
<light pos="0 0 4" dir="0 0 -1" directional="true"/>
<geom name="floor" type="plane" size="0 0 0.1" group="1" rgba="0.5 0.5 0.5 1"/>
<geom name="front" type="box" pos="2.55 0 1.5" size="0.05 2.2 1.5" group="1"/>
<geom name="back" type="box" pos="-2.55 0 1.5" size="0.05 2.2 1.5" group="1"/>
<geom name="left" type="box" pos="0 2.05 1.5" size="2.7 0.05 1.5" group="1"/>
<geom name="right" type="box" pos="0 -2.05 1.5" size="2.7 0.05 1.5" group="1"/>
<geom name="screen_geom" type="box" pos="{SCREEN_X + 0.01} 0 0.8" size="0.01 0.6 0.8" group="1"
      rgba="0.2 0.6 0.2 1"/>
<geom name="screen_mat" type="box" pos="{-SCREEN_X - 0.01} 0 0.8" size="0.01 0.6 0.8" group="1"
      material="screen"/>
"""


def _model():
    root = g1_robosuite.build_robot_xml()
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", name="screen", rgba="0.6 0.2 0.2 1")
    world = root.find("worldbody")
    for elem in ET.fromstring(f"<w>{ROOM}</w>"):
        world.append(elem)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _standing(model):
    data = mujoco.MjData(model)
    data.qpos[2] = PELVIS_Z
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    return data


def _submit(scans, data, t):
    """scans.submit(), once the ring has a free slot: the loops here outrun real
    time, and a skipped cast would change the frames"""
    while scans._free.empty():
        time.sleep(0.0005)
    scans.submit(data, t)


def _wait_for(frames, count, timeout=30.0):
    end = time.time() + timeout
    while len(frames) < count and time.time() < end:
        time.sleep(0.01)


# ------------------------------------------------------------------ SensorConfig
def test_sensor_config_defaults():
    """Unset or empty knobs (compose passes unset ones as empty) are the defaults"""
    default = bridge.SensorConfig()
    assert bridge.SensorConfig.from_env({}) == default
    empty = {k: "" for k in ("GOLEM_LIDAR", "GOLEM_LIDAR_POINTS", "GOLEM_LIDAR_HZ", "GOLEM_CAMERAS",
                             "GOLEM_CAMERA_HZ", "GOLEM_CAMERA_SCALE")}
    assert bridge.SensorConfig.from_env(empty) == default
    assert (default.lidar, default.lidar_points, default.lidar_hz, default.imu_hz) == (True, 5000, 10.0, 200.0)
    assert (default.camera, default.camera_hz, default.camera_width, default.camera_height) == \
        (True, 6.0, 1280, 720)


def test_sensor_config_knobs():
    config = bridge.SensorConfig.from_env({"GOLEM_LIDAR": "0", "GOLEM_CAMERAS": "off", "GOLEM_LIDAR_POINTS": "2000",
                                           "GOLEM_LIDAR_HZ": "100", "GOLEM_CAMERA_HZ": "30",
                                           "GOLEM_CAMERA_SCALE": "0.5"})
    assert (config.lidar, config.camera, config.lidar_points, config.lidar_hz) == (False, False, 2000, 100.0)
    assert (config.camera_hz, config.camera_width, config.camera_height) == (30.0, 640, 360)


def test_sensor_config_rejects_bad_knobs():
    """A bad knob fails (before the kitchen is built), NaN and infinity included:
    a NaN rate would make the schedule due on every physics step"""
    bad = [("GOLEM_LIDAR_HZ", v) for v in ("nan", "inf", "-inf", "0", "-1", "101", "fast")]
    bad += [("GOLEM_CAMERA_HZ", v) for v in ("nan", "NaN", "inf", "0", "-6", "30.5", "60")]
    bad += [("GOLEM_LIDAR_POINTS", v) for v in ("nan", "inf", "0", "-5", "2.5", "1e4")]
    bad += [("GOLEM_CAMERA_SCALE", v) for v in ("nan", "inf", "0", "-0.5", "half", "640x360", "1e-4")]
    for name, value in bad:
        try:
            bridge.SensorConfig.from_env({name: value})
        except ValueError as e:
            assert name in str(e), (name, value, e)
        else:
            raise AssertionError(f"{name}={value!r} was accepted")


# ------------------------------------------------------------------ the lidar thread
def test_lidar_thread_matches_live_casts():
    """The thread (pose snapshots, its own MjData on its own model copy) gives
    byte-identical frames to a Mid360 cast on the live model and MjData at the
    same times, while the physics steps beside it: the robot flails under random
    torques and falls, and both screens' alphas change at runtime"""
    model = _model()
    data = _standing(model)
    config = bridge.SensorConfig()
    got = []
    scans = bridge._LidarScans(model, data, "", config, got.append)
    try:
        # its own copy of the model: cast() edits body_bvhadr, the alpha is set per cast
        for field in ("body_bvhadr", "geom_rgba", "mat_rgba"):
            assert not np.shares_memory(getattr(scans.model, field), getattr(model, field)), field
        live = lidar.Mid360(model, "", points_per_frame=config.lidar_points, rate_hz=config.lidar_hz)
        want = []
        screen = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screen_geom")
        rng = np.random.default_rng(3)
        z = []
        for step in range(2500):   # 5 s of sim time
            if step in (800, 1600, 2000):   # hide the screens, then show them again
                model.geom_rgba[screen, 3] = 0.0 if step == 800 else 1.0
                model.mat_rgba[0, 3] = 0.0 if step == 1600 else 1.0
            data.ctrl[:] = rng.uniform(-8.0, 8.0, model.nu)
            mujoco.mj_step(model, data)
            t = float(data.time) - model.opt.timestep
            if step % bridge.LIDAR_CAST_EVERY == 0:
                _submit(scans, data, t)
                frame = live.update(data, t)
                if frame is not None:
                    want.append(sm.livox_custom_msg_cdr(frame))
            z.append(float(data.qpos[2]))
        _wait_for(got, len(want))
    finally:
        scans.close()
    assert max(z) - min(z) > 0.3, (max(z), min(z))   # the robot fell
    assert scans.skipped == 0, scans.skipped
    assert len(want) == 49 and len(got) == len(want), (len(got), len(want))
    same = sum(bytes(a) == bytes(b) for a, b in zip(got, want))
    assert same == len(want), f"{same}/{len(want)} frames byte-identical"


def test_lidar_thread_sees_runtime_alpha_edits():
    """Each cast uses the alpha in effect when its poses were taken: no point lies on a
    screen while it is hidden (alpha 0, by its geom or by its material), some do
    while it is shown"""
    model = _model()
    data = _standing(model)
    screen = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screen_geom")
    got = []
    scans = bridge._LidarScans(model, data, "", bridge.SensorConfig(), got.append)
    period = bridge.LIDAR_CAST_EVERY * model.opt.timestep   # 10 ms
    geom_hidden, mat_hidden = range(20, 40), range(30, 50)   # in casts
    try:
        for k in range(61):   # 0.6 s: six frames
            model.geom_rgba[screen, 3] = 0.0 if k in geom_hidden else 1.0
            model.mat_rgba[0, 3] = 0.0 if k in mat_hidden else 1.0
            _submit(scans, data, k * period)   # the robot does not move
        _wait_for(got, 6)
    finally:
        scans.close()
    assert len(got) == 6, len(got)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, lidar.SITE)
    position, rotation = data.site_xpos[site], data.site_xmat[site].reshape(3, 3)
    world, cast = [], []
    for raw in got:
        msg = deserialize_message(bytes(raw), CustomMsg)
        p = np.array([[q.x, q.y, q.z] for q in msg.points], np.float64).reshape(-1, 3)
        world.append(position + p @ rotation.T)
        when = msg.timebase * 1e-9 + np.array([q.offset_time for q in msg.points]) * 1e-9
        cast.append(np.rint(when / period).astype(int))
    world, cast = np.concatenate(world), np.concatenate(cast)
    on_screen = (np.abs(np.abs(world[:, 0]) - SCREEN_X) < 1e-4) & (np.abs(world[:, 1]) <= 0.6) & \
        (world[:, 2] > 0.01) & (world[:, 2] <= 1.6)   # above the floor
    for side, hidden in ((world[:, 0] > 0, geom_hidden), (world[:, 0] < 0, mat_hidden)):
        when_hidden = np.isin(cast, hidden)
        assert (on_screen & side & ~when_hidden).sum() > 50, (on_screen & side & ~when_hidden).sum()
        assert (on_screen & side & when_hidden).sum() == 0, (on_screen & side & when_hidden).sum()
        assert (side & when_hidden).sum() > 50   # the rays went past it


# ------------------------------------------------------------------ _HeadCamera
class _Node:
    """Stands in for the bridge's node: each publisher appends to a list"""

    def __init__(self):
        self.sent = {}

    def create_publisher(self, kind, topic, qos):
        sent = self.sent.setdefault(topic, [])

        class Publisher:
            publish = staticmethod(sent.append)
        return Publisher()


def test_head_camera_thread_renders_the_state_at_its_tick():
    """The head_camera thread's images are HeadCamera.render() of the state the
    physics thread had at tick(), although the state has moved on before the
    thread renders; tick() only fills the scene, and a frame due while the
    thread renders is dropped. Depth is bit for bit; colour can differ by one
    level on a few pixels, as two renders of one scene on one thread do too
    (MuJoCo's 4x multisampling: 15-94 of 921600 pixels, by 1; none with it off)"""
    model = _model()
    data = _standing(model)
    node = _Node()
    camera = bridge._HeadCamera(node, model, data, "", bridge.SensorConfig(lidar=False))
    reference = g1_head_camera.HeadCamera(model, "")
    period = camera._schedule.period
    rng = np.random.default_rng(0)
    expected, tick_ms = [], []
    try:
        for k in range(6):
            data.qpos[7:] = rng.uniform(-0.4, 0.4, model.nq - 7)   # the arms move through the view
            mujoco.mj_forward(model, data)
            assert camera._idle.wait(10.0)
            start = time.perf_counter()
            camera.tick(data, 2 * k * period)
            tick_ms.append((time.perf_counter() - start) * 1e3)
            if k == 3:   # the next frame due at once: the thread is still rendering
                camera.tick(data, (2 * k + 1) * period)
            expected.append(reference.render(data))
            data.qpos[7:] = 0.0   # the next steps, before the thread renders
            mujoco.mj_forward(model, data)
    finally:
        camera.close()
        reference.close()
    color = node.sent["/realsense/head/color/image_raw"]
    depth = node.sent["/realsense/head/aligned_depth_to_color/image_raw"]
    info = node.sent["/realsense/head/color/camera_info"]
    assert camera.published == len(color) == len(depth) == len(info) == 6, camera.stats()
    assert camera.dropped == 1 and camera.rendered == 6, camera.stats()
    assert len(node.sent["/tf_static"]) == 1
    for (rgb, mm), c, d in zip(expected, color, depth):
        assert bytes(d.data) == mm.astype("<u2").tobytes()
        got = np.frombuffer(bytes(c.data), np.uint8).reshape(rgb.shape)
        off = (got != rgb).any(axis=2)
        assert np.abs(got.astype(int) - rgb).max() <= 1 and off.mean() < 1e-3, (off.sum(), off.mean())
    assert len({bytes(c.data) for c in color}) == 6   # the frames differ: the arms moved
    k = info[0].k
    assert (k[0], k[4], k[2], k[5]) == (reference.fx, reference.fy, reference.cx, reference.cy)
    print(f"  6 frames as render() gives them; tick {max(tick_ms):.2f} ms at most on the physics thread")


# ------------------------------------------------------------------ G1Sensors
def test_g1_sensors_stamp_one_step_before_sim_t_and_build_what_the_config_enables():
    """tick(sim_t, stamp) hands the sensors sim_t - dt, the state mj_step started
    from, so the first sample of the first tick (sim_t = dt) is stamped 0 and
    none of a second of ticks is stamped 1 s; the config picks the sensors, and
    stats() covers them"""
    model = _model()
    data = _standing(model)
    dt = model.opt.timestep
    node = _Node()
    sensors = bridge.G1Sensors(node, model, data, "", bridge.SensorConfig(camera=False))
    try:
        assert sensors.head_camera is None and "/realsense/head/color/image_raw" not in node.sent
        for step in range(1, 501):
            while sensors.mid360.scans._free.empty():   # the loop outruns real time, as in _submit
                time.sleep(0.0005)
            sim_t = step * dt
            sensors.tick(sim_t, bridge._sim_time_to_msg(sim_t))
    finally:
        sensors.shutdown()
    stamps = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec for m in node.sent["/livox/imu"]]
    assert stamps[0] == 0 and 0 < stamps[-1] < 10**9, (stamps[0], stamps[-1])
    assert len(stamps) == sensors.mid360.imu_samples == 200
    assert "200 IMU samples" in sensors.stats() and "head camera" not in sensors.stats()
    none = bridge.G1Sensors(_Node(), model, data, "", bridge.SensorConfig(lidar=False, camera=False))
    none.tick(dt, bridge._sim_time_to_msg(dt))
    none.shutdown()
    assert none.mid360 is None and none.head_camera is None and none.stats() == ""


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
