"""Check g1.g1_sensors without DDS: its knobs, the MID-360's scan thread and the head camera's render thread.

SensorConfig reads the env knobs; _LidarScans' frames go to a list instead of a publisher,
and _HeadCamera's messages go to a stand-in node's lists. No ROS node is created. The camera
needs GL (MUJOCO_GL=egl with the GPU, or Mesa's).

The G1 is g1_robosuite's robot MJCF (names unprefixed) in a room with two screens, one
coloured by its geom's rgba and one by its material's, so that both alphas MuJoCo's rays
read can change at runtime, as RoboCasa tasks change them. Run in the robocasa image with
ROS 2 Humble and the Livox messages sourced and golem_robocasa on PYTHONPATH, as
tests/robocasa/g1/run_in_container.sh does.
"""
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from livox_ros_driver2.msg import CustomMsg
from rclpy.serialization import deserialize_message

import mid360
from g1 import g1_head_camera
from g1 import g1_lidar as lidar
from g1 import g1_robosuite
from g1 import g1_sensor_msgs as sm
from g1 import g1_sensors as bridge

PELVIS_Z = 0.793  # m, the MJCF's pelvis height with every joint at 0
SCREEN_X = 1.0  # m: the screens' faces towards the robot, ahead (geom alpha) and behind (material alpha)
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
    """Return the G1 robot MJCF compiled inside ROOM."""
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
    """Return data with the robot upright at PELVIS_Z, every joint at 0."""
    data = mujoco.MjData(model)
    data.qpos[2] = PELVIS_Z
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    return data


def _submit(scans, data, t):
    """Submit a cast once the ring has a free slot.

    The loops here outrun real time, and a skipped cast would change the frames.
    """
    while scans._free.empty():
        time.sleep(0.0005)
    scans.submit(data, t)


def _backends():
    """Return the lidar backends to test here: cpu, and gpu when a GPU renders (else say why not)."""
    try:
        mid360.GpuCaster(_model(), -1, lidar.GEOM_GROUPS).close()
    except mid360.GpuUnavailable as e:
        print(f"  (gpu skipped: {e})")
        return ("cpu",)
    return ("cpu", "gpu")


def _wait_for(frames, count, timeout=30.0):
    """Wait up to `timeout` s of wall time for `frames` to hold `count` items."""
    end = time.time() + timeout
    while len(frames) < count and time.time() < end:
        time.sleep(0.01)


def test_sensor_config_defaults():
    """Unset or empty knobs (compose passes unset ones as empty) give the defaults."""
    default = bridge.SensorConfig()
    assert bridge.SensorConfig.from_env({}) == default
    empty = {k: "" for k in ("GOLEM_LIDAR", "GOLEM_LIDAR_POINTS", "GOLEM_LIDAR_HZ", "GOLEM_LIDAR_BACKEND",
                             "GOLEM_CAMERAS", "GOLEM_CAMERA_HZ", "GOLEM_CAMERA_SCALE")}
    assert bridge.SensorConfig.from_env(empty) == default
    assert (default.lidar, default.lidar_points, default.lidar_hz, default.imu_hz) == (True, 20_000, 10.0, 200.0)
    assert default.lidar_backend == "auto"
    assert (default.camera, default.camera_hz, default.camera_width, default.camera_height) == \
        (True, 6.0, 640, 360)


def test_sensor_config_knobs():
    """Each knob sets its value; the camera scale sets the image size."""
    config = bridge.SensorConfig.from_env({"GOLEM_LIDAR": "0", "GOLEM_CAMERAS": "off", "GOLEM_LIDAR_POINTS": "2000",
                                           "GOLEM_LIDAR_HZ": "100", "GOLEM_CAMERA_HZ": "30",
                                           "GOLEM_CAMERA_SCALE": "0.25", "GOLEM_LIDAR_BACKEND": " CPU "})
    assert (config.lidar, config.camera, config.lidar_points, config.lidar_hz) == (False, False, 2000, 100.0)
    assert config.lidar_backend == "cpu"
    assert (config.camera_hz, config.camera_width, config.camera_height) == (30.0, 320, 180)


def test_sensor_config_rejects_bad_knobs():
    """A bad knob raises ValueError naming it, NaN and infinity included.

    It fails before the kitchen is built; a NaN rate would make the schedule due on every
    physics step.
    """
    bad = [("GOLEM_LIDAR_HZ", v) for v in ("nan", "inf", "-inf", "0", "-1", "101", "fast")]
    bad += [("GOLEM_CAMERA_HZ", v) for v in ("nan", "NaN", "inf", "0", "-6", "30.5", "60")]
    bad += [("GOLEM_LIDAR_POINTS", v) for v in ("nan", "inf", "0", "-5", "2.5", "1e4")]
    bad += [("GOLEM_CAMERA_SCALE", v) for v in ("nan", "inf", "0", "-0.5", "half", "640x360", "1e-4")]
    bad += [("GOLEM_LIDAR_BACKEND", v) for v in ("cuda", "0", "egl")]
    for name, value in bad:
        try:
            bridge.SensorConfig.from_env({name: value})
        except ValueError as e:
            assert name in str(e), (name, value, e)
        else:
            raise AssertionError(f"{name}={value!r} was accepted")


def test_lidar_thread_matches_live_casts():
    """The scan thread's frames are byte-identical to casts on the live model at the same times.

    The thread works from pose snapshots on its own MjData while the physics steps beside it:
    the robot flails under random torques and falls, and both screens' alphas change at
    runtime. The physics submits a cast every caster's batch_s of sim time (10 ms on the CPU,
    20 ms on the GPU), as G1Sensors.tick() does; for each backend the reference casts with that
    backend's caster, on the live model's poses and alpha.
    """
    for backend in _backends():
        model = _model()
        data = _standing(model)
        config = bridge.SensorConfig(lidar_backend=backend)
        got = []
        scans = bridge._LidarScans(model, data, "", config, got.append)
        sensor = lidar.Mid360(model, "")
        live = lidar.make_scanner(sensor.make_caster(backend), config.lidar_points, config.lidar_hz)
        try:
            assert scans.interval == live.caster.batch_s == {"cpu": 0.01, "gpu": 0.02}[backend]
            # The caster casts on its own copy of the model: the CPU's cast() edits body_bvhadr,
            # and both write the alpha into the model they cast on.
            for field in ("body_bvhadr", "geom_rgba", "mat_rgba"):
                assert not np.shares_memory(getattr(scans.caster.model, field), getattr(model, field)), field
            schedule = bridge._Schedule(1.0 / scans.interval)
            want = []
            screen = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screen_geom")
            rng = np.random.default_rng(3)
            z = []
            for step in range(2500):  # 5 s of sim time
                if step in (800, 1600, 2000):  # hide the screens, then show them again
                    model.geom_rgba[screen, 3] = 0.0 if step == 800 else 1.0
                    model.mat_rgba[0, 3] = 0.0 if step == 1600 else 1.0
                data.ctrl[:] = rng.uniform(-8.0, 8.0, model.nu)
                mujoco.mj_step(model, data)
                t = float(data.time) - model.opt.timestep
                if schedule.due(t):
                    _submit(scans, data, t)
                    frame = live.update(sensor.snapshot(data, model.geom_rgba[:, 3].copy(),
                                                        model.mat_rgba[:, 3].copy()), t)
                    if frame is not None:
                        want.append(sm.livox_custom_msg_cdr(frame))
                z.append(float(data.qpos[2]))
            _wait_for(got, len(want))
        finally:
            scans.close()
            live.caster.close()
        assert max(z) - min(z) > 0.3, (max(z), min(z))  # the robot fell
        assert scans.skipped == 0, scans.skipped
        assert len(want) == 49 and len(got) == len(want), (backend, len(got), len(want))
        same = sum(bytes(a) == bytes(b) for a, b in zip(got, want))
        assert same == len(want), f"{backend}: {same}/{len(want)} frames byte-identical"
        print(f"  {backend}: {len(want)} frames byte-identical")


def test_lidar_thread_sees_runtime_alpha_edits():
    """Each cast uses the alpha in effect when its poses were taken, on either backend.

    No point lies on a screen while it is hidden (alpha 0, by its geom or by its material);
    some do while it is shown.
    """
    for backend in _backends():
        model = _model()
        data = _standing(model)
        screen = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screen_geom")
        got = []
        scans = bridge._LidarScans(model, data, "", bridge.SensorConfig(lidar_backend=backend), got.append)
        period = scans.interval  # s of sim time between casts
        # Cast indices: the geom screen hidden over 0.2-0.4 s, the material one over 0.3-0.5 s.
        geom_hidden = range(round(0.2 / period), round(0.4 / period))
        mat_hidden = range(round(0.3 / period), round(0.5 / period))
        try:
            for k in range(round(0.6 / period) + 1):  # 0.6 s: six frames
                model.geom_rgba[screen, 3] = 0.0 if k in geom_hidden else 1.0
                model.mat_rgba[0, 3] = 0.0 if k in mat_hidden else 1.0
                _submit(scans, data, k * period)  # the robot does not move
            _wait_for(got, 6)
        finally:
            scans.close()
        assert len(got) == 6, (backend, len(got))
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
        # Within the range noise (2 cm, 1 sigma) of a screen's face, above the floor.
        on_screen = (np.abs(np.abs(world[:, 0]) - SCREEN_X) < 0.1) & (np.abs(world[:, 1]) <= 0.6) & \
            (world[:, 2] > 0.1) & (world[:, 2] <= 1.6)
        for side, hidden in ((world[:, 0] > 0, geom_hidden), (world[:, 0] < 0, mat_hidden)):
            when_hidden = np.isin(cast, hidden)
            assert (on_screen & side & ~when_hidden).sum() > 50, (backend, (on_screen & side & ~when_hidden).sum())
            assert (on_screen & side & when_hidden).sum() == 0, (backend, (on_screen & side & when_hidden).sum())
            assert (side & when_hidden).sum() > 50  # the rays went past it


def test_casts_come_every_batch_or_every_frame():
    """The physics submits a cast every caster's batch_s, or every frame period when frames come faster.

    So every frame is cast from a pose of its own: at 100 Hz the GPU caster's 20 ms batch would
    merge frames.
    """
    model = _model()
    data = _standing(model)
    for backend in _backends():
        batch = {"cpu": 0.01, "gpu": 0.02}[backend]
        for hz, interval in ((10.0, batch), (100.0, 0.01)):
            scans = bridge._LidarScans(model, data, "", bridge.SensorConfig(lidar_backend=backend, lidar_hz=hz),
                                       [].append)
            try:
                assert scans.caster.batch_s == batch and scans.interval == interval, (backend, hz, scans.interval)
            finally:
                scans.close()


class _Node:
    """Stand in for the bridge's node: each publisher appends to a list per topic.

    `subscribers` maps a topic to its subscription count (0 for a topic it does not name).
    """

    def __init__(self):
        """Start with no topics and no subscribers."""
        self.sent = {}
        self.subscribers = {}

    def create_publisher(self, kind, topic, qos):
        """Return a publisher that appends to the topic's list."""
        sent = self.sent.setdefault(topic, [])
        subscribers = self.subscribers

        class Publisher:
            """Append each published message to the topic's list."""

            publish = staticmethod(sent.append)

            @staticmethod
            def get_subscription_count():
                """Return the topic's subscription count."""
                return subscribers.get(topic, 0)
        return Publisher()


def test_head_camera_thread_renders_the_state_at_its_tick():
    """The render thread's images are HeadCamera.render() of the state at tick(), not at render time.

    tick() only fills the scene, and a frame due while the thread renders is dropped. Depth is
    bit for bit; colour can differ by one level on a few pixels, as two renders of one scene on
    one thread do too (MuJoCo's 4x multisampling: a few pixels in 1e3, by 1; none with it off).
    """
    model = _model()
    data = _standing(model)
    node = _Node()
    config = bridge.SensorConfig(lidar=False)
    camera = bridge._HeadCamera(node, model, data, "", config)
    reference = g1_head_camera.HeadCamera(model, "", config.camera_width, config.camera_height)
    period = camera._schedule.period
    rng = np.random.default_rng(0)
    expected, tick_ms = [], []
    try:
        for k in range(6):
            data.qpos[7:] = rng.uniform(-0.4, 0.4, model.nq - 7)  # the arms move through the view
            mujoco.mj_forward(model, data)
            assert camera._idle.wait(10.0)
            start = time.perf_counter()
            camera.tick(data, 2 * k * period)
            tick_ms.append((time.perf_counter() - start) * 1e3)
            if k == 3:  # the next frame due at once: the thread is still rendering
                camera.tick(data, (2 * k + 1) * period)
            expected.append(reference.render(data))
            data.qpos[7:] = 0.0  # the next steps, before the thread renders
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
    assert len({bytes(c.data) for c in color}) == 6  # the frames differ: the arms moved
    k = info[0].k
    assert (k[0], k[4], k[2], k[5]) == (reference.fx, reference.fy, reference.cx, reference.cy)
    print(f"  6 frames as render() gives them; tick {max(tick_ms):.2f} ms at most on the physics thread")


def test_g1_compressed_not_encoded_without_a_subscriber():
    """The head camera encodes a compressed image only while its topic has a subscriber.

    The raw images and camera_info publish regardless; each compressed topic is judged for
    every frame on its own, so a subscriber that appears or leaves changes the next frame.
    """
    model = _model()
    data = _standing(model)
    node = _Node()
    config = bridge.SensorConfig(lidar=False)
    camera = bridge._HeadCamera(node, model, data, "", config)
    jpeg_topic = "/realsense/head/color/image_raw/compressed"
    png_topic = "/realsense/head/aligned_depth_to_color/image_raw/compressedDepth"
    encoded = []
    real = {name: getattr(sm, name) for name in ("color_compressed_msg", "depth_compressed_msg")}

    def counting(name):
        """Return sm's encoder `name` that also records its calls."""
        def encode(*args, **kwargs):
            encoded.append(name)
            return real[name](*args, **kwargs)
        return encode

    for name in real:
        setattr(sm, name, counting(name))
    try:
        period = camera._schedule.period
        # Frames 0-3: no subscriber, then the JPEG topic's, then both, then only the depth's.
        steps = [({}, [], 0, 0), ({jpeg_topic: 1}, ["color_compressed_msg"], 1, 0),
                 ({jpeg_topic: 1, png_topic: 2}, ["color_compressed_msg", "depth_compressed_msg"], 2, 1),
                 ({png_topic: 1}, ["depth_compressed_msg"], 2, 2)]
        for k, (subscribers, calls, jpegs, pngs) in enumerate(steps):
            node.subscribers.clear()
            node.subscribers.update(subscribers)
            assert camera._idle.wait(10.0)
            encoded.clear()
            camera.tick(data, k * period)
            deadline = time.time() + 10.0
            while camera.published <= k and time.time() < deadline:
                time.sleep(0.01)
            assert camera.published == k + 1, camera.stats()
            assert sorted(encoded) == sorted(calls), (k, encoded)
            assert (len(node.sent[jpeg_topic]), len(node.sent[png_topic])) == (jpegs, pngs), k
            assert len(node.sent["/realsense/head/color/image_raw"]) == k + 1
    finally:
        for name, encode in real.items():
            setattr(sm, name, encode)
        camera.close()


def test_g1_sensors_stamp_one_step_before_sim_t_and_build_what_the_config_enables():
    """tick(sim_t, stamp) stamps samples sim_t - dt, and the config picks which sensors exist.

    sim_t - dt is the state mj_step started from, so the first sample of the first tick
    (sim_t = dt) is stamped 0 and none of a second of ticks is stamped 1 s; stats() covers the
    sensors built.
    """
    model = _model()
    data = _standing(model)
    dt = model.opt.timestep
    node = _Node()
    sensors = bridge.G1Sensors(node, model, data, "", bridge.SensorConfig(camera=False))
    casts = round(1.0 / sensors.mid360.scans.interval)  # a cast every batch_s of the second's ticks
    try:
        assert sensors.head_camera is None and "/realsense/head/color/image_raw" not in node.sent
        for step in range(1, 501):
            while sensors.mid360.scans._free.empty():  # the loop outruns real time, as in _submit
                time.sleep(0.0005)
            sim_t = step * dt
            sensors.tick(sim_t, bridge._sim_time_to_msg(sim_t))
    finally:
        sensors.shutdown()
    stamps = [m.header.stamp.sec * 10**9 + m.header.stamp.nanosec for m in node.sent["/livox/imu"]]
    assert stamps[0] == 0 and 0 < stamps[-1] < 10**9, (stamps[0], stamps[-1])
    assert len(stamps) == sensors.mid360.imu_samples == 200
    assert "200 IMU samples" in sensors.stats() and "head camera" not in sensors.stats()
    # Frames complete at 0.1 ... 0.9 s; the last tick is stamped 0.998 s.
    assert f"9 lidar frames (0 of {casts} casts skipped)" in sensors.stats(), sensors.stats()
    none = bridge.G1Sensors(_Node(), model, data, "", bridge.SensorConfig(lidar=False, camera=False))
    none.tick(dt, bridge._sim_time_to_msg(dt))
    none.shutdown()
    assert none.mid360 is None and none.head_camera is None and none.stats() == ""


if __name__ == "__main__":
    # Each file runs as a script (run_in_container.sh), so it runs its own test_* functions.
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
