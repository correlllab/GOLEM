"""Check g1_sensor_msgs: the G1 sim's MID-360 and D435 ROS 2 messages, field by field.

No DDS: messages are only built and (de)serialised. Run in the robocasa image with ROS 2
Humble and the Livox messages sourced and golem_robocasa on PYTHONPATH, as
tests/robocasa/g1/run_in_container.sh does.
"""
import io
import struct
import sys

import mujoco
import numpy as np
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from PIL import Image as PILImage
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage, Image

import mid360
from g1 import g1_lidar as lidar
from g1 import g1_sensor_msgs as msgs


def _frame(n, timebase=12.3456789012, seed=0):
    """Return a random lidar frame of `n` points at `timebase` (s), offsets sorted.

    Every third point, from the second on, is a ray without a return, at (0, 0, 0).
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32)
    points[1::3] = 0.0
    return mid360.LidarFrame(
        timebase=timebase,
        offset_time=np.sort(rng.integers(0, 100_000_000, n)).astype(np.uint32),
        points=points, line=(np.arange(n) % 4).astype(np.uint8), rays=n)


def _custom_msg(frame, frame_id="livox_frame"):
    """Return the CustomMsg the driver publishes for a frame, built field by field: the CDR's reference.

    Reflectivity is REFLECTIVITY for a return and 0 for a ray without one, at (0, 0, 0).
    """
    msg = CustomMsg()
    msg.header.stamp = msgs.time_msg(frame.timebase)
    msg.header.frame_id = frame_id
    msg.timebase = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    msg.point_num = len(frame.points)
    msg.lidar_id = 192
    msg.points = [CustomPoint(offset_time=o, x=p[0], y=p[1], z=p[2], reflectivity=msgs.REFLECTIVITY if any(p) else 0,
                              tag=0, line=line)
                  for o, p, line in zip(frame.offset_time.tolist(), frame.points.tolist(), frame.line.tolist())]
    return msg


def test_time_msg():
    """Sim seconds become (sec, nanosec) rounded to the nanosecond, carrying into the seconds."""
    assert (msgs.time_msg(12.3456789012).sec, msgs.time_msg(12.3456789012).nanosec) == (12, 345678901)
    assert (msgs.time_msg(0.1).sec, msgs.time_msg(0.1).nanosec) == (0, 100_000_000)
    assert (msgs.time_msg(1.9999999999).sec, msgs.time_msg(1.9999999999).nanosec) == (2, 0)


def test_custom_msg_fields():
    """The CustomMsg carries the frame's stamp, points, offsets and lines with the driver's fixed fields.

    A point without a return, (0, 0, 0), has reflectivity 0, as the real driver's do.
    """
    frame = _frame(5)
    msg = deserialize_message(msgs.livox_custom_msg_cdr(frame), CustomMsg)
    assert msg.header.frame_id == "livox_frame"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 345678901)
    assert msg.timebase == 12_345_678_901 and msg.point_num == 5
    assert msg.lidar_id == 192 and list(msg.rsvd) == [0, 0, 0]
    for p, offset, xyz, line in zip(msg.points, frame.offset_time, frame.points, frame.line):
        assert p.offset_time == offset and p.line == line
        assert np.array_equal(np.float32([p.x, p.y, p.z]), xyz)
        assert p.reflectivity == (msgs.REFLECTIVITY if xyz.any() else 0) and p.tag == 0
    assert [p.reflectivity for p in msg.points] == [msgs.REFLECTIVITY, 0, msgs.REFLECTIVITY,
                                                    msgs.REFLECTIVITY, 0]


def test_custom_msg_cdr_is_rclpys_serialization():
    """The hand-packed CDR equals rclpy's serialisation for sizes and frame_id lengths that move the alignment."""
    for n in (0, 1, 2, 3, 7, 1000):
        for frame_id in ("livox_frame", "a", "ab", "abc", "abcd", "abcdefgh"):
            frame = _frame(n, timebase=n * 0.1 + 0.000123, seed=n)
            cdr = msgs.livox_custom_msg_cdr(frame, frame_id)
            assert cdr == serialize_message(_custom_msg(frame, frame_id)), (n, frame_id)
            back = deserialize_message(cdr, CustomMsg)
            assert back.header.frame_id == frame_id and back.point_num == n


def test_custom_msg_of_a_scan():
    """A real scan survives serialisation: a subscriber reads back every point, offset and line."""
    import test_g1_sensors as scene
    data = mujoco.MjData(scene.ROOM_MODEL)
    scene.place(scene.ROOM_MODEL, data)
    sensor = lidar.Mid360(scene.ROOM_MODEL, prefix="")
    scanner = lidar.make_scanner(scene.exact_caster(sensor))
    try:
        scanner.update(scene.snapshot(sensor, data), 7.0)
        frame = scanner.update(scene.snapshot(sensor, data), 7.1)
    finally:
        scanner.caster.close()
    msg = deserialize_message(msgs.livox_custom_msg_cdr(frame), CustomMsg)
    assert msg.point_num == len(msg.points) == len(frame.points) == scanner.points_per_frame
    assert scanner.points_per_frame == 20_000
    returns = frame.points.any(axis=1)
    assert returns.sum() > 0.9 * scanner.points_per_frame
    assert msg.timebase == 7_000_000_000 and (msg.header.stamp.sec, msg.header.stamp.nanosec) == (7, 0)
    assert msg.lidar_id == 192 and list(msg.rsvd) == [0, 0, 0]
    xyz = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float32)
    assert np.array_equal(xyz, frame.points)
    assert [p.offset_time for p in msg.points] == frame.offset_time.tolist()
    # Cast by the two calls: the frame's first ray at 7.0 s, the rest at 7.1 s.
    assert {p.offset_time for p in msg.points} <= {0, 100_000_000}
    assert msg.points[-1].offset_time == 100_000_000
    assert [p.line for p in msg.points] == frame.line.tolist()
    reflectivity = np.array([p.reflectivity for p in msg.points])
    assert (reflectivity == np.where(returns, msgs.REFLECTIVITY, 0)).all()
    assert {p.tag for p in msg.points} == {0}


def test_imu_msg():
    """The IMU message carries the stamp, rates and accelerations, and nothing the driver leaves unset."""
    msg = msgs.livox_imu_msg(3.005, (0.1, -0.2, 0.3), (-0.05, 0.0, -0.9987))
    assert msg.header.frame_id == "livox_frame"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (3, 5_000_000)
    w, a = msg.angular_velocity, msg.linear_acceleration
    assert (w.x, w.y, w.z) == (0.1, -0.2, 0.3) and (a.x, a.y, a.z) == (-0.05, 0.0, -0.9987)
    # The driver sets nothing else: the default orientation and zero covariances.
    q = msg.orientation
    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)
    for cov in (msg.orientation_covariance, msg.angular_velocity_covariance,
                msg.linear_acceleration_covariance):
        assert list(cov) == [0.0] * 9


def _images(height=36, width=64):
    """Return a patterned RGB image and a depth image (mm) with 0, 65535 and 1 in its corner."""
    v, u = np.mgrid[:height, :width]
    rgb = np.stack([4 * u, 7 * v, (2 * u + 3 * v) % 256], axis=-1).astype(np.uint8)
    depth = (u * 1000 + v * 7).astype(np.uint16)
    depth[0, 0], depth[0, 1], depth[1, 0] = 0, 65535, 1
    return rgb, depth


def test_color_msgs():
    """The raw and JPEG colour messages carry the image in RGB order with the right layout and stamp."""
    rgb, _ = _images()
    raw = msgs.color_image_msg(1.5, rgb)
    assert raw.header.frame_id == "head_color_optical_frame" and raw.encoding == "rgb8"
    assert (raw.height, raw.width, raw.step, raw.is_bigendian) == (36, 64, 192, 0)
    assert bytes(raw.data) == rgb.tobytes()
    back = deserialize_message(serialize_message(raw), Image)
    assert bytes(back.data) == rgb.tobytes()
    jpeg = msgs.color_compressed_msg(1.5, rgb)
    assert jpeg.format == "rgb8; jpeg compressed bgr8"
    assert (jpeg.header.stamp.sec, jpeg.header.stamp.nanosec) == (1, 500_000_000)
    decoded = np.array(PILImage.open(io.BytesIO(bytes(jpeg.data))))
    assert decoded.shape == rgb.shape
    assert np.abs(decoded.astype(int) - rgb).mean() < 4  # lossy, but these colours, not BGR
    assert deserialize_message(serialize_message(jpeg), CompressedImage).format == jpeg.format


def test_depth_msgs():
    """Raw depth is lossless; compressedDepth is lossless to 10 m and 0 beyond, as the ROS plugin does."""
    _, depth = _images()
    raw = msgs.depth_image_msg(2.0, depth)
    assert raw.encoding == "16UC1" and (raw.height, raw.width, raw.step, raw.is_bigendian) == (36, 64, 128, 0)
    assert np.array_equal(np.frombuffer(bytes(raw.data), "<u2").reshape(depth.shape), depth)
    packed = msgs.depth_compressed_msg(2.0, depth)
    assert packed.format == "16UC1; compressedDepth"
    data = bytes(packed.data)
    assert struct.unpack("<iff", data[:12]) == (0, 0.0, 0.0)
    png = PILImage.open(io.BytesIO(data[12:]))
    assert png.format == "PNG" and png.mode in ("I;16", "I;16B", "I")
    decoded = np.array(png).astype(np.uint16)
    near = depth <= 10000
    assert (~near).sum() > 100 and depth[0, 1] == 65535  # the raw image keeps depths over 10 m
    assert np.array_equal(decoded[near], depth[near])  # lossless up to the plugin's 10 m
    assert (decoded[~near] == 0).all()  # and 0 beyond, as the plugin does


def test_camera_info():
    """camera_info carries the intrinsics as plumb_bob with no distortion and an identity R."""
    info = msgs.camera_info_msg(4.0, 1280, 720, 649.4, 649.4, 639.5, 359.5)
    assert info.header.frame_id == "head_color_optical_frame"
    assert (info.width, info.height, info.distortion_model) == (1280, 720, "plumb_bob")
    assert list(info.d) == [0.0] * 5
    assert list(info.k) == [649.4, 0.0, 639.5, 0.0, 649.4, 359.5, 0.0, 0.0, 1.0]
    assert list(info.p) == [649.4, 0.0, 639.5, 0.0, 0.0, 649.4, 359.5, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert list(info.r) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


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
