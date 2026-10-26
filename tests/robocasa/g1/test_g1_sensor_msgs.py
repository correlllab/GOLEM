"""Tests for g1_sensor_msgs: the MID-360's and D435's ROS 2 messages, field by field.

No DDS: messages are only built and (de)serialized. Run in the robocasa image
with ROS 2 Humble and the Livox messages sourced and h1_robocasa on PYTHONPATH,
as tests/robocasa/g1/run_in_container.sh does.
"""
import io
import struct
import sys

import mujoco
import numpy as np
from livox_ros_driver2.msg import CustomMsg
from PIL import Image as PILImage
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage, Image

from g1 import g1_lidar as lidar
from g1 import g1_sensor_msgs as msgs


def _frame(n, timebase=12.3456789012, seed=0):
    rng = np.random.default_rng(seed)
    return lidar.LidarFrame(
        timebase=timebase,
        offset_time=np.sort(rng.integers(0, 100_000_000, n)).astype(np.uint32),
        points=rng.uniform(-5.0, 5.0, (n, 3)).astype(np.float32),
        line=(np.arange(n) % 4).astype(np.uint8))


def test_time_msg():
    assert (msgs.time_msg(12.3456789012).sec, msgs.time_msg(12.3456789012).nanosec) == (12, 345678901)
    assert (msgs.time_msg(0.1).sec, msgs.time_msg(0.1).nanosec) == (0, 100_000_000)
    assert (msgs.time_msg(1.9999999999).sec, msgs.time_msg(1.9999999999).nanosec) == (2, 0)


def test_custom_msg_fields():
    frame = _frame(5)
    msg = msgs.livox_custom_msg(frame)
    assert msg.header.frame_id == "livox_frame"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 345678901)
    assert msg.timebase == 12_345_678_901 and msg.point_num == 5
    assert msg.lidar_id == 192 and list(msg.rsvd) == [0, 0, 0]
    for p, offset, xyz, line in zip(msg.points, frame.offset_time, frame.points, frame.line):
        assert p.offset_time == offset and p.line == line
        assert np.array_equal(np.float32([p.x, p.y, p.z]), xyz)
        assert p.reflectivity == msgs.REFLECTIVITY and p.tag == 0


def test_custom_msg_cdr_is_rclpys_serialization():
    """Byte for byte, for frame sizes and frame_id lengths that move the alignment"""
    for n in (0, 1, 2, 3, 7, 1000):
        for frame_id in ("livox_frame", "a", "ab", "abc", "abcd", "abcdefgh"):
            frame = _frame(n, timebase=n * 0.1 + 0.000123, seed=n)
            cdr = msgs.livox_custom_msg_cdr(frame, frame_id)
            assert cdr == serialize_message(msgs.livox_custom_msg(frame, frame_id)), (n, frame_id)
            back = deserialize_message(cdr, CustomMsg)
            assert back.header.frame_id == frame_id and back.point_num == n


def test_custom_msg_of_a_scan():
    """A real scan through the serialized form: what a subscriber reads back"""
    import test_g1_sensors as scene
    data = mujoco.MjData(scene.ROOM_MODEL)
    scene._place(scene.ROOM_MODEL, data)
    sensor = lidar.Mid360(scene.ROOM_MODEL, prefix="")
    sensor.update(data, 7.0)
    frame = sensor.update(data, 7.1)
    msg = deserialize_message(msgs.livox_custom_msg_cdr(frame), CustomMsg)
    assert msg.point_num == len(msg.points) == len(frame.points) > 0.9 * sensor.points_per_frame
    assert msg.timebase == 7_000_000_000 and (msg.header.stamp.sec, msg.header.stamp.nanosec) == (7, 0)
    assert msg.lidar_id == 192 and list(msg.rsvd) == [0, 0, 0]
    xyz = np.array([(p.x, p.y, p.z) for p in msg.points], dtype=np.float32)
    assert np.array_equal(xyz, frame.points)
    assert [p.offset_time for p in msg.points] == frame.offset_time.tolist()
    assert msg.points[0].offset_time == 0 and msg.points[-1].offset_time == 100_000_000
    assert [p.line for p in msg.points] == frame.line.tolist()
    assert {p.reflectivity for p in msg.points} == {msgs.REFLECTIVITY} and {p.tag for p in msg.points} == {0}


def test_imu_msg():
    msg = msgs.livox_imu_msg(3.005, (0.1, -0.2, 0.3), (-0.05, 0.0, -0.9987))
    assert msg.header.frame_id == "livox_frame"
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (3, 5_000_000)
    w, a = msg.angular_velocity, msg.linear_acceleration
    assert (w.x, w.y, w.z) == (0.1, -0.2, 0.3) and (a.x, a.y, a.z) == (-0.05, 0.0, -0.9987)
    # the driver sets nothing else: the default orientation, zero covariances
    q = msg.orientation
    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)
    for cov in (msg.orientation_covariance, msg.angular_velocity_covariance,
                msg.linear_acceleration_covariance):
        assert list(cov) == [0.0] * 9


def _images(height=36, width=64):
    v, u = np.mgrid[:height, :width]
    rgb = np.stack([4 * u, 7 * v, (2 * u + 3 * v) % 256], axis=-1).astype(np.uint8)
    depth = (u * 1000 + v * 7).astype(np.uint16)
    depth[0, 0], depth[0, 1], depth[1, 0] = 0, 65535, 1
    return rgb, depth


def test_color_msgs():
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
    assert np.abs(decoded.astype(int) - rgb).mean() < 4   # lossy, but these colours, not BGR
    assert deserialize_message(serialize_message(jpeg), CompressedImage).format == jpeg.format


def test_depth_msgs():
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
    assert (~near).sum() > 100 and depth[0, 1] == 65535   # the raw image keeps depths over 10 m
    assert np.array_equal(decoded[near], depth[near])      # lossless up to the plugin's 10 m
    assert (decoded[~near] == 0).all()                     # and 0 beyond, as the plugin does


def test_camera_info():
    info = msgs.camera_info_msg(4.0, 1280, 720, 649.4, 649.4, 639.5, 359.5)
    assert info.header.frame_id == "head_color_optical_frame"
    assert (info.width, info.height, info.distortion_model) == (1280, 720, "plumb_bob")
    assert list(info.d) == [0.0] * 5
    assert list(info.k) == [649.4, 0.0, 639.5, 0.0, 649.4, 359.5, 0.0, 0.0, 1.0]
    assert list(info.p) == [649.4, 0.0, 639.5, 0.0, 0.0, 649.4, 359.5, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert list(info.r) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


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
