"""ROS 2 messages for the G1 sim's MID-360 and head D435, as their drivers publish them.

Message construction only: no node, publisher or DDS. g1_lidar.py and
g1_head_camera.py produce the data; the sim's ROS bridge publishes these.

MID-360 (livox_ros_driver2 in CustomMsg mode, xfer_format 1, msg_MID360_launch.py):
  /livox/lidar  livox_ros_driver2/CustomMsg, frame_id livox_frame (the launch
                file's frame_id), header stamp = timebase = the first point's time,
                offset_time in ns after it, lidar_id 192, rsvd 0. The sim has no
                reflectance model: every point has reflectivity REFLECTIVITY and
                tag 0 (a normal point). livox_custom_msg_cdr() gives the same message
                already serialized, which rclpy's Publisher.publish() accepts as
                bytes: building and converting 20,000 CustomPoint objects costs
                ~190 ms in Python, the serialized form under 1 ms.
  /livox/imu    sensor_msgs/Imu, frame_id livox_frame (hard-coded in the driver's
                lddc.cpp), angular_velocity in rad/s and linear_acceleration in g,
                as the driver passes the sensor's values through. The driver sets
                nothing else: the orientation stays at the message default
                (identity) and every covariance at 0.
RealSense D435 (realsense2_camera, camera_name head, namespace realsense,
align_depth on, in head_color_optical_frame):
  color/image_raw                               sensor_msgs/Image rgb8
  color/image_raw/compressed                    sensor_msgs/CompressedImage, the JPEG
                                                image_transport's compressed plugin
                                                makes: format "rgb8; jpeg compressed bgr8"
  aligned_depth_to_color/image_raw              sensor_msgs/Image 16UC1, mm
  aligned_depth_to_color/image_raw/compressedDepth
                                                sensor_msgs/CompressedImage, the
                                                compressedDepth plugin's: format
                                                "16UC1; compressedDepth", a 12-byte
                                                ConfigHeader (all zero), then a
                                                16-bit PNG, depths over 10 m at 0
  color/camera_info                             sensor_msgs/CameraInfo, plumb_bob, no
                                                distortion
"""
import array
import io
import struct

import numpy as np
from builtin_interfaces.msg import Time
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from PIL import Image as PILImage
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu

LIVOX_FRAME = "livox_frame"
HEAD_OPTICAL_FRAME = "head_color_optical_frame"
REFLECTIVITY = 100
TAG = 0
# the driver's lidar_id is its handle cast to uint8: inet_addr of the lidar's IP
# (192.168.123.120, the address in golem_slam's MID360_config.json; the G1 has no
# driver config of its own yet), whose low byte on a little-endian host is the
# first octet
LIDAR_ID = 192
# image_transport's plugins as installed in Humble (compressed/compressedDepth 2.5.5),
# checked by republishing through them: JPEG quality 95; compressedDepth zeroes
# depths over 10 m (its depth_max) and writes its PNG at a zlib level of 2 to 5
# (out.png_level is 3). PNG is lossless, so the level only sets the size.
JPEG_QUALITY = 95
DEPTH_MAX_MM = 10000
PNG_LEVEL = 3


def time_msg(t):
    """builtin_interfaces/Time of t seconds (sim time), to the nearest ns"""
    ns = int(round(t * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


# ------------------------------------------------------------------- MID-360
def livox_custom_msg(frame, frame_id=LIVOX_FRAME):
    """livox_ros_driver2/CustomMsg of a g1_lidar.LidarFrame (slow for full frames;
    publish livox_custom_msg_cdr() instead)"""
    msg = CustomMsg()
    msg.header.stamp = time_msg(frame.timebase)
    msg.header.frame_id = frame_id
    msg.timebase = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    msg.point_num = len(frame.points)
    msg.lidar_id = LIDAR_ID
    msg.points = [CustomPoint(offset_time=o, x=x, y=y, z=z, reflectivity=REFLECTIVITY, tag=TAG,
                              line=line)
                  for o, (x, y, z), line in zip(frame.offset_time.tolist(), frame.points.tolist(),
                                                frame.line.tolist())]
    return msg


# CustomPoint in CDR: uint32 offset_time, float32 x y z, uint8 reflectivity tag
# line; 19 bytes, and the next element's uint32 pads it to 20.
_CDR_POINT = np.dtype({"names": ["offset_time", "x", "y", "z", "reflectivity", "tag", "line"],
                       "formats": ["<u4", "<f4", "<f4", "<f4", "u1", "u1", "u1"],
                       "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 20})


def livox_custom_msg_cdr(frame, frame_id=LIVOX_FRAME):
    """The CustomMsg of livox_custom_msg(), serialized as rclpy serializes it (plain
    little-endian CDR, alignment counted from after the 4-byte encapsulation header)"""
    stamp = time_msg(frame.timebase)
    out = bytearray(b"\x00\x01\x00\x00")

    def align(n):
        out.extend(bytes(-(len(out) - 4) % n))

    name = frame_id.encode() + b"\x00"
    out += struct.pack("<iII", stamp.sec, stamp.nanosec, len(name)) + name
    align(8)
    count = len(frame.points)
    out += struct.pack("<QIB3x", stamp.sec * 1_000_000_000 + stamp.nanosec, count, LIDAR_ID)
    align(4)
    out += struct.pack("<I", count)
    if count:
        points = np.zeros(count, dtype=_CDR_POINT)
        points["offset_time"] = frame.offset_time
        points["x"], points["y"], points["z"] = frame.points.T
        points["reflectivity"] = REFLECTIVITY
        points["tag"] = TAG
        points["line"] = frame.line
        out += points.tobytes()[:-1]   # no padding after the last element
    return bytes(out)


def livox_imu_msg(t, angular_velocity, linear_acceleration, frame_id=LIVOX_FRAME):
    """sensor_msgs/Imu as the driver fills it: gyro rad/s, accelerometer in g"""
    msg = Imu()
    msg.header.stamp = time_msg(t)
    msg.header.frame_id = frame_id
    w, a = msg.angular_velocity, msg.linear_acceleration
    w.x, w.y, w.z = (float(v) for v in angular_velocity)
    a.x, a.y, a.z = (float(v) for v in linear_acceleration)
    return msg


# --------------------------------------------------------------------- D435
def _image(t, frame_id, pixels, encoding):
    msg = Image()
    msg.header.stamp = time_msg(t)
    msg.header.frame_id = frame_id
    msg.height, msg.width = pixels.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * pixels.itemsize * (pixels.shape[2] if pixels.ndim == 3 else 1)
    # an array.array is copied as is; bytes would be checked one by one
    msg.data = array.array("B", np.ascontiguousarray(pixels, pixels.dtype.newbyteorder("<")).tobytes())
    return msg


def _compressed(t, frame_id, fmt, data):
    msg = CompressedImage()
    msg.header.stamp = time_msg(t)
    msg.header.frame_id = frame_id
    msg.format = fmt
    msg.data = array.array("B", data)
    return msg


def color_image_msg(t, rgb, frame_id=HEAD_OPTICAL_FRAME):
    """sensor_msgs/Image rgb8 of an (h, w, 3) uint8 image"""
    return _image(t, frame_id, rgb, "rgb8")


def color_compressed_msg(t, rgb, frame_id=HEAD_OPTICAL_FRAME, quality=JPEG_QUALITY):
    """sensor_msgs/CompressedImage JPEG of an (h, w, 3) uint8 rgb image"""
    jpeg = io.BytesIO()
    PILImage.fromarray(rgb).save(jpeg, format="JPEG", quality=quality)
    return _compressed(t, frame_id, "rgb8; jpeg compressed bgr8", jpeg.getvalue())


def depth_image_msg(t, depth_mm, frame_id=HEAD_OPTICAL_FRAME):
    """sensor_msgs/Image 16UC1 of an (h, w) uint16 depth image in mm"""
    return _image(t, frame_id, depth_mm, "16UC1")


def depth_compressed_msg(t, depth_mm, frame_id=HEAD_OPTICAL_FRAME, png_level=PNG_LEVEL):
    """sensor_msgs/CompressedImage compressedDepth of an (h, w) uint16 depth image in mm;
    like the plugin, depths over DEPTH_MAX_MM read 0 (the raw image keeps them)"""
    depth_mm = np.where(depth_mm > DEPTH_MAX_MM, 0, depth_mm).astype(np.uint16)
    png = io.BytesIO()
    PILImage.fromarray(depth_mm).save(png, format="PNG", compress_level=png_level)
    # ConfigHeader: compressionFormat INV_DEPTH (0) and two quantization floats that
    # 16UC1 does not use; all 12 bytes are 0, as in the plugin's own messages
    header = struct.pack("<iff", 0, 0.0, 0.0)
    return _compressed(t, frame_id, "16UC1; compressedDepth", header + png.getvalue())


def camera_info_msg(t, width, height, fx, fy, cx, cy, frame_id=HEAD_OPTICAL_FRAME):
    """sensor_msgs/CameraInfo of an undistorted pinhole camera"""
    msg = CameraInfo()
    msg.header.stamp = time_msg(t)
    msg.header.frame_id = frame_id
    msg.width, msg.height = int(width), int(height)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg
