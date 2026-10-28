"""The G1's head RealSense D435 in MuJoCo: color, aligned depth and intrinsics.

Rendering only (numpy + mujoco, no ROS): g1_sensor_msgs.py packs the images and
the intrinsics into what the RealSense driver publishes under /realsense/head/
(color/image_raw[/compressed], aligned_depth_to_color/image_raw[/compressedDepth],
color/camera_info), all in head_color_optical_frame.

The camera is g1_robosuite's head_cam: at the URDF's d435_link, looking along its
+x with image up along its +z, with the fovy g1_robosuite gives it: 42.5 deg, the
D435's color stream (Intel's datasheet: 69 x 42 deg, +-1; with square pixels at
16:9 this is 69.3 x 42.5 deg, fx = fy = 925.7 at 1280x720, where real units'
calibrations read about 910-930). The depth sensor's own, wider FOV (87 x 58 deg)
does not show: aligned_depth_to_color is depth reprojected into the color
camera, and here it is so trivially, as both come from the same MuJoCo camera:
a depth pixel is the color pixel's, with the color intrinsics. The images are
16:9 (the H1's head camera profile in cl_realsense's h12_rs_cams.launch.py is
1280x720).

Intrinsics are the pinhole of MuJoCo's projection: square pixels, fy = height /
(2 tan(fovy / 2)), and the principal point at the image centre in the ROS/OpenCV
pixel convention (pixel centres at integers), cx = (width - 1) / 2,
cy = (height - 1) / 2. No distortion.

Depth is 16UC1 in millimetres, which is both what the RealSense driver publishes
on aligned_depth_to_color (depth units 1 mm) and what the H1 sim published. Like
the RealSense it is the distance along the optical axis (z in the optical frame),
not the range along the pixel's ray; that is what MuJoCo's depth buffer holds.
Pixels with no surface (MuJoCo's far plane) read 0, the RealSense's "no depth".

Only visual geoms (group 1) are rendered: the robot's and the kitchen's meshes.
Collision geoms (group 0), robosuite's eef-target markers (group 2) and sites are
hidden. The robot's own hands and arms show, as on the real camera.

Threads. MuJoCo's GL context is thread-affine: the renderer is made, used and
closed on one thread (make_renderer(), or the first render()). A frame takes
three steps, so that the thread stepping the physics only reads MjData:
update_scene() copies what the camera sees of data into the renderer's scene
(MuJoCo's mjv_updateScene, CPU only, no GL), on any thread that may read data;
render_scene() renders that scene on the renderer's thread (GL); depth_mm()
converts the depth, on any thread. render() does all three on the caller's
thread. The scene is the renderer's own, so update_scene() must not run while
render_scene() does; the sim's bridge hands the scene between its two threads.
"""
import mujoco
import numpy as np

CAMERA = "head_cam"
WIDTH, HEIGHT = 1280, 720
GEOM_GROUPS = (0, 1, 0, 0, 0, 0)

# head_color_optical_frame in d435_link: z along the optical axis (d435_link +x),
# x to the image's right (-y), y down (-z). (qx, qy, qz, qw), no translation:
# head_cam sits at d435_link's origin.
OPTICAL_IN_D435_XYZW = (-0.5, 0.5, -0.5, 0.5)


def intrinsics(width, height, fovy_deg):
    """(fx, fy, cx, cy) in pixels of a MuJoCo camera rendered at width x height"""
    fy = height / (2.0 * np.tan(np.radians(fovy_deg) / 2.0))
    return fy, fy, (width - 1) / 2.0, (height - 1) / 2.0


class HeadCamera:
    """Renders head_cam. prefix is the robot's name prefix ('robot0_' in RoboCasa)."""

    def __init__(self, model, prefix="robot0_", width=WIDTH, height=HEIGHT):
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, prefix + CAMERA)
        if self.cam_id < 0:
            raise KeyError(f"camera not found in model: {prefix + CAMERA}")
        self.width, self.height = int(width), int(height)
        self.fovy = float(model.cam_fovy[self.cam_id])
        self.fx, self.fy, self.cx, self.cy = intrinsics(self.width, self.height, self.fovy)
        self.far = model.vis.map.zfar * model.stat.extent   # m, MuJoCo's far plane
        self._renderer = None
        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = GEOM_GROUPS
        self._option.sitegroup[:] = 0

    def make_renderer(self):
        """Make the renderer (if not made yet) on the calling thread, which then
        owns its GL context: render_scene() and close() run there"""
        if self._renderer is not None:
            return
        visual = self.model.vis.global_
        visual.offwidth = max(visual.offwidth, self.width)
        visual.offheight = max(visual.offheight, self.height)
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

    def render(self, data):
        """(rgb uint8 (h, w, 3), depth uint16 (h, w) in mm, 0 = no depth) from data's kinematics"""
        rgb, depth = self.render_raw(data)
        return rgb, self.depth_mm(depth)

    def render_raw(self, data):
        """(rgb uint8 (h, w, 3), depth float32 (h, w) in m as MuJoCo renders it) from
        data's kinematics, as new arrays; on the renderer's thread"""
        self.make_renderer()
        self.update_scene(data)
        return self.render_scene()

    def update_scene(self, data):
        """Copy what the camera sees of data's kinematics into the renderer's scene:
        CPU only, no GL, on any thread that may read data, never while
        render_scene() runs. Needs make_renderer() first."""
        if self._renderer is None:
            raise RuntimeError("update_scene() before make_renderer()")
        self._renderer.update_scene(data, camera=self.cam_id, scene_option=self._option)

    def render_scene(self):
        """(rgb uint8 (h, w, 3), depth float32 (h, w) in m as MuJoCo renders it) of the
        scene update_scene() last filled, as new arrays; on the renderer's thread"""
        r = self._renderer
        r.disable_depth_rendering()
        rgb = r.render()
        r.enable_depth_rendering()
        depth = r.render()
        r.disable_depth_rendering()
        return rgb, depth

    def depth_mm(self, depth):
        """uint16 mm of render_raw()'s depth, 0 where there is no surface (or beyond 65.535 m)"""
        mm = np.round(depth.astype(np.float64) * 1000.0)
        mm[~np.isfinite(depth) | (depth <= 0.0) | (depth >= 0.999 * self.far) | (mm > 65535.0)] = 0.0
        return mm.astype(np.uint16)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
