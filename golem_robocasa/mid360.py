"""Model the Livox MID-360's scans in MuJoCo for both robots: the real scan pattern, cast exactly.

Geometry only (numpy + mujoco, no ROS): each robot's sensor code takes the sensor's pose
from its model, calls Mid360Scanner.update() and packs the LidarFrame it returns into the
messages the Livox driver publishes. The MID-360's IMU is not here: g1/g1_lidar.py (G1) and
h12_sensors.py (H1-2) model it.

Frames. Directions and points are in the sensor's own frame (livox_frame, the frame the
driver publishes in with MID360_config.json's extrinsic at zero); the caller gives the
sensor's pose in the world, origin and rotation (sensor -> world). Both robots hang the
sensor upside down, so that frame's +z points at the floor.

Pattern. third_party/mujoco_lidar/mid360.npy is the MID-360's non-repetitive pattern from
MuJoCo-LiDAR (its README there): 800,000 (azimuth, elevation) pairs in rad, in firing order.
Azimuth is measured from +x towards +y; elevation is positive towards +z, the sensor's own
up, and spans -7 to +52 deg. 500 consecutive points already sweep ~155 deg of azimuth and
20,000, one 10 Hz frame at 200,000 points/s, cover the whole field of view. The scanner
walks the pattern without restarting it at each frame, so consecutive frames fill each
other's gaps as the real sensor's do; the file repeats after 40 frames.

Timing. update() is called from the sensor's loop with the time of the snapshot's
kinematics. Each call casts the rays that came due since the last call, from that call's
pose, and labels them with its time, so offset_time always matches the pose a point was
measured from (FAST_LIO's motion compensation relies on it). Point k of a frame is due at
k / points_per_frame of the frame period. A frame's timebase is its first point's time, as
in the driver (lddc.cpp InitCustomMsg). The caller decides how often to call: every
caster's batch_s of sim time.

Returns. A ray gives a point when its first hit lies between MIN_RANGE (the blind zone)
and MAX_RANGE (the range at 10% reflectivity). The measured range is the true range plus
Gaussian noise of RANGE_NOISE (1 sigma; the MID-360's 2 cm range precision); the blind zone
and the maximum range apply to the true range. Rays without a return give no point.

Self returns. The sensor sits inside a shell of the body it is mounted on, so rays pass
through that body's geoms (MuJoCo's bodyexclude). A ray whose first hit is any other geom
of the robot gives no point, so the scan holds no returns from the robot's own limbs while
the limbs still occlude what is behind them. Only geoms in the caster's geomgroup are hit;
MuJoCo's ray functions also skip geoms with alpha 0 (RoboCasa's placement regions and
hidden collision boxes).

Casters. CpuCaster casts every ray exactly on the CPU; GpuCaster reads the ranges from depth
images the GPU renders around the sensor (EGL), at a fraction of the cost, and follows the same
rules (see its docstring for how close its ranges come). make_caster() picks one by the
backend GOLEM_LIDAR_BACKEND names: the GPU wherever MuJoCo renders on a GPU through EGL, else
the CPU (the Mac).
"""
import atexit
import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import NamedTuple

import mujoco
import numpy as np

POINTS_PER_SECOND = 200_000          # the MID-360's point rate
FRAME_RATE_HZ = 10.0                 # the driver's publish_freq (msg_MID360_launch.py)
LINES = 4                            # kLineNumberMid360: the driver numbers points i % 4
MIN_RANGE = 0.1                      # m, the blind zone
MAX_RANGE = 40.0                     # m, the range at 10% reflectivity
RANGE_NOISE = 0.02                   # m, 1 sigma

PATTERN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "third_party", "mujoco_lidar", "mid360.npy")
# Worker threads of a CpuCaster: a quarter of the cores, at most 4, so the casts leave the
# physics and the cameras their cores.
DEFAULT_THREADS = max(1, min(4, (os.cpu_count() or 1) // 4))
# Rays below which another thread is not used: handing a chunk to a thread costs tens of us,
# the work of about 10 rays in a kitchen.
_MIN_RAYS_PER_THREAD = 250
_EPS_T = 1e-9                        # s, tolerance on accumulated sim time

BACKENDS = ("auto", "gpu", "cpu")    # make_caster's backends, GOLEM_LIDAR_BACKEND's values
# Serialises OpenGL rendering across the process's threads: with two threads rendering in
# their own contexts at once, NVIDIA's driver slows each render several-fold, and the GPU
# caster's casts fall far behind beside the H1-2's cameras. GpuCaster holds it while it
# renders; every other renderer in the process (the sims' cameras) should hold it around its
# update_scene() and render() calls.
GL_LOCK = threading.Lock()
# GL_RENDERER substrings of software rasterisers, on which rendering is slower than the CPU
# caster's exact rays.
_SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "software", "swiftshader")
# The GPU caster's cube map: each face's (forward, up) in the world. The faces are aligned
# with the world axes, so walls and floors aligned with them read back exactly (their depth
# is constant across a pixel). A face's image x runs along forward x up, its y along up.
_FACE_AXES = np.array([[[1, 0, 0], [0, 0, 1]], [[-1, 0, 0], [0, 0, 1]],
                       [[0, 1, 0], [0, 0, 1]], [[0, -1, 0], [0, 0, 1]],
                       [[0, 0, 1], [1, 0, 0]], [[0, 0, -1], [1, 0, 0]]], dtype=np.float64)
_FACE_RIGHT = np.cross(_FACE_AXES[:, 0], _FACE_AXES[:, 1])
# m along a face's axis, the renders' near plane: well inside the blind zone, so geometry
# there still blocks the rays as it does on the CPU caster (the scanner then drops the return).
_GPU_NEAR = 0.01


class LidarFrame(NamedTuple):
    """Hold one published scan, the points in the order they were measured."""
    timebase: float            # s, time of the first ray (the CustomMsg's timebase and stamp)
    offset_time: np.ndarray    # uint32 ns after timebase, non-decreasing
    points: np.ndarray         # float32 (n, 3), m, in the sensor's own frame
    line: np.ndarray           # uint8, 0..LINES-1
    rays: int                  # rays fired in the frame, with a return or not


class Snapshot(NamedTuple):
    """Hold what one cast reads: the sensor's pose, the scene's geom poses and its live alpha."""
    origin: np.ndarray         # (3,) m, the sensor's position in the world
    rotation: np.ndarray       # (3, 3), sensor -> world
    data: mujoco.MjData        # its geom_xpos and geom_xmat are what the rays hit
    # The sim model's geom_rgba[:, 3] and mat_rgba[:, 3] at the snapshot: RoboCasa hides and
    # shows geoms at run time, and rays skip those with alpha 0. None keeps the alpha the
    # caster last cast with (at first its model's).
    geom_alpha: np.ndarray = None
    mat_alpha: np.ndarray = None


def load_pattern():
    """Return the MID-360's scan pattern: (800,000, 2) float64, azimuth and elevation in rad."""
    return np.load(PATTERN_FILE).astype(np.float64)


def directions(pattern, start, count):
    """Return the unit directions (count, 3) of pattern rows start.., wrapping past its end.

    The directions are in the sensor's own frame (see the module docstring).
    """
    rows = pattern[(start + np.arange(count)) % len(pattern)]
    azimuth, elevation = rows[:, 0], rows[:, 1]
    cos_el = np.cos(elevation)
    return np.stack([cos_el * np.cos(azimuth), cos_el * np.sin(azimuth), np.sin(elevation)],
                    axis=-1)


class CpuCaster:
    """Cast rays exactly with mj_multiRay, split over threads.

    Threads. cast() clears model.body_bvhadr for its duration, and mj_step's collision
    reads it: give the caster a model no other thread steps or casts on while it runs (as
    make_caster() does, with a copy of the model), and call cast() from one thread at a
    time. The extra threads cast on scratch MjData of their own, since mj_multiRay
    allocates on the MjData it is given.
    """

    batch_s = 0.01             # s of sim time between the scanner's update() calls

    def __init__(self, model, exclude_body, geomgroup, threads=DEFAULT_THREADS,
                 robot_geoms=None):
        """Cast on model, through the geoms of body id exclude_body (-1: none).

        geomgroup is the 6 flags of the geom groups the rays hit. threads is how many
        threads cast at once, the calling one included. robot_geoms is a bool mask
        (ngeom,) of geoms whose first hit gives no return; by default every geom of
        exclude_body's root subtree (the robot the sensor is on), none with exclude_body -1.
        """
        self.model = model
        self.exclude_body = int(exclude_body)
        self._geomgroup = np.array(geomgroup, dtype=np.uint8).reshape(6, 1)
        if robot_geoms is not None:
            self.robot_geoms = np.asarray(robot_geoms, dtype=bool)
        elif self.exclude_body >= 0:
            root = model.body_rootid[self.exclude_body]
            self.robot_geoms = model.body_rootid[model.geom_bodyid] == root
        else:
            self.robot_geoms = np.zeros(model.ngeom, dtype=bool)
        # The scratch MjData carry geom poses only, so a model with flexes casts on the
        # caller's MjData alone.
        self.threads = 1 if model.nflex else max(1, int(threads))
        self._scratch = [mujoco.MjData(model) for _ in range(self.threads - 1)]
        self._pool = (ThreadPoolExecutor(self.threads - 1, thread_name_prefix="mid360_cast")
                      if self.threads > 1 else None)

    def close(self):
        """Stop the worker threads."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _cast_chunk(self, d, source, origin, world, geomid, dist, lo, hi):
        """Cast rays lo..hi on d, after copying source's geom poses into it unless source is d."""
        if source is not d:
            np.copyto(d.geom_xpos, source.geom_xpos)
            np.copyto(d.geom_xmat, source.geom_xmat)
        mujoco.mj_multiRay(self.model, d, origin, world[lo:hi].reshape(-1, 1), self._geomgroup,
                           1, self.exclude_body, geomid[lo:hi].reshape(-1, 1),
                           dist[lo:hi].reshape(-1, 1), hi - lo, MAX_RANGE)

    def cast(self, origin, rotation, directions, data, geom_alpha=None, mat_alpha=None):
        """Return the distance (m) along each unit direction (n, 3) to its first hit, NaN for none.

        origin (3,) m and rotation (3, 3) sensor -> world are the sensor's pose; directions
        are in its own frame; data holds the geom poses. geom_alpha (ngeom,) and mat_alpha
        (nmat,) are written into the model's geom_rgba and mat_rgba alpha first (Snapshot's
        fields; None leaves them). NaN where nothing is hit within MAX_RANGE or the robot is
        hit first.
        """
        if geom_alpha is not None:
            self.model.geom_rgba[:, 3] = geom_alpha
        if mat_alpha is not None:
            self.model.mat_rgba[:, 3] = mat_alpha
        n = len(directions)
        dist = np.empty(n)
        geomid = np.empty(n, dtype=np.int32)
        if n == 0:
            return dist
        world = np.ascontiguousarray(np.asarray(directions) @ np.asarray(rotation).T,
                                     dtype=np.float64)
        origin = np.array(origin, dtype=np.float64).reshape(3, 1)
        chunks = min(self.threads, max(1, n // _MIN_RAYS_PER_THREAD))
        bounds = np.linspace(0, n, chunks + 1).astype(int)
        # mj_multiRay culls each geom by the azimuth and elevation of its bounding box's 8
        # corners as seen from the sensor, a range that can be narrower than the geom's: a
        # wall's bottom edge is seen lower than its corners, so rays in between pass through
        # the wall (3 to 5% of the rays in a RoboCasa kitchen, up to 1.6 m too far or with no
        # return). MuJoCo skips the culling for bodies without a BVH, so for this call none
        # has one; the result is then mj_ray's, ray for ray. Cleared once around every
        # thread's cast, and restored only once they have all finished.
        bvhadr = self.model.body_bvhadr.copy()
        self.model.body_bvhadr[:] = -1
        futures = []
        try:
            for k, scratch in enumerate(self._scratch[:chunks - 1], start=1):
                futures.append(self._pool.submit(self._cast_chunk, scratch, data, origin, world,
                                                 geomid, dist, bounds[k], bounds[k + 1]))
            self._cast_chunk(data, data, origin, world, geomid, dist, bounds[0], bounds[1])
            for future in futures:
                future.result()
        finally:
            wait(futures)
            self.model.body_bvhadr[:] = bvhadr
        hit = geomid >= 0
        robot = hit & self.robot_geoms[np.where(hit, geomid, 0)]
        dist[~hit | robot] = np.nan
        return dist


class GpuUnavailable(RuntimeError):
    """Say this process cannot render on a GPU: MuJoCo not on EGL, no EGL device, software GL."""


class GpuCaster:
    """Read each ray's range from depth images rendered on the GPU around the sensor (EGL).

    The renders form a cube map: six 90-degree faces of width x width pixels, aligned with the
    world axes, so every direction is covered whatever the sensor's orientation. A cast
    renders the faces its rays fall in side by side into one offscreen image, reads it back
    once, and gives each ray the depth of the pixel its direction falls in, divided by the
    cosine of the ray's angle off the face's axis. Surfaces square to a face's axis read back
    exactly; on surfaces seen obliquely and at silhouettes a ray takes its pixel centre's
    range (tests/robocasa/test_mid360_gpu.py holds the agreement with CpuCaster).

    The renders follow CpuCaster's rules. The caster draws its own copy of the model, in
    which a geom is drawn, opaque, exactly when a ray would hit it: in geomgroup, alpha not 0
    (the material's alpha when it has a material; each cast's live alpha, as CpuCaster), not
    on exclude_body. A
    segmentation render beside the depth tells which geom each pixel shows, so a ray whose
    first hit is the robot gives no return. Range limits are CpuCaster's: no return past
    MAX_RANGE; ranges under MIN_RANGE are returned, for the scanner to drop.

    Threads. The GL context is current only inside cast() and close(), under GL_LOCK, so any
    thread may build the caster and cast, one cast at a time. Call close() to free the GL
    context; the caster otherwise frees it at exit.
    """

    # s of sim time between the scanner's update() calls. A cast costs about one render per
    # face whatever its ray count, so the GPU casts fewer, larger batches than the CPU; this
    # interval keeps the lidar thread a few times faster than real time beside the H1-2's
    # cameras, provided they hold GL_LOCK while they render.
    batch_s = 0.02

    def __init__(self, model, exclude_body, geomgroup, width=512, robot_geoms=None):
        """Render a copy of model, without the geoms of body id exclude_body (-1: none).

        geomgroup is the 6 flags of the geom groups the rays hit; width is each face's size in
        pixels. robot_geoms is a bool mask (ngeom,) of geoms whose first hit gives no return; by
        default every geom of exclude_body's root subtree, none with exclude_body -1. Raises
        GpuUnavailable, naming why, when this process cannot render on a GPU (MuJoCo not on
        EGL, no EGL device, a software renderer), and RuntimeError when the GPU fails to
        render or the renders cannot match the rays (a model with flexes).
        """
        backend = os.environ.get("MUJOCO_GL", "")
        if backend.strip().lower() != "egl":
            raise GpuUnavailable(f"MUJOCO_GL is {backend!r}, not 'egl'")
        if model.nflex:
            raise RuntimeError("the model has flexes, which the renders do not match")
        try:
            from mujoco import egl
            from OpenGL import GL
        except ImportError as exc:
            raise GpuUnavailable(f"no EGL: {exc}") from exc
        self._egl = egl
        self._gl = None
        self._context = None
        self.width = w = int(width)
        self.exclude_body = int(exclude_body)
        self.model = m = copy.copy(model)
        if robot_geoms is not None:
            self.robot_geoms = np.asarray(robot_geoms, dtype=bool)
        elif self.exclude_body >= 0:
            root = m.body_rootid[self.exclude_body]
            self.robot_geoms = m.body_rootid[m.geom_bodyid] == root
        else:
            self.robot_geoms = np.zeros(m.ngeom, dtype=bool)

        # Draw exactly the geoms mj_multiRay hits, opaque: a geom colour other than MuJoCo's
        # default overrides its material's, and geoms with alpha 0 are not drawn. Which are
        # hit depends on the live alpha, so _draw() sets the copy's alpha before each cast.
        group = np.array(geomgroup, dtype=bool).reshape(6)
        self._candidates = (group[np.clip(m.geom_group, 0, 5)]
                            & (m.geom_bodyid != self.exclude_body))
        self._geom_alpha = m.geom_rgba[:, 3].copy()
        self._mat_alpha = m.mat_rgba[:, 3].copy()
        self._drawn = None
        m.geom_rgba[:] = 1.0
        # The faces sit side by side, 3 x 2; no multisampling, which would blend the
        # segmentation colours at silhouettes into other geoms' ids.
        m.vis.global_.offwidth = 3 * w
        m.vis.global_.offheight = 2 * w
        m.vis.quality.offsamples = 0
        # Segmentation renders read no texture: upload each as one texel rather than the
        # kitchen's textures again, most of the render context's GPU memory.
        m.tex_width[:] = 1
        m.tex_height[:] = 1
        self.data = mujoco.MjData(m)

        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = 1          # the geoms' alpha decides what is drawn
        self._option.sitegroup[:] = 0
        self._option.flags[:] = 0
        self._option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1
        self._camera = mujoco.MjvCamera()      # mjv_updateScene needs one; cast() sets the view
        # One slot over the candidates: mjv_updateScene warns once a scene is full, even when
        # nothing more is drawn.
        self._scene = mujoco.MjvScene(m, maxgeom=int(self._candidates.sum()) + 1)
        # Segmentation colours only: no lighting effects, and no back-face culling, since rays
        # hit a face from either side.
        self._scene.flags[:] = 0
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        # The scene lists the drawn geoms in model order, so a segmentation colour (scene
        # index + 1) maps to a geom without reading the scene back; checked once here.
        self._draw(None, None)
        self._update_scene()
        geoms = self._scene.geoms[:self._scene.ngeom]
        objid = np.array([g.objid for g in geoms], dtype=np.intp)
        if (any(g.objtype != mujoco.mjtObj.mjOBJ_GEOM for g in geoms)
                or not np.array_equal(objid, np.flatnonzero(self._drawn))):
            raise RuntimeError("the scene does not draw exactly the geoms the rays hit")

        self._viewports = [mujoco.MjrRect((k % 3) * w, (k // 3) * w, w, w) for k in range(6)]
        self._image = mujoco.MjrRect(0, 0, 3 * w, 2 * w)
        self._rgb = np.empty((2 * w, 3 * w, 3), dtype=np.uint8)   # rows from the bottom
        self._depth = np.empty((2 * w, 3 * w), dtype=np.float32)
        # The depth buffer's reversed-Z values -> m along the face's axis, computed as
        # mujoco.Renderer does (float32 projection coefficients, as glFrustum makes them).
        near, far = np.float32(_GPU_NEAR), np.float32(MAX_RANGE)
        c = -(far + near) / (far - near)
        d = -(np.float32(2) * far * near) / (far - near)
        self._depth_c = float(np.float32(-0.5) * c - np.float32(0.5))
        self._depth_d = float(np.float32(-0.5) * d)

        with GL_LOCK:
            try:
                try:
                    self._gl = egl.GLContext(3 * w, 2 * w)
                    self._gl.make_current()
                    renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
                except Exception as exc:   # EGL's and PyOpenGL's own error types
                    raise GpuUnavailable(f"no EGL device: {exc!r}") from exc
                if any(name in renderer.lower() for name in _SOFTWARE_RENDERERS):
                    raise GpuUnavailable(f"the GL renderer is {renderer}, a software renderer")
                self._context = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
                self._context.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
            except Exception as exc:
                self._free()
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(f"EGL rendering failed: {exc!r}") from exc
            finally:
                self._release()
        self.gl_renderer = renderer
        atexit.register(self.close)

    def _draw(self, geom_alpha, mat_alpha):
        """Draw the geoms the rays hit with this alpha (None: the last); map segments to robot.

        geom_alpha (ngeom,) and mat_alpha (nmat,) are the sim model's alpha (Snapshot's).
        """
        if geom_alpha is not None:
            np.copyto(self._geom_alpha, geom_alpha)
        if mat_alpha is not None:
            np.copyto(self._mat_alpha, mat_alpha)
        m = self.model
        alpha = self._geom_alpha.copy()
        has_material = m.geom_matid >= 0
        alpha[has_material] = self._mat_alpha[m.geom_matid[has_material]]
        drawn = self._candidates & (alpha > 0)
        if self._drawn is not None and np.array_equal(drawn, self._drawn):
            return
        self._drawn = drawn
        m.geom_rgba[:, 3] = drawn
        # Segmentation colour (scene index + 1, 0 where nothing is drawn) -> robot or not.
        self._robot_segment = np.concatenate([[False], self.robot_geoms[drawn]])

    def _update_scene(self):
        """Fill the scene with the drawn geoms at the caster's MjData's poses."""
        mujoco.mjv_updateScene(self.model, self.data, self._option, None, self._camera,
                               mujoco.mjtCatBit.mjCAT_STATIC | mujoco.mjtCatBit.mjCAT_DYNAMIC,
                               self._scene)

    def _release(self):
        """Make no GL context current on this thread, so another thread can take the caster's."""
        egl = self._egl
        if egl.EGL_DISPLAY is not None:
            egl.EGL.eglMakeCurrent(egl.EGL_DISPLAY, egl.EGL.EGL_NO_SURFACE,
                                   egl.EGL.EGL_NO_SURFACE, egl.EGL.EGL_NO_CONTEXT)

    def _free(self):
        """Free the MuJoCo render context and the GL context; call with GL_LOCK held."""
        try:
            if self._context is not None:
                self._gl.make_current()
                self._context.free()
        finally:
            self._context = None
            if self._gl is not None:
                self._gl.free()
                self._gl = None

    def close(self):
        """Free the GL context, what was uploaded to it and the model copy; later casts raise."""
        atexit.unregister(self.close)
        with GL_LOCK:
            self._free()
        self.model = self.data = self._scene = None

    def cast(self, origin, rotation, directions, data, geom_alpha=None, mat_alpha=None):
        """Return the distance (m) along each unit direction (n, 3) to its first hit, NaN for none.

        As CpuCaster.cast(): origin (3,) m and rotation (3, 3) sensor -> world are the sensor's
        pose; directions are in its own frame; data holds the geom poses (geom_xpos,
        geom_xmat); geom_alpha and mat_alpha are the live alpha (None: the last cast's). NaN
        where nothing is drawn within MAX_RANGE or the robot is drawn first.
        """
        if self.model is None:
            raise RuntimeError("cast() after close()")
        self._draw(geom_alpha, mat_alpha)
        n = len(directions)
        dist = np.full(n, np.nan)
        if n == 0:
            return dist
        world = np.asarray(directions, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T
        axis = np.abs(world).argmax(axis=1)
        face = 2 * axis + (world[np.arange(n), axis] < 0)
        faces = np.unique(face)
        np.copyto(self.data.geom_xpos, data.geom_xpos)
        np.copyto(self.data.geom_xmat, data.geom_xmat)
        self._update_scene()
        if self._scene.ngeom != len(self._robot_segment) - 1:
            raise RuntimeError("the scene does not draw exactly the geoms the rays hit")
        with GL_LOCK:
            self._gl.make_current()
            try:
                for camera in self._scene.camera:   # both eyes; mjr_render averages them
                    camera.pos[:] = origin
                    camera.frustum_center = 0.0
                    camera.frustum_width = 0.0       # 0: from the viewport's aspect, square
                    camera.frustum_bottom = -_GPU_NEAR   # 90 deg field of view
                    camera.frustum_top = _GPU_NEAR
                    camera.frustum_near = _GPU_NEAR
                    camera.frustum_far = MAX_RANGE
                    camera.orthographic = 0
                for k in faces:
                    for camera in self._scene.camera:
                        camera.forward[:] = _FACE_AXES[k, 0]
                        camera.up[:] = _FACE_AXES[k, 1]
                    mujoco.mjr_render(self._viewports[k], self._scene, self._context)
                mujoco.mjr_readPixels(self._rgb, self._depth, self._image, self._context)
            finally:
                self._release()
        w = self.width
        for k in faces:
            rows = np.flatnonzero(face == k)
            ray = world[rows]
            along = ray @ _FACE_AXES[k, 0]   # cosine of the ray's angle off the face's axis
            col = np.minimum(((ray @ _FACE_RIGHT[k] / along + 1.0) * 0.5 * w).astype(np.intp),
                             w - 1) + (k % 3) * w
            row = np.minimum(((ray @ _FACE_AXES[k, 1] / along + 1.0) * 0.5 * w).astype(np.intp),
                             w - 1) + (k // 3) * w
            rgb = self._rgb[row, col].astype(np.intp)
            segment = rgb[:, 0] | (rgb[:, 1] << 8) | (rgb[:, 2] << 16)
            depth = self._depth_d / (self._depth[row, col].astype(np.float64) + self._depth_c)
            ranges = depth / along
            ranges[(segment == 0) | self._robot_segment[segment] | (ranges > MAX_RANGE)] = np.nan
            dist[rows] = ranges
        return dist


def _log(text):
    """Print text to the sim's log at once."""
    print(text, flush=True)


def make_caster(model, exclude_body, geomgroup, backend="auto", robot_geoms=None, log=_log):
    """Return the caster backend names, casting on model through body exclude_body.

    backend is one of BACKENDS (GOLEM_LIDAR_BACKEND): auto builds a GpuCaster and falls back to
    a CpuCaster, with one log() line naming why, when it cannot; gpu builds a GpuCaster or
    raises RuntimeError; cpu builds a CpuCaster without trying the GPU. ValueError for any
    other backend. geomgroup and robot_geoms are the casters' own. Either caster casts on its
    own copy of model (the CpuCaster edits the model it casts on, CpuCaster's docstring), so
    the physics may step model while it casts; runtime edits of model reach the casts only
    through the snapshot (poses, alpha).
    """
    if backend not in BACKENDS:
        raise ValueError(f"lidar backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
    if backend != "cpu":
        try:
            return GpuCaster(model, exclude_body, geomgroup, robot_geoms=robot_geoms)
        except Exception as exc:   # under auto, any failure to build falls back to the CPU
            if backend == "gpu":
                raise RuntimeError(f"lidar backend gpu: no GPU caster: {exc}") from exc
            log(f"[mid360] lidar rays cast on the CPU: no GPU caster ({exc})")
    return CpuCaster(copy.copy(model), exclude_body, geomgroup, robot_geoms=robot_geoms)


class Mid360Scanner:
    """Fire the MID-360's pattern through a caster and assemble its 10 Hz frames."""

    def __init__(self, caster, rng_seed=0, points_per_second=POINTS_PER_SECOND,
                 rate_hz=FRAME_RATE_HZ):
        """Scan with caster (its cast() as CpuCaster's); rates are per sim second.

        rng_seed seeds the range noise, so the same seed and calls give the same frames.
        """
        self.caster = caster
        self.pattern = load_pattern()
        self.points_per_frame = int(round(points_per_second / rate_hz))
        self.period = 1.0 / rate_hz
        self._rng_seed = rng_seed
        n = self.points_per_frame
        self._dist = np.empty(n)
        self._when = np.empty(n)
        self._line = (np.arange(n) % LINES).astype(np.uint8)
        self.reset()

    def reset(self):
        """Drop the frame in progress; the next update() starts a frame at the pattern's start.

        The range noise restarts from rng_seed.
        """
        self._start = None
        self._last_t = None
        self._pattern_start = 0
        self._rng = np.random.default_rng(self._rng_seed)

    def _begin(self, t):
        """Start the next frame at sim time t, at the pattern row after the last frame's."""
        self._start = t
        self._timebase = None
        self._next = 0
        self._directions = directions(self.pattern, self._pattern_start, self.points_per_frame)
        self._pattern_start = (self._pattern_start + self.points_per_frame) % len(self.pattern)

    def _cast_until(self, snapshot, t, count):
        """Cast the frame's rays up to index count from snapshot's pose, labelled with time t."""
        first, last = self._next, count
        if last <= first:
            return
        if self._timebase is None:
            self._timebase = t
        self._dist[first:last] = self.caster.cast(
            snapshot.origin, snapshot.rotation, self._directions[first:last], snapshot.data,
            geom_alpha=snapshot.geom_alpha, mat_alpha=snapshot.mat_alpha)
        self._when[first:last] = t
        self._next = last

    def _finish(self):
        """Return the frame's returns, with range noise, as a LidarFrame."""
        with np.errstate(invalid="ignore"):   # NaN, no hit, compares False
            valid = (self._dist >= MIN_RANGE) & (self._dist <= MAX_RANGE)
        measured = self._dist[valid] + self._rng.normal(0.0, RANGE_NOISE, int(valid.sum()))
        offset = np.round((self._when[valid] - self._timebase) * 1e9)
        points = self._directions[valid] * measured[:, None]
        return LidarFrame(self._timebase, offset.astype(np.uint32), points.astype(np.float32),
                          self._line[valid].copy(), self.points_per_frame)

    def update(self, snapshot, t):
        """Cast the rays due by sim time t from snapshot; return the frame t completed, else None.

        t (s) is the time of the snapshot's kinematics. Time going back from the last call's
        (a sim reset, even to within the frame in progress) drops the frame in progress and
        starts again as after reset().
        """
        frame = None
        if self._last_t is None or t < self._last_t - _EPS_T:   # first call, or the sim was reset
            self.reset()
            self._begin(t)
        elif t - self._start >= self.period - _EPS_T:
            self._cast_until(snapshot, t, self.points_per_frame)
            frame = self._finish()
            following = self._start + self.period
            self._begin(following if t - following < self.period - _EPS_T else t)
        due = int(np.floor((t - self._start) / self.period * self.points_per_frame + 1e-6)) + 1
        self._cast_until(snapshot, t, min(due, self.points_per_frame))
        self._last_t = t
        return frame
