#!/usr/bin/env python3
"""
Serve custom_ros_messages/srv/GraspGen on `graspgen`: an object cloud in, ranked 6-DOF grasps out.

The GraspGenXSampler (the GPU model) is built once per gripper and cached. The
checkpoints are read from $GRASPGENX_CHECKPOINT_DIR/release/{gen,dis}
(required; the ros compose points it at <package>/weights/graspgen) and the
gripper descriptions from GRASPGENX_ASSETS_DIR (required; the ros compose sets
/opt/graspgenx/assets, where the ros image stages the magpie gripper).

Grasps come back in the request cloud's frame, best-first by confidence, with
the finger-closing axis turned toward +Y and, when the request carries a scene
cloud, the grasps whose gripper sweep volume touches the scene removed.

Runs in the ros container, where GraspGenX is pip-installed.
"""

import inspect
import math
import os
import threading
from collections import namedtuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import trimesh
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.collision_filter import filter_colliding_grasps
from custom_ros_messages.srv import GraspGen

from model_server.model_logging import ModelLogger, declare_logging_params
import cv2

# viser (imported by graspgenx.utils.viser_utils) is an optional,
# visualization-only dependency; without it the live viser scene is skipped
try:
    from graspgenx.utils.viser_utils import (
        create_visualizer, visualize_pointcloud, visualize_x_grasp,
        visualize_bbox, get_color_from_score,
    )
    _VISER_OK = True
except Exception:
    _VISER_OK = False


# gripper planned for when a request's `gripper_name` is empty: the magpie
# description staged in the GraspGenX assets
DEFAULT_GRIPPER = 'magpie'
# fewest object points planned on; golem_skills' base.MIN_GRASP_POINTS matches it
MIN_OBJECT_POINTS = 30

# Collision filtering runs only when the request carries a scene cloud. A grasp
# is dropped when the gripper SWEEP VOLUME (the jaw region,
# _gripper_surface_points; not the bulky visual mesh) comes within
# COLLISION_THRESHOLD_M of a scene point. Scene points within
# OBJECT_EXCLUDE_RADIUS_M of the object cloud are removed first, so the grasped
# object never counts as an obstacle. The scene is randomly downsampled to
# MAX_SCENE_POINTS, and each gripper's surface sampled once with
# NUM_COLLISION_SAMPLES points.
#
# m; larger is stricter. The downsampled scene is sparse, so this is a
# clearance band rather than a penetration test, and a rolled gripper (fingers
# closing vertically) passes only ~2-5 mm from the support surface: 5 mm
# rejects those, leaves only upright grasps and returns none on most calls.
# Raise it toward 5 mm only if grasps clip obstacles.
COLLISION_THRESHOLD_M = 0.002
# m; must be >= COLLISION_THRESHOLD_M: a grasp brings the gripper that close
# to the object's surface, so an object point left in the scene (e.g. a rim just
# outside the SAM mask) would flag every good grasp as colliding with the object
# itself. The margin above it absorbs segmentation slop.
OBJECT_EXCLUDE_RADIUS_M = 0.02
MAX_SCENE_POINTS = 8192
NUM_COLLISION_SAMPLES = 2000
# m. A gripper without coll_mesh.obj gets GraspGenX's ~1 cm dummy box at its
# base, which would make the filter a silent no-op; a collision mesh smaller
# than this counts as none and filtering is skipped with a warning. Real
# grippers reach ~0.11-0.19 m from the base.
DUMMY_MESH_MIN_EXTENT_M = 0.05

# port of the viser web view (http://localhost:VISER_PORT) of the scene, object
# and ranked grasps; the ros service uses host networking, so it is reachable
# from the host
VISER_PORT = 8080

# Every grasp the server returns is appended to one (N, 4, 4) .npy file, always
# (unlike the per-call artifacts, which need enable_logging). It sits at the
# model_server logs root, the parent of the graspgen log directory, so
# clear_logs leaves it; GRASPGEN_GRASPS_FILE overrides the full path.
GRASPGEN_GRASPS_FILENAME = 'generated_grasps.npy'

# per-gripper cache entry: the sampler, its opening width [m], the gripper info
# (sweep volume and collision mesh), and the surface points for the collision
# filter (None disables filtering for that gripper)
_Gripper = namedtuple('_Gripper', 'sampler width info surf_pts')

# run_planner_on_object kwargs, filtered to its installed signature. Tuned for
# the most grasps and variety: the server does not filter by approach
# direction (reachability is the grasp skill's job). Against GraspGenX's stock
# config:
#   * num_grasps / topk_num_grasps 2048 (stock 1024): twice the ranked grasps;
#   * moe_num_yaws 72 (stock 36): a yaw every 5 deg instead of 10;
#   * moe_z_offsets_cm (-4, -2, 0, 2) (stock (-2, 0)): four approach depths;
#   * moe_obb_position_spacing_cm 0.5 (stock 1.0): grasp positions on a 5 mm
#     grid, about 4x the density.
# grasp_threshold -1.0 is no confidence cut. The levers multiply the candidate
# pool, so planning is slower on the GPU; trim them if latency matters more.
PLANNER_KWARGS = dict(
    planner='graspmoe', grasp_threshold=-1.0, num_grasps=2048,
    topk_num_grasps=2048, moe_num_yaws=72, moe_z_offsets_cm=(-4.0, -2.0, 0.0, 2.0),
    moe_outlier_threshold=0.014, moe_outlier_k=20, moe_obb_mode='advanced',
    moe_skip_obb_rule='auto', moe_obb_density='dense-topandside',
    moe_obb_position_spacing_cm=0.5,
)


def mat_to_quat(R):
    """Convert a 3x3 rotation matrix to a quaternion (x, y, z, w) by Shepperd's method."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    return (x, y, z, w)


def _sampler_width(sampler):
    """
    Return the gripper's opening width [m] from the sampler, best-effort; 0.08 when absent.

    The attribute holding it differs between GraspGenX versions, so several are tried.
    """
    for path in ('gripper_info', 'gripper'):
        obj = getattr(sampler, path, None)
        for attr in ('width', 'maximum_aperture', 'max_width'):
            val = getattr(obj, attr, None) if obj is not None else None
            if isinstance(val, (int, float)) and val > 0:
                return float(val)
    cfg = getattr(sampler, 'gripper_config', None)
    if isinstance(cfg, dict):
        for k in ('width', 'maximum_aperture'):
            if isinstance(cfg.get(k), (int, float)) and cfg[k] > 0:
                return float(cfg[k])
    return 0.08


class GraspGenServer(Node):
    """Serve graspgen with GraspGenX."""

    def __init__(self):
        """Load the model config, warm the default gripper, create the publishers and service."""
        super().__init__('graspgen_server')
        log, viz, clear = declare_logging_params(self)
        self.logger = ModelLogger(self, 'graspgen', 'model_server', __file__,
                                  log=log, visualize=viz, clear=clear)

        self._run_planner = run_planner_on_object
        self._GraspGenXSampler = GraspGenXSampler

        ckpt_root = os.environ['GRASPGENX_CHECKPOINT_DIR']
        checkpoints = os.path.join(ckpt_root, 'release')
        self._assets_dir = os.environ['GRASPGENX_ASSETS_DIR']
        self.get_logger().info(f'Loading GraspGenX model cfg from {checkpoints} ...')
        self._model_cfg = load_model_cfg(
            os.path.join(checkpoints, 'gen'), os.path.join(checkpoints, 'dis'), None, None)

        sig = inspect.signature(run_planner_on_object)
        self._kwargs = {k: v for k, v in PLANNER_KWARGS.items() if k in sig.parameters}

        self._grasps_file = os.environ.get(
            'GRASPGEN_GRASPS_FILE',
            os.path.join(os.path.dirname(self.logger.dir), GRASPGEN_GRASPS_FILENAME))
        self._grasps_lock = threading.Lock()
        self.get_logger().info(
            f'accumulating generated grasps -> {self._grasps_file}')

        # gripper_name -> _Gripper
        self._samplers = {}
        try:
            self._get_sampler(DEFAULT_GRIPPER)
        except Exception as e:
            # keep serving: a request naming a valid gripper still works, and
            # pays the load on its first call
            self.get_logger().error(
                f'default gripper {DEFAULT_GRIPPER!r} failed to load ({e}); '
                f'pass a valid gripper_name in the request')

        # The last planned-on object cloud, for RViz beside the grasp skill's
        # 'graspgen_markers' (the skill draws the candidates). Latched, so RViz
        # gets it even when it subscribes after the plan.
        self._cloud_pub = self.create_publisher(
            PointCloud2, 'grasp_cloud',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # the last request's scene cloud, the collision filter's obstacles; latched
        self._scene_pub = self.create_publisher(
            PointCloud2, 'scene_cloud',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # live viser scene, redrawn by every plan; None without viser
        self._vis = None
        if _VISER_OK:
            try:
                self._vis = create_visualizer(port=VISER_PORT)
                self.get_logger().info(
                    f'viser visualizer at http://localhost:{VISER_PORT}')
            except Exception as e:
                self.get_logger().warn(f'viser init failed ({e}); no live grasp viz')
        else:
            self.get_logger().info('viser not installed; no live grasp viz')

        self.create_service(GraspGen, 'graspgen', self._plan_cb,
                            callback_group=ReentrantCallbackGroup())
        self.get_logger().info("graspgen_server ready on 'graspgen'")

    def _get_sampler(self, gripper_name):
        """Return the _Gripper of `gripper_name` (empty: DEFAULT_GRIPPER), built on first use."""
        name = gripper_name or DEFAULT_GRIPPER
        if name not in self._samplers:
            self.get_logger().info(f'Building GraspGenXSampler (gripper={name}) ...')
            sampler = self._GraspGenXSampler(
                self._model_cfg, name, assets_dir=self._assets_dir)
            # gripper geometry for the collision filter and viser, best-effort:
            # a GraspGenX version may expose none
            info = getattr(sampler, 'gripper', None)
            if info is None and hasattr(sampler, 'get_gripper_info'):
                try:
                    info = sampler.get_gripper_info()
                except Exception:
                    info = None
            surf_pts = self._gripper_surface_points(info, name)
            self._samplers[name] = _Gripper(
                sampler, _sampler_width(sampler), info, surf_pts)
        return self._samplers[name]

    def _gripper_surface_points(self, info, name):
        """
        Return (M, 3) gripper-frame surface points for the collision filter, or None.

        Sampled from the gripper's SWEEP-VOLUME box (the jaw region viser
        draws), not the visual mesh, which includes the wide body and mount
        (magpie: 21.8 x 17.3 cm against an 11.2 x 11.6 cm jaw sweep) and would
        reject grasps whose jaws are clear of the scene. Without a sweep volume
        it samples a real collision mesh; None (filtering off for this gripper)
        when there is neither, the mesh is GraspGenX's dummy box, or sampling
        fails.
        """
        sv = getattr(info, 'sweep_volume', None) if info is not None else None
        if sv is not None:
            sv = np.asarray(sv, dtype=np.float64).reshape(-1)
            # sweep_volume = [extents_xyz, offset_xyz]: a box of those extents at
            # that offset, in the gripper-base frame
            if sv.shape[0] >= 6 and np.all(sv[:3] > 0):
                box = trimesh.primitives.Box(extents=sv[:3])
                box.apply_translation(sv[3:6])
                try:
                    sp, _ = trimesh.sample.sample_surface(box, NUM_COLLISION_SAMPLES)
                    return np.asarray(sp, dtype=np.float32)
                except Exception as e:
                    self.get_logger().warn(
                        f'sweep-volume sampling failed for {name!r} ({e})')
        mesh = getattr(info, 'collision_mesh', None) if info is not None else None
        extent = (float(np.ptp(np.asarray(mesh.bounds), axis=0).max())
                  if mesh is not None else 0.0)
        if mesh is None:
            self.get_logger().warn(
                f'gripper {name!r} has no sweep volume or collision mesh; '
                f'collision filtering disabled for it')
            return None
        if extent < DUMMY_MESH_MIN_EXTENT_M:
            self.get_logger().warn(
                f'gripper {name!r} has no sweep volume and only a '
                f'{extent * 100:.1f} cm dummy collision mesh; collision filtering '
                f'disabled for it')
            return None
        try:
            sp, _ = trimesh.sample.sample_surface(mesh, NUM_COLLISION_SAMPLES)
            return np.asarray(sp, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(
                f'gripper surface sampling failed for {name!r} ({e}); '
                f'collision filtering disabled for this gripper')
            return None

    def _plan_cb(self, request, response):
        """
        Answer one GraspGen request with ranked, canonicalized, collision-free grasps.

        Fails (success false, message set) on a malformed or too-small object
        cloud, a gripper that does not load, a planner error, no grasps, or
        every grasp colliding with the scene.
        """
        rec = self.logger.start()
        frame = request.object_cloud.header.frame_id
        rec.set(gripper_name=request.gripper_name or DEFAULT_GRIPPER, frame=frame)
        try:
            pts = point_cloud2.read_points_numpy(
                request.object_cloud, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as e:
            response.success, response.message = False, f'bad cloud: {e}'
            rec.finish(success=False, message=response.message)
            return response
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        rec.set(n_input_points=int(pts.shape[0]))
        rec.save_array('input_cloud', pts)
        self._cloud_pub.publish(request.object_cloud)
        if pts.shape[0] < MIN_OBJECT_POINTS:
            response.success, response.message = False, f'too few points ({pts.shape[0]})'
            rec.finish(success=False, message=response.message)
            return response

        # the optional scene cloud (obstacles, object_cloud's frame); empty or
        # malformed skips the collision filter instead of failing
        scene_pts = None
        if len(request.scene_cloud.data) > 0:
            try:
                sp = point_cloud2.read_points_numpy(
                    request.scene_cloud, field_names=('x', 'y', 'z'), skip_nans=True)
                scene_pts = np.asarray(sp, dtype=np.float32).reshape(-1, 3)
                rec.set(n_scene_points=int(scene_pts.shape[0]))
                rec.save_array('scene_cloud', scene_pts)
                self._scene_pub.publish(request.scene_cloud)
                self.get_logger().info(
                    f'scene cloud: {scene_pts.shape[0]} points (frame={frame})')
            except Exception as e:
                self.get_logger().warn(
                    f'scene_cloud decode failed ({e}); skipping collision filter')
                scene_pts = None

        try:
            gr = self._get_sampler(request.gripper_name)
        except Exception as e:
            response.success, response.message = False, f'sampler load failed: {e}'
            self.get_logger().error(response.message)
            rec.finish(success=False, message=response.message)
            return response
        width = gr.width

        # GraspGenX centers the cloud itself and returns grasps in the input
        # frame, so the raw cloud goes in and the grasps come back in `frame`
        try:
            out = self._run_planner(pts, gr.sampler, **self._kwargs)
        except Exception as e:
            response.success, response.message = False, f'planner failed: {e}'
            self.get_logger().error(response.message)
            rec.finish(success=False, message=response.message)
            return response

        grasps = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
        # before any len() or indexing: a 0-d return must fail the request, not
        # raise in the service thread (ndim is safe on 0-d)
        if grasps.ndim != 3 or grasps.shape[0] == 0:
            response.success, response.message = False, 'no grasps produced'
            rec.finish(success=False, message=response.message)
            return response
        n = grasps.shape[0]
        confs = (np.asarray(out[1]).reshape(-1)
                 if isinstance(out, (tuple, list)) and len(out) > 1 else np.ones(n))
        # scores that do not match the grasps are replaced, not used to drop grasps
        if confs.shape[0] != n:
            confs = np.ones(n)
        # out[3], the object's bounding box, is drawn in viser only; may be absent
        obb_dict = out[3] if isinstance(out, (tuple, list)) and len(out) > 3 else None

        # The planner returns its grasps unsorted; the response is ranked
        # best-first by confidence (the filter below keeps the order).
        order = np.argsort(-confs)
        grasps, confs = grasps[order], confs[order]

        rec.set(arm=(request.arm or '').strip().lower() or 'unknown')

        # Canonicalize the finger-closing axis toward +Y (robot-left in the
        # pelvis frame): grasps[:, 1, 0] is the gripper +X axis's Y component.
        # A 180 deg turn about the approach (+Z) negates local X and Y and keeps
        # the approach and origin: the fingers swap, the same physical grasp.
        needs_flip = grasps[:, 1, 0] < 0.0
        T_flip = np.eye(4)
        T_flip[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        grasps[needs_flip] = grasps[needs_flip] @ T_flip
        self.get_logger().info(
            f'flipped: {int(needs_flip.sum())} grasps with +X toward -Y '
            f'(cloud frame +Y left)')

        # grasps and scene_pts are both in `frame`
        if scene_pts is not None and len(scene_pts) and gr.surf_pts is not None:
            grasps_world = grasps.astype(np.float64)
            scene_obstacles = self._scene_minus_object(scene_pts, pts)
            rec.save_array('scene_obstacles', scene_obstacles)
            rec.save_array('grasps_world', grasps_world)
            free = filter_colliding_grasps(
                scene_pc=scene_obstacles, grasp_poses=grasps_world,
                gripper_surface_points=gr.surf_pts,
                collision_threshold=COLLISION_THRESHOLD_M)
            kept, kept_confs = grasps[free], confs[free]
            n_before = len(grasps)
            rec.set(n_scene_obstacle_points=int(len(scene_obstacles)),
                    n_collision_free=int(len(kept)),
                    n_collision=int(n_before - len(kept)))
            self.get_logger().info(
                f'collision filter: {len(kept)}/{n_before} free '
                f'(thr={COLLISION_THRESHOLD_M}m, scene_pts={len(scene_obstacles)})')
            if len(kept) == 0:
                # draw the rejected grasps against the filter's obstacles, so
                # viser and the PNG show why every grasp collided
                self._render_viser(pts, scene_obstacles, grasps, confs,
                                   gr.info, obb_dict)
                self._save_grasp_viz(rec, pts, grasps, width)
                response.success, response.message = False, \
                    'all grasps in collision with scene'
                rec.finish(success=False, message=response.message)
                return response
            grasps, confs = kept, kept_confs

        for g, c in zip(grasps, confs):
            g = np.asarray(g, dtype=np.float64).reshape(4, 4)
            qx, qy, qz, qw = mat_to_quat(g[:3, :3])
            ps = PoseStamped()
            ps.header.frame_id = frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(g[0, 3]), float(g[1, 3]), float(g[2, 3]))
            ps.pose.orientation.x = float(qx)
            ps.pose.orientation.y = float(qy)
            ps.pose.orientation.z = float(qz)
            ps.pose.orientation.w = float(qw)
            response.grasps.append(ps)
            response.scores.append(float(c))

        response.gripper_width = float(width)
        response.success = True
        response.message = f'{len(response.grasps)} grasps'
        # confs are sorted best-first
        best = float(confs[0]) if len(confs) else 0.0
        worst = float(confs[-1]) if len(confs) else 0.0
        rec.set(n_grasps=len(response.grasps), gripper_width=float(width),
                score_best=round(best, 3), score_worst=round(worst, 3))
        self.get_logger().info(
            f"grasp metrics: {len(response.grasps)} grasps "
            f"(gripper={request.gripper_name or DEFAULT_GRIPPER}, "
            f"score best={best:.3f} worst={worst:.3f}, width={width * 1000:.0f}mm, "
            f"object_pts={pts.shape[0]}, "
            f"scene_pts={0 if scene_pts is None else len(scene_pts)})")
        # (M, 4, 4) in the cloud frame
        rec.save_array('grasps', grasps)
        rec.save_array('scores', confs)
        self._append_generated_grasps(grasps)
        self._save_grasp_viz(rec, pts, grasps, width)
        self._render_viser(pts, scene_pts, grasps, confs, gr.info, obb_dict)
        rec.finish(success=True, message=response.message)
        return response

    def _scene_minus_object(self, scene_pts, object_pts):
        """
        Return the obstacle cloud: the scene without the object, at most MAX_SCENE_POINTS.

        Scene points within OBJECT_EXCLUDE_RADIUS_M of an object point are
        removed, and the rest randomly downsampled.
        """
        scene_pts = np.asarray(scene_pts, dtype=np.float32)
        if len(object_pts):
            from scipy.spatial import cKDTree
            d, _ = cKDTree(np.asarray(object_pts, dtype=np.float32)).query(scene_pts, k=1)
            scene_pts = scene_pts[d > OBJECT_EXCLUDE_RADIUS_M]
        if len(scene_pts) > MAX_SCENE_POINTS:
            idx = np.random.choice(len(scene_pts), MAX_SCENE_POINTS, replace=False)
            scene_pts = scene_pts[idx]
        return scene_pts.astype(np.float32)

    def _append_generated_grasps(self, grasps):
        """
        Append (M, 4, 4) grasp poses to the cumulative grasps file, best-effort.

        Load, concatenate and save under a lock, so concurrent service calls do
        not overwrite each other, through a temporary file and os.replace, so a
        crash mid-write cannot corrupt the file. A failure is logged and never
        fails the request.
        """
        try:
            new = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
            if new.shape[0] == 0:
                return
            with self._grasps_lock:
                allg = new
                if os.path.exists(self._grasps_file):
                    try:
                        existing = np.load(self._grasps_file).reshape(-1, 4, 4)
                        allg = np.concatenate([existing, new], axis=0)
                    except Exception as e:
                        self.get_logger().warn(
                            f'could not read {self._grasps_file} ({e}); recreating')
                os.makedirs(os.path.dirname(self._grasps_file), exist_ok=True)
                tmp = self._grasps_file + '.tmp'
                with open(tmp, 'wb') as f:
                    np.save(f, allg)
                os.replace(tmp, self._grasps_file)
            self.get_logger().info(
                f'appended {new.shape[0]} grasps -> {self._grasps_file} '
                f'({allg.shape[0]} total)')
        except Exception as e:
            self.get_logger().warn(f'grasp accumulation failed: {e}')

    def _render_viser(self, object_pts, scene_pts, grasps, confs,
                      gripper_info, obb_dict=None):
        """
        Redraw the live viser scene; do nothing without viser.

        Gray scene cloud, blue object cloud, the bounding-box wireframe, and the
        ranked grasps (best thick blue, the rest colored by score), all in the
        cloud frame. Drawing errors are logged, never raised.
        """
        if self._vis is None:
            return
        try:
            vis = self._vis
            vis.scene.reset()
            if scene_pts is not None and len(scene_pts):
                visualize_pointcloud(vis, 'scene', scene_pts,
                                     color=[150, 150, 150], size=0.0025)
            visualize_pointcloud(vis, 'object', np.asarray(object_pts),
                                 color=[0, 150, 255], size=0.004)
            if not len(grasps):
                return
            G = np.asarray(grasps, dtype=np.float64)
            # guarded on its own: the installed viser's add_box() lacks the
            # `wireframe` argument visualize_bbox passes, and an unguarded error
            # here would stop the grasps from being drawn
            if obb_dict is not None:
                try:
                    T = np.eye(4)
                    T[:3, :3] = obb_dict['R']
                    T[:3, 3] = np.asarray(obb_dict['center'])
                    visualize_bbox(vis, 'obb',
                                   2.0 * np.asarray(obb_dict['half_extent']),
                                   T=T, color=[255, 130, 0])
                except Exception as e:
                    self.get_logger().warn(f'viser OBB draw skipped: {e}')
            colors = get_color_from_score(np.asarray(confs, dtype=np.float32),
                                          use_255_scale=True)
            colors = np.atleast_2d(colors)
            for i in range(len(G)):
                # grasps are sorted best-first
                best = (i == 0)
                color = [0, 100, 255] if best else colors[i]
                visualize_x_grasp(vis, f'grasps/g{i:03d}', G[i], color=color,
                                  gripper_info=gripper_info,
                                  linewidth=5.0 if best else 1.5)
        except Exception as e:
            self.get_logger().warn(f'viser render failed: {e}')

    def _save_grasp_viz(self, rec, pts, grasps, width):
        """
        Save orthographic front (Y-Z) and side (X-Z) views of the cloud and the top grasps.

        Drawn with OpenCV, because the ros image's matplotlib 3-D backend does
        not work. Each of the top 5 grasps is drawn as its finger-opening axis
        (red) and approach axis (blue). Only with enable_visualization on.
        """
        if not self.logger.visualize:
            return

        try:
            G = np.asarray([np.asarray(g, dtype=np.float64).reshape(4, 4) for g in grasps])
            centers = G[:, :3, 3]
            # GraspGenX pose: +Z approaches the object, +X is the finger-closing axis
            topk = min(5, len(G))
            half = max(0.01, float(width) / 2.0)
            # m, drawn length of the approach axis
            approach_len = 0.06
            # (point a, point b, BGR color)
            segments = []
            for i in range(topk):
                R, p = G[i, :3, :3], centers[i]
                segments.append((p - half * R[:, 0], p + half * R[:, 0], (0, 0, 255)))
                segments.append((p - approach_len * R[:, 2], p, (255, 0, 0)))
            panels = [
                self._render_grasp_panel(cv2, pts, centers, segments, ai, bi,
                                         f'{label}  (top {topk}/{len(G)})')
                for (ai, bi), label in (((1, 2), 'front Y-Z'), ((0, 2), 'side X-Z'))
            ]
            cv2.imwrite(rec.path('grasps', 'png'), np.hstack(panels))
        except Exception as e:
            self.get_logger().warn(f'grasp viz failed: {e}')

    @staticmethod
    def _render_grasp_panel(cv2, pts, centers, segments, ai, bi, label,
                            size=520, pad=36):
        """
        Draw one orthographic view: axis `ai` horizontal, `bi` vertical (up is up).

        Gray cloud dots and the colored grasp segments on a white `size` px
        canvas, at a uniform scale that fits the cloud and grasp centers.
        """
        view = np.vstack([pts[:, [ai, bi]], centers[:, [ai, bi]]])
        mins, maxs = view.min(axis=0), view.max(axis=0)
        c = (mins + maxs) / 2.0
        scale = (size - 2 * pad) / (float((maxs - mins).max()) or 0.1)

        def to_px(h, v):
            """Return the pixel of view coordinates (h, v); image y grows downward."""
            return (int((h - c[0]) * scale + size / 2.0),
                    int(size / 2.0 - (v - c[1]) * scale))

        img = np.full((size, size, 3), 255, np.uint8)
        for h, v in pts[:, [ai, bi]]:
            cv2.circle(img, to_px(h, v), 1, (170, 170, 170), -1)
        for a, b, color in segments:
            cv2.line(img, to_px(a[ai], a[bi]), to_px(b[ai], b[bi]), color, 2)
        cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        return img


def main():
    """Spin a GraspGenServer on a MultiThreadedExecutor until interrupted."""
    rclpy.init()
    node = GraspGenServer()
    # multithreaded, with the service in a reentrant group, so a long GPU plan
    # does not block discovery, the latched publishers or a second request
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
