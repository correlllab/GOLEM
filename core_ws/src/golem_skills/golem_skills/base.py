#!/usr/bin/env python3
"""
Hold the machinery every skill shares: clients, camera caches, perception, motion.

SkillsBase is the node every skill mixin plugs into (SkillsNode in node.py
inherits from it and from every mixin under skills/). It owns:

  * the clients: /frame_task and /named_config (arm IK), the per-arm gripper
    services, gemini_query, sam_segment and graspgen, and the node's own
    /skill/grasp (for pick_place);
  * the head- and hand-camera caches (color, aligned depth, intrinsics, YOLO
    bundles) and TF;
  * the executor-safe call plumbing (_wait_future, _call_service,
    _send_action) and the per-goal context _Run;
  * the perception pipeline (query_gemini, segment, detect_object_cloud,
    mask_to_cloud, plan_grasp) and the motion primitives (move_frame_to,
    servo_frame_to_world, goto_named_config, the gripper calls);
  * the arm halt on /safety/heartbeat (arm_halt_reason).

Skill deadlines, arm-move waits, the grip-hold period and the camera waits count robot time
(robot_now): the node's clock, the sim's /clock, under use_sim_time, and the steady clock
otherwise. The heartbeat's silence counts the steady clock always, and so do the waits on
services, the action accepts and the cancels: host and link latency, not robot time.
"""

import threading
import time

import numpy as np

from rclpy.action import ActionClient, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3Stamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Bool, Header

from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformBroadcaster, TransformListener, TransformException

from custom_ros_messages.msg import DetectionBundle
from custom_ros_messages.srv import GeminiQuery, SamSegment, GraspGen
from custom_ros_messages.action import FrameTask, NamedConfig
from custom_ros_messages.action import (
    SkillCloseDoor, SkillOpenDoor, SkillCloseLid, SkillOpenLid,
    SkillNavigate, SkillGrasp, SkillPickPlace, SkillPress, SkillSlideRack,
    SkillTurnLever, SkillTwistKnob, SkillFrontierExplore,
)
from magpie_msgs.srv import SetGripperPosition, SetGripperForce
from magpie_msgs.msg import GripperState
from std_srvs.srv import Trigger

from .perception_utils import (
    decode_compressed_depth_image, deproject_mask, transform_points,
    transform_to_matrix, pose_to_matrix, matrix_to_pose, quat_geodesic,
    extract_json,
)
from .grasp_synthesis import DEFAULT_METHOD, METHODS, method_name
from model_server.model_logging import _log_root
from .run_telemetry import RunRecorder, describe_goal, run_dir


CAMERA_NS = '/realsense/head'
# color (JPEG), depth aligned to the color grid (compressedDepth, uint16 mm), intrinsics
COLOR_IMAGE_TOPIC = f'{CAMERA_NS}/color/image_raw/compressed'
DEPTH_IMAGE_TOPIC = f'{CAMERA_NS}/aligned_depth_to_color/image_raw/compressedDepth'
CAMERA_INFO_TOPIC = f'{CAMERA_NS}/color/camera_info'
# yolo_server's DetectionBundle for the head color stream ('<image_topic>/detections');
# its boxes are on the color grid, so they index the aligned depth cache directly
DETECTION_TOPIC = f'{COLOR_IMAGE_TOPIC}/detections'
# yolo_server's DetectionBundle per hand camera; the visual servo reads the
# grasping arm's bundle, which carries that camera's aligned depth and intrinsics
ARM_DETECTION_TOPICS = {
    'left':  '/realsense/left_hand/color/image_raw/compressed/detections',
    'right': '/realsense/right_hand/color/image_raw/compressed/detections',
}
# The hand cameras' color streams, cached directly so the grasp skill's
# approach snapshot has a frame even when yolo_server's hand channels are down.
ARM_COLOR_IMAGE_TOPICS = {
    'left':  '/realsense/left_hand/color/image_raw/compressed',
    'right': '/realsense/right_hand/color/image_raw/compressed',
}

# URDF gripper-BASE frames whose axes match GraspGenX's grasp convention (+Z
# approach, +X finger closing, origin at the magpie gripper base), so a grasp
# pose drives them as given, with no axis permutation or TCP offset.
GRASP_FRAMES = {'left': 'left_graspgenx_frame', 'right': 'right_graspgenx_frame'}
# per-arm gripper namespace: the magpie driver and the sims serve <ns>/set_position,
# <ns>/set_force, <ns>/open, <ns>/close and <ns>/reset_overload, and publish <ns>/state
GRIPPER_NS = {'left': '/left/gripper', 'right': '/right/gripper'}

# N, the grip-force limit set before every close. More than a small part needs
# drives the Dynamixels into the overload the grip-hold loop has to keep clearing.
GRIP_FORCE_N = 15.0
# Hz of the grip-hold loop (close_gripper, _grip_hold_loop). The magpie fingers
# are Dynamixels that trip into OVERLOAD while holding against an object and
# then go limp, dropping it; reset_overload clears that without opening the
# fingers, so a thread per arm calls reset_overload then close every tick. The
# loop prevents those gripper overload crashes. It outlives the skill that
# started it (the object is still held after /skill/grasp returns) and stops
# only when the gripper is commanded to an aperture again (open_gripper,
# set_gripper) or the node exits (daemon thread).
GRIP_HOLD_RATE_HZ = 5.0
# s per service call inside the loop: well under the 1/rate period so a stalled
# gripper service cannot drag the loop off cadence; a timed-out tick is retried
GRIP_HOLD_CALL_TIMEOUT_SEC = 0.15
# A dead gripper service fails two calls per tick; warn once per this many
# consecutive failed ticks. Failures are never fatal: a flaky gripper service
# must not turn a good grasp into a failed goal.
GRIP_HOLD_WARN_EVERY = 25
# s of ROS time (sim time in the sim) waited after an open or close before the
# arm moves: the /close Trigger returns before the fingers have finished
# travelling, so without it the arm moves while the gripper is still moving
GRIPPER_SETTLE_SEC = 1.0
# mm, full-open aperture: the magpie driver clamps set_position to 0-106 mm, so
# open_gripper maps 0..1 onto 0..106 mm
GRIPPER_MAX_WIDTH_MM = 106.0
# s, the skill deadline when goal.timeout is zero, for the skills on _Run
# (grasp, pick_place; frontier_explore has its `default_timeout` parameter).
# Infinite: a goal with no timeout runs until it finishes, fails or is
# canceled, because a long skill (a visual-servo grasp, a pick_place carrying
# one) has no meaningful upper bound. With it, _Run.remaining() is inf, every
# caller that spends it bounds it with min(finite, ...), and pick_place sends
# its inner grasp a zero Duration. Send a non-zero goal.timeout for a bound.
DEFAULT_SKILL_TIMEOUT = float('inf')
# s, the gemini_query call timeout: the grasp-detection model can take minutes
# to answer, far beyond _call_service's 30 s default
GEMINI_TIMEOUT_SEC = 240.0

# A box that FULLY ENCOMPASSES the object, because it prompts SAM, whose mask
# must cover the whole object for the cloud that GraspGenX planning and the
# place-target localization use. box_2d is [y1, x1, y2, x2] normalized to 0-1000.
GEMINI_GRASP_PROMPT = (
    'Locate the {obj} so a segmentation model can be prompted with a bounding box. '
    'Return a JSON array with exactly one box that FULLY ENCOMPASSES the entire '
    '{obj} — every visible part of it (including any handle, spout, lid, or other '
    'protrusion) must fall inside the box, with the edges snug to the object\'s '
    'outermost extent. Do not box just one part and do not crop the object: '
    '[{{"box_2d": [y1, x1, y2, x2], "label": "{obj}", "score": 0.9}}], integer '
    'coordinates normalized to 0-1000 with y first. Always return a 4-number '
    '"box_2d" bounding box — never a single point. '
    'If the {obj} is not visible, return [].'
)
# m, the depth range back-projected into clouds
DEPTH_MIN_M = 0.1
DEPTH_MAX_M = 3.0
# Fewest object points worth planning on. Keep equal to
# graspgen_server.MIN_OBJECT_POINTS so a cloud the server would refuse is
# dropped here with an accurate message.
MIN_GRASP_POINTS = 30

# s; debug TF frames are re-broadcast at this period so they stay visible in RViz
TF_REPUBLISH_PERIOD_SEC = 0.1

# World frame of the grasp servo (servo_frame_to_world): FAST-LIO's odometry
# origin (camera_init -> body -> pelvis). The grasp is anchored here and
# re-resolved into the live pelvis frame each iteration, cancelling pelvis
# drift. 'map' (slam_toolbox) would be globally consistent but jumps on loop
# closure.
WORLD_FRAME = 'camera_init'
# m: once the measured world error exceeds the best so far by this much, the
# drift fixes are moving the base rather than the hand (arm blocked by contact,
# the balance controller stepping away from the reach) and only chase the target
SERVO_DIVERGE_M = 0.01
# A FrameTask duration is a timeout, and slow_mode scales joint velocity by the
# frame_task server's SLOW_MODE_SCALE (0.25), so a slow move needs its budget
# multiplied by 1 / SLOW_MODE_SCALE to reach as far as a full-speed one.
SLOW_MODE_TIME_SCALE = 4.0

# Early stop of a frame_task move (_send_frame_move's stable_stop_msgs): a
# feedback counts as unchanged when both errors stay within these bands of the
# running reference. Above the arm's steady-state jitter (~0.1 mm, <1 mrad), so
# a settled hold is recognized, and well below real motion.
FRAME_TASK_STABLE_EPS_LIN = 0.0005    # m
FRAME_TASK_STABLE_EPS_ANG = 0.002     # rad

# s of steady time: the longest a robot-time wait goes without checking its stop event
ROBOT_WAIT_POLL_SEC = 0.01
# s of steady time; default of the `heartbeat_timeout` parameter: the longest
# /safety/heartbeat (10 Hz) may stay silent before arm motion halts
HEARTBEAT_TIMEOUT_SEC = 0.5

# Skills whose runs get a run_telemetry directory: the implemented manipulation
# skills that move the base. pick_place's in-process grasp shares the outer
# pick_place recorder.
RECORDED_SKILLS = frozenset({'grasp', 'pick_place'})

# skill name -> (action type, action server name); SkillsNode serves every
# entry and the grasp client below uses the grasp entry
SKILL_ACTIONS = {
    'close_door':  (SkillCloseDoor,  '/skill/close_door'),
    'open_door':   (SkillOpenDoor,   '/skill/open_door'),
    'close_lid':   (SkillCloseLid,   '/skill/close_lid'),
    'open_lid':    (SkillOpenLid,    '/skill/open_lid'),
    'navigate':    (SkillNavigate,   '/skill/navigate'),
    'grasp':       (SkillGrasp,      '/skill/grasp'),
    'pick_place':  (SkillPickPlace,  '/skill/pick_place'),
    'press':       (SkillPress,      '/skill/press'),
    'slide_rack':  (SkillSlideRack,  '/skill/slide_rack'),
    'turn_lever':  (SkillTurnLever,  '/skill/turn_lever'),
    'twist_knob':  (SkillTwistKnob,  '/skill/twist_knob'),
    'frontier_explore': (SkillFrontierExplore, '/skill/frontier_explore'),
}


class _Run:
    """
    Track one skill goal: feedback, result, cancel and its deadline in robot time.

    In an execute callback:

        run = _Run(self, goal_handle, SkillOpenDoor, 'open_door')
        if not run.phase('detect', 0.0):
            return run.result            # canceled or timed out
        ...
        return run.succeed('door opened')   # or run.abort('reason')
    """

    def __init__(self, node, goal_handle, action_type, label):
        """Start the goal's deadline and clear the previous goal's arm halt."""
        self._node = node
        self._gh = goal_handle
        self._label = label
        self.feedback = action_type.Feedback()
        self.result = action_type.Result()
        self.deadline = node.robot_now() + node._timeout_sec(goal_handle.request)
        # an aborted arm move halts the arms for the goal that saw it only
        node._arm_halt_reason = None

    def remaining(self):
        """Return the s of robot time left before the deadline (0 when past; inf without one)."""
        return max(0.0, self.deadline - self._node.robot_now())

    def _finish_canceled(self):
        """End the goal as canceled and return its result."""
        self.result.success = False
        self.result.message = 'canceled'
        self._gh.canceled()
        self._node.get_logger().info(f'[{self._label}] canceled')
        return self.result

    def phase(self, name, progress):
        """
        Publish feedback phase `name` at `progress`; return False if the goal has ended.

        A requested cancel ends the goal as canceled, a passed deadline aborts
        it with 'skill timeout'; either way self.result is filled in.
        """
        if self._gh.is_cancel_requested:
            self._finish_canceled()
            return False
        if self._node.robot_now() > self.deadline:
            self.result.success = False
            self.result.message = 'skill timeout'
            self._gh.abort()
            self._node.get_logger().error(f'[{self._label}] timed out')
            return False
        self.feedback.phase = name
        self.feedback.progress = float(progress)
        self._gh.publish_feedback(self.feedback)
        self._node.get_logger().info(f'[{self._label}] phase={name}')
        return True

    def abort(self, message):
        """Abort the goal with `message`, or end it as canceled if a cancel was requested."""
        # a step that failed because the goal was canceled mid-motion ends as canceled
        if self._gh.is_cancel_requested:
            return self._finish_canceled()
        self.result.success = False
        self.result.message = message
        self._node.get_logger().error(f'[{self._label}] {message}')
        self._gh.abort()
        return self.result

    def succeed(self, message='ok'):
        """Succeed the goal with `message`, or end it as canceled if a cancel was requested."""
        if self._gh.is_cancel_requested:
            return self._finish_canceled()
        self.feedback.phase = 'done'
        self.feedback.progress = 1.0
        self._gh.publish_feedback(self.feedback)
        self.result.success = True
        self.result.message = message
        self._node.get_logger().info(f'[{self._label}] done: {message}')
        self._gh.succeed()
        return self.result


class SkillsBase(Node):
    """Hold the clients, camera caches, call plumbing, perception and motion every skill uses."""

    def __init__(self, node_name='golem_skills'):
        """Create the clients, caches, subscriptions and parameters every skill uses."""
        super().__init__(node_name)

        # One reentrant group for everything, so a skill's execute callback can
        # block on inner futures while the executor's other threads complete
        # them, and pick_place can call this node's own /skill/grasp.
        self._cb_group = ReentrantCallbackGroup()

        self.gripper_clis = {
            arm: self.create_client(SetGripperPosition, f'{ns}/set_position',
                                    callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        self.gripper_open_clis = {
            arm: self.create_client(Trigger, f'{ns}/open', callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        self.gripper_close_clis = {
            arm: self.create_client(Trigger, f'{ns}/close', callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        self.gripper_force_clis = {
            arm: self.create_client(SetGripperForce, f'{ns}/set_force',
                                    callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        # clears a Dynamixel overload without opening the fingers; the grip-hold
        # loop's other call
        self.gripper_reset_clis = {
            arm: self.create_client(Trigger, f'{ns}/reset_overload',
                                    callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        # arm -> (thread, stop Event) of the running grip-hold loop
        self._grip_hold = {}
        # mm, the driver's published aperture per arm; tracks every close,
        # including the force-based /close, so it shows whether an object is held
        self._gripper_state_mm = {}
        self.gripper_state_subs = {
            arm: self.create_subscription(
                GripperState, f'{ns}/state',
                lambda msg, a=arm: self._on_gripper_state(a, msg),
                10, callback_group=self._cb_group)
            for arm, ns in GRIPPER_NS.items()
        }
        # mm, the aperture the driver reported after the last set_gripper per
        # arm; gripper_aperture's fallback before any state message
        self._gripper_actual_mm = {}

        # gemini_query: image (+ text) -> text; sam_segment: image + text/boxes ->
        # mono8 mask; graspgen: object cloud -> ranked 6-DOF grasps
        self.gemini_cli = self.create_client(
            GeminiQuery, 'gemini_query', callback_group=self._cb_group)
        self.sam_cli = self.create_client(
            SamSegment, 'sam_segment', callback_group=self._cb_group)
        self.graspgen_cli = self.create_client(
            GraspGen, 'graspgen', callback_group=self._cb_group)

        self.frame_task_cli = ActionClient(
            self, FrameTask, '/frame_task', callback_group=self._cb_group)
        # named arm configurations (prep, home, ...), served by frame_task_server too
        self.named_config_cli = ActionClient(
            self, NamedConfig, '/named_config', callback_group=self._cb_group)

        # SAFETY: arm motion halts on golem_safety's /safety/heartbeat. Nothing
        # beats until golem_safety can relay, then True at 10 Hz, then False
        # once its estop latches (until restart). No arm goal is sent before the
        # first true beat, after a false one (for the node's life), or while no
        # beat has come for `heartbeat_timeout`, and a goal in flight is
        # canceled on the server; a frame_task or named_config goal the server
        # ABORTS (estop, stale robot state, execution error) halts the arms for
        # the rest of the skill goal. See arm_halt_reason().
        self.declare_parameter('heartbeat_timeout', HEARTBEAT_TIMEOUT_SEC)
        # SAFETY: the steady clock, whatever use_sim_time says, so a golem_safety that has
        # stopped halts the arms while a sim's /clock stands still; a fake in the tests
        self._monotonic = time.monotonic
        # robot time is the node's clock (the sim's /clock) under use_sim_time
        self._sim_time = bool(self.get_parameter('use_sim_time').value)
        # self._monotonic() at the last beat
        self._last_beat = None
        self._safety_latched = False
        self._arm_halt_reason = None
        self.safety_heartbeat_sub = self.create_subscription(
            Bool, '/safety/heartbeat', self._on_safety_heartbeat, 10,
            callback_group=self._cb_group)

        # pick_place runs the grasp through this node's own /skill/grasp server
        # (safe under the reentrant group and multithreaded executor). The
        # server comes up with this node, so there is no wait_for_server. The
        # SkillGrasp result carries only success and message, so the grasp
        # hands its GraspOutcome to the caller on the instance (skills/grasp.py).
        self.grasp_skill_cli = ActionClient(
            self, SkillGrasp, SKILL_ACTIONS['grasp'][1],
            callback_group=self._cb_group)
        # GraspOutcome of the last successful grasp, or None
        self._last_grasp_outcome = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Planned grasp and place targets for RViz, re-sent on a timer
        # (publish_tf, _republish_tfs) so they do not expire; the skill that
        # registered a frame drops it (drop_tf) when its goal ends.
        self.tf_broadcaster = TransformBroadcaster(self)
        # child_frame_id -> TransformStamped
        self._persistent_tfs = {}
        self._tf_lock = threading.Lock()
        self.create_timer(TF_REPUBLISH_PERIOD_SEC, self._republish_tfs,
                          callback_group=self._cb_group)
        # visual-servo residual (object minus setpoint) in the hand-camera
        # optical frame, one per iteration, for live plotting
        self.servo_error_pub = self.create_publisher(
            Vector3Stamped, '/skill/grasp/servo_error', 10)
        # visual-servo executed move (gripper TF pose after the move minus
        # before), pelvis frame: how much of each correction the arm delivered
        self.servo_move_pub = self.create_publisher(
            Vector3Stamped, '/skill/grasp/servo_move', 10)
        # executed move minus the commanded correction, pelvis frame: the move's
        # tracking shortfall (zero = the arm did exactly as asked)
        self.servo_move_residual_pub = self.create_publisher(
            Vector3Stamped, '/skill/grasp/servo_move_residual', 10)
        # Annotated hand-camera frame per servo iteration: the measured object,
        # the setpoint with its X/Y tolerance and the commanded correction. Raw
        # sensor_msgs/Image, like yolo_server's '/annotated', because RViz's
        # Image display lists only raw topics. Depth 1 and published only while
        # a servo runs, so the bandwidth is bounded.
        self.servo_image_pub = self.create_publisher(
            Image, '/skill/grasp/servo_image', 1)

        # color CompressedImage, the gemini/sam input
        self._latest_image = None
        # aligned depth CompressedImage [mm]
        self._latest_depth = None
        # color CameraInfo (intrinsics and optical frame)
        self._latest_caminfo = None
        # head-camera YOLO DetectionBundle
        self._latest_detections = None
        self._latest_arm_detections = {arm: None for arm in ARM_DETECTION_TOPICS}
        self._latest_arm_images = {arm: None for arm in ARM_COLOR_IMAGE_TOPICS}
        self.create_subscription(
            CompressedImage, COLOR_IMAGE_TOPIC, self._on_color_image,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            CompressedImage, DEPTH_IMAGE_TOPIC, self._on_depth_image,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._on_caminfo,
            qos_profile_sensor_data, callback_group=self._cb_group)
        self.create_subscription(
            DetectionBundle, DETECTION_TOPIC, self._on_detections,
            qos_profile_sensor_data, callback_group=self._cb_group)
        for arm, topic in ARM_DETECTION_TOPICS.items():
            self.create_subscription(
                DetectionBundle, topic,
                lambda msg, a=arm: self._on_arm_detections(a, msg),
                qos_profile_sensor_data, callback_group=self._cb_group)
        for arm, topic in ARM_COLOR_IMAGE_TOPICS.items():
            self.create_subscription(
                CompressedImage, topic,
                lambda msg, a=arm: self._on_arm_image(a, msg),
                qos_profile_sensor_data, callback_group=self._cb_group)

        # With `wait_for_services`, wait for the H1-2's perception and gripper
        # services and the arm actions, 10 s in total; a missing one is not fatal.
        wait_for_services = bool(
            self.declare_parameter('wait_for_services', True).value)
        if wait_for_services:
            self.get_logger().info('Waiting for gemini/sam/graspgen, grippers, frame_task...')
            deadline = time.monotonic() + 10.0
            for cli in (self.gemini_cli, self.sam_cli, self.graspgen_cli,
                        *self.gripper_clis.values(), *self.gripper_open_clis.values(),
                        *self.gripper_close_clis.values(),
                        *self.gripper_force_clis.values()):
                rem = max(0.0, deadline - time.monotonic())
                if rem <= 0.0:
                    break
                cli.wait_for_service(timeout_sec=rem)
            for act_cli in (self.frame_task_cli, self.named_config_cli):
                rem = max(0.0, deadline - time.monotonic())
                if rem <= 0.0:
                    break
                act_cli.wait_for_server(timeout_sec=rem)

        # `record_runs`: each RECORDED_SKILLS run writes run_telemetry files under
        # <package>/logs/runs/<skill>/. One recorder is active at a time
        # (_active_run), so pick_place's inner grasp shares the outer one.
        self._record_runs = bool(
            self.declare_parameter('record_runs', True).value)
        self._run_log_root = _log_root('golem_skills', __file__)
        self._run_lock = threading.Lock()
        self._active_run = None

        # read per goal, so `ros2 param set /golem_skills grasp_method <name>`
        # applies to the next grasp; _check_grasp_method refuses unknown names
        self.declare_parameter('grasp_method', DEFAULT_METHOD, ParameterDescriptor(
            description=('grasp_synthesis method for /skill/grasp, one of '
                         f'{sorted(METHODS)}; goal.top_down overrides it')))
        self.add_on_set_parameters_callback(self._check_grasp_method)
        # Grasp execution depths, read per goal like grasp_method; the defaults
        # are grasp.py's constants, imported here because grasp.py imports this
        # module.
        from .skills.grasp import CONTACT_OFFSET, VISUAL_SERVO_DEPTH_M
        self.declare_parameter('contact_offset', CONTACT_OFFSET, ParameterDescriptor(
            description=('metres the open-loop contact move stops short of the '
                         'grasp pose along its approach axis (0 = full pose)')))
        self.declare_parameter('servo_depth', VISUAL_SERVO_DEPTH_M, ParameterDescriptor(
            description=('visual-servo target camera->object depth [m]; the finger '
                         'contact point is 0.0706 m ahead of the hand camera')))

    def _on_color_image(self, msg: CompressedImage):
        """Cache the latest head-camera color frame."""
        self._latest_image = msg

    def _on_depth_image(self, msg: CompressedImage):
        """Cache the latest head-camera aligned-depth frame."""
        self._latest_depth = msg

    def _on_caminfo(self, msg: CameraInfo):
        """Cache the latest head-camera color CameraInfo."""
        self._latest_caminfo = msg

    def _on_detections(self, msg: DetectionBundle):
        """Cache the latest head-camera YOLO DetectionBundle."""
        self._latest_detections = msg

    def _on_arm_detections(self, arm, msg: DetectionBundle):
        """Cache the latest hand-camera YOLO DetectionBundle for `arm`."""
        self._latest_arm_detections[arm] = msg

    def _on_arm_image(self, arm, msg: CompressedImage):
        """Cache the latest hand-camera color frame for `arm`."""
        self._latest_arm_images[arm] = msg

    def latest_image(self):
        """Return the latest head-camera color frame (CompressedImage), or None."""
        return self._latest_image

    def latest_caminfo(self):
        """Return the latest head-camera color CameraInfo, or None."""
        return self._latest_caminfo

    def latest_detections(self):
        """Return the latest head-camera YOLO DetectionBundle, or None."""
        return self._latest_detections

    def latest_arm_detections(self, arm):
        """Return the latest hand-camera YOLO DetectionBundle for `arm`, or None."""
        return self._latest_arm_detections.get(arm)

    def latest_arm_image(self, arm):
        """Return the latest hand-camera color frame (CompressedImage) for `arm`, or None."""
        return self._latest_arm_images.get(arm)

    def publish_tf(self, transform):
        """
        Broadcast a debug TransformStamped now and re-stamp it every TF_REPUBLISH_PERIOD_SEC.

        Keyed by child_frame_id: a second call with the same child moves the
        frame, drop_tf(child) stops it.
        """
        with self._tf_lock:
            self._persistent_tfs[transform.child_frame_id] = transform
        self.tf_broadcaster.sendTransform(transform)

    def drop_tf(self, child_frame_id):
        """Stop re-broadcasting a frame registered with publish_tf."""
        with self._tf_lock:
            self._persistent_tfs.pop(child_frame_id, None)

    def _republish_tfs(self):
        """Re-stamp every registered debug frame to now and broadcast it."""
        with self._tf_lock:
            tfs = list(self._persistent_tfs.values())
        if not tfs:
            return
        stamp = self.get_clock().now().to_msg()
        for t in tfs:
            t.header.stamp = stamp
        self.tf_broadcaster.sendTransform(tfs)

    def _check_grasp_method(self, params):
        """Refuse a `grasp_method` value that is not in METHODS."""
        for p in params:
            if p.name == 'grasp_method' and p.value not in METHODS:
                return SetParametersResult(
                    successful=False,
                    reason=f'grasp_method must be one of {sorted(METHODS)}')
        return SetParametersResult(successful=True)

    def _on_skill_cancel(self, _cancel_request):
        """Accept every cancel request of a skill goal."""
        return CancelResponse.ACCEPT

    def robot_now(self):
        """Return robot time, s: the node's clock under use_sim_time, else the steady clock."""
        if self._sim_time:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.monotonic()

    def _sleep_robot(self, seconds, stop):
        """
        Wait `seconds` of robot time, or less once `stop` (a threading.Event) is set.

        Return whether `stop` is set. Without use_sim_time it is stop.wait(seconds); under
        it the wait ends with the deadline of sim time, so it lasts while /clock stands
        still, and checks `stop` every ROBOT_WAIT_POLL_SEC of steady time.
        """
        if not self._sim_time:
            return stop.wait(seconds)
        end = self.robot_now() + seconds
        while self.robot_now() < end:
            if stop.wait(ROBOT_WAIT_POLL_SEC):
                return True
        return stop.is_set()

    def _timeout_sec(self, goal, default=DEFAULT_SKILL_TIMEOUT):
        """Return goal.timeout in seconds, or `default` when it is zero."""
        t = float(goal.timeout.sec) + float(goal.timeout.nanosec) * 1e-9
        return t if t > 0.0 else default

    def _wait_future(self, future, timeout_sec, outer_gh=None, stop_event=None,
                     halt_check=None, robot_time=False):
        """
        Block this executor worker thread until `future` resolves; return its result or None.

        Safe under the MultiThreadedExecutor: the other threads keep spinning.
        Polls every 0.1 s of steady time and returns None on timeout, when the
        outer skill goal `outer_gh` is asked to cancel, when `stop_event` (a
        threading.Event) is set, or when `halt_check` (a callable) returns a
        reason. `timeout_sec` is s of steady time (a service's or a link's
        latency), or of robot time with `robot_time` (a robot's motion). A
        wait on sim time also returns None once the node's context has shut
        down, so a stopped /clock cannot hold it at exit.
        """
        now = self.robot_now if robot_time else time.monotonic
        on_sim_time = robot_time and self._sim_time
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        deadline = now() + max(0.0, timeout_sec)
        while not event.is_set():
            if outer_gh is not None and outer_gh.is_cancel_requested:
                return None
            if stop_event is not None and stop_event.is_set():
                return None
            if halt_check is not None and halt_check() is not None:
                return None
            if on_sim_time and not self.context.ok():
                return None
            remaining = deadline - now()
            if remaining <= 0.0:
                return None
            event.wait(timeout=min(0.1, remaining))
        return future.result()

    def _call_service(self, client, request, name, timeout_sec=30.0, outer_gh=None):
        """
        Call `client` and block (executor-safe) for the response; return it or None.

        Returns None at once when the service is not ready. Pass the skill's
        goal handle as `outer_gh` so a cancel of that goal ends the wait.
        """
        if not client.service_is_ready():
            self.get_logger().error(f'{name} service not available')
            return None
        result = self._wait_future(
            client.call_async(request), timeout_sec, outer_gh)
        if result is None:
            self.get_logger().error(f'{name} call failed or timed out')
        return result

    def _send_action(self, client, goal, feedback_cb=None,
                     accept_timeout=10.0, result_timeout=120.0, outer_gh=None,
                     stop_event=None, halt_check=None):
        """
        Send an action goal, wait for acceptance, then for the result; return it or None.

        The result response has .status and .result. `accept_timeout` is s of
        steady time, `result_timeout` s of robot time. When the result wait
        ends early (timeout, outer-skill cancel, `stop_event` set or
        `halt_check` returning a reason), the goal is canceled ON THE SERVER.
        Arm goals pass `halt_check=self.arm_halt_reason`.
        """
        if result_timeout <= 0.05:
            self.get_logger().error('no time left to send action goal')
            return None
        handle = self._wait_future(
            client.send_goal_async(goal, feedback_callback=feedback_cb), accept_timeout)
        if handle is None or not handle.accepted:
            self.get_logger().error('action goal rejected or send timed out')
            return None
        # the result comes when the robot has moved: robot time
        response = self._wait_future(handle.get_result_async(), result_timeout,
                                     outer_gh, stop_event=stop_event,
                                     halt_check=halt_check, robot_time=True)
        if response is None:
            # SAFETY: cancel on the server; dropping the local future would
            # leave the robot executing an orphaned goal
            self.get_logger().warn('canceling in-flight inner action goal')
            self._wait_future(handle.cancel_goal_async(), 2.0)
        return response

    def query_gemini(self, prompt, image=None, timeout_sec=GEMINI_TIMEOUT_SEC,
                     outer_gh=None):
        """
        Ask the gemini_query service `prompt` about `image`; return Gemini's text or None.

        `image` defaults to the latest head-camera frame. Latency-sensitive
        callers pass a smaller `timeout_sec` (e.g. the skill's remaining
        budget); pass the skill goal handle as `outer_gh` so a cancel ends the
        possibly minutes-long call.
        """
        img = image if image is not None else self.latest_image()
        if img is None:
            self.get_logger().error('query_gemini: no head-camera image yet')
            return None
        req = GeminiQuery.Request()
        req.image = img
        req.prompt = prompt
        resp = self._call_service(self.gemini_cli, req, 'GeminiQuery',
                                  timeout_sec=timeout_sec, outer_gh=outer_gh)
        if resp is None or not resp.success:
            return None
        return resp.response

    def segment(self, text='', positive_boxes=None, negative_boxes=None,
                image=None, outer_gh=None):
        """
        Run the sam_segment service on `image`; return the mono8 mask (Image) or None.

        `image` defaults to the latest head-camera frame. Prompts are a text
        and/or pixel-xyxy box exemplars. Pass the skill goal handle as
        `outer_gh` so a cancel ends the wait.
        """
        img = image if image is not None else self.latest_image()
        if img is None:
            self.get_logger().error('segment: no head-camera image yet')
            return None
        req = SamSegment.Request()
        req.image = img
        req.text = text or ''
        req.positive_boxes = [float(v) for v in (positive_boxes or [])]
        req.negative_boxes = [float(v) for v in (negative_boxes or [])]
        resp = self._call_service(self.sam_cli, req, 'SamSegment',
                                  outer_gh=outer_gh)
        if resp is None or not resp.success:
            return None
        return resp.mask

    def _gemini_box(self, obj, run, gh):
        """
        Ask Gemini for one box around `obj` in the head camera; return pixel [x1, y1, x2, y2].

        Returns None when Gemini gives no usable box (SAM then uses the text
        prompt alone). The call is capped at the skill's remaining budget and
        GEMINI_TIMEOUT_SEC, and a cancel of `gh` ends it.
        """
        timeout = min(GEMINI_TIMEOUT_SEC, run.remaining())
        txt = self.query_gemini(GEMINI_GRASP_PROMPT.format(obj=obj),
                                timeout_sec=timeout, outer_gh=gh)
        data = extract_json(txt)
        entry = None
        if isinstance(data, list) and data:
            entry = data[0]
        elif isinstance(data, dict):
            entry = data
        if not isinstance(entry, dict) or 'box_2d' not in entry:
            return None
        info = self.latest_caminfo()
        if info is None or not info.width or not info.height:
            return None
        try:
            y1, x1, y2, x2 = (float(v) for v in entry['box_2d'])
        except (ValueError, TypeError):
            return None
        w, h = info.width, info.height
        px1, px2 = sorted((x1 / 1000.0 * w, x2 / 1000.0 * w))
        py1, py2 = sorted((y1 / 1000.0 * h, y2 / 1000.0 * h))
        return [px1, py1, px2, py2]

    def detect_object_cloud(self, obj, run, gh):
        """
        Locate `obj` in the head camera and return (obj_cloud, scene_cloud, None).

        Gemini box -> SAM mask -> pelvis-frame object cloud, plus the
        whole-frame scene cloud as obstacle context; (None, None, reason) on
        failure. pick_place uses it for the place target (the grasp skill has
        its own detect_object / segment_object). SAM always gets the object
        name, and Gemini's box too when there is one, so text and box together
        pick the right instance.
        """
        box = self._gemini_box(obj, run, gh)
        # None from Gemini is also a legitimate "no box", so tell a cancel or
        # timeout apart here before spending the SAM call
        if gh.is_cancel_requested or run.remaining() <= 0.0:
            return None, None, 'detection canceled or timed out'
        mask = self.segment(text=obj, positive_boxes=box, outer_gh=gh)
        if mask is None:
            return None, None, f'no mask for {obj!r}'
        obj_cloud = self.mask_to_cloud(mask, target_frame='pelvis')
        if obj_cloud is None:
            return None, None, f'{obj!r} mask produced no usable cloud'
        scene = self.scene_to_cloud(target_frame='pelvis')
        return obj_cloud, scene, None

    def _depth_to_cloud(self, mask, target_frame):
        """
        Back-project `mask` with the latest head depth into (N, 3) points in `target_frame`.

        `mask` is bool HxW on the color grid, or None for every valid pixel.
        Returns None when depth, intrinsics or TF are unavailable.
        """
        depth_msg, info = self._latest_depth, self._latest_caminfo
        if depth_msg is None or info is None:
            self.get_logger().error('_depth_to_cloud: missing depth or caminfo')
            return None
        try:
            depth = decode_compressed_depth_image(depth_msg).astype(np.float32) / 1000.0
        except (ValueError, TypeError) as e:
            self.get_logger().error(f'_depth_to_cloud: depth decode failed: {e}')
            return None
        if mask is None:
            mask = np.ones(depth.shape, dtype=bool)
        elif depth.shape != mask.shape:
            self.get_logger().error(
                f'_depth_to_cloud: mask {mask.shape} != depth {depth.shape}')
            return None
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pts_cam = deproject_mask(mask, depth, fx, fy, cx, cy, DEPTH_MIN_M, DEPTH_MAX_M)
        cam_frame = depth_msg.header.frame_id or info.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, cam_frame, Time(), timeout=RclpyDuration(seconds=1.0))
        except TransformException as e:
            self.get_logger().error(
                f'_depth_to_cloud: TF {cam_frame!r} -> {target_frame!r} failed: {e}')
            return None
        return transform_points(pts_cam, transform_to_matrix(tf.transform)).astype(np.float32)

    def mask_to_cloud(self, mask_msg, target_frame='pelvis'):
        """
        Back-project a mono8 mask with the latest head depth into an (N, 3) object cloud.

        The cloud is in `target_frame`; None when fewer than MIN_GRASP_POINTS
        are valid. The mask must be on the color grid, which the aligned depth
        shares.
        """
        if mask_msg is None:
            self.get_logger().error('mask_to_cloud: missing mask')
            return None
        mask = (np.frombuffer(bytes(mask_msg.data), dtype=np.uint8)
                .reshape(mask_msg.height, mask_msg.width) > 127)
        pts = self._depth_to_cloud(mask, target_frame)
        if pts is None:
            return None
        if len(pts) < MIN_GRASP_POINTS:
            self.get_logger().warn(f'mask_to_cloud: only {len(pts)} valid points')
            return None
        return pts

    def box_to_cloud(self, box, target_frame='pelvis'):
        """
        Back-project every valid pixel inside the pixel-xyxy `box` into an (N, 3) cloud.

        For small parts (screws, nuts, bolts) where SAM is unreliable, the
        detector box and depth are the object cloud. The box is clamped to the
        image and also picks up any surface showing around the part. Returns
        None when fewer than MIN_GRASP_POINTS are valid.
        """
        info = self.latest_caminfo()
        if box is None or info is None or not info.width or not info.height:
            self.get_logger().error('box_to_cloud: missing box or caminfo')
            return None
        w, h = int(info.width), int(info.height)
        x1, y1, x2, y2 = box
        u0, u1 = max(0, int(round(min(x1, x2)))), min(w, int(round(max(x1, x2))))
        v0, v1 = max(0, int(round(min(y1, y2)))), min(h, int(round(max(y1, y2))))
        if u1 <= u0 or v1 <= v0:
            self.get_logger().warn(f'box_to_cloud: degenerate box {box}')
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        pts = self._depth_to_cloud(mask, target_frame)
        if pts is None:
            return None
        if len(pts) < MIN_GRASP_POINTS:
            self.get_logger().warn(f'box_to_cloud: only {len(pts)} valid points')
            return None
        return pts

    def scene_to_cloud(self, target_frame='pelvis'):
        """Back-project the whole latest head depth frame into an (N, 3) scene cloud, or None."""
        return self._depth_to_cloud(None, target_frame)

    @staticmethod
    def bundle_camera_frame(bundle):
        """Return the optical frame of `bundle`'s depth (camera_info's as fallback), or None."""
        if bundle is None:
            return None
        return (bundle.depth_image.header.frame_id
                or bundle.camera_info.header.frame_id or None)

    def bundle_detection_point(self, bundle, det, min_depth=DEPTH_MIN_M):
        """
        Back-project detection `det` of `bundle` to one (x, y, z) point, or None.

        The point is in the bundle camera's own optical frame
        (bundle_camera_frame). It uses only what the bundle carries
        (yolo_server attaches each camera's aligned depth and intrinsics), so
        it works for any camera the detector runs on; that is what the
        hand-camera visual servo relies on. The depth must be aligned to the
        color grid the boxes were measured in.

        Takes the median over the CENTRAL QUARTER of the box, as
        _detection_point does: box edges pick up background depth, which on a
        small part is the support surface and would bias the range.

        `min_depth` [m] replaces DEPTH_MIN_M, which suits the head camera and
        would clip a hand camera that is deliberately closer to its target.
        """
        if bundle is None or det is None:
            return None
        depth, info = self._bundle_depth(bundle, 'bundle_detection_point')
        if depth is None:
            return None
        h, w = depth.shape
        cx_box = (det.bbox_min.x + det.bbox_max.x) / 2.0
        cy_box = (det.bbox_min.y + det.bbox_max.y) / 2.0
        half_w = max(1.0, abs(det.bbox_max.x - det.bbox_min.x) * 0.25)
        half_h = max(1.0, abs(det.bbox_max.y - det.bbox_min.y) * 0.25)
        u0, u1 = max(0, int(round(cx_box - half_w))), min(w, int(round(cx_box + half_w)))
        v0, v1 = max(0, int(round(cy_box - half_h))), min(h, int(round(cy_box + half_h)))
        if u1 <= u0 or v1 <= v0:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[v0:v1, u0:u1] = True
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pts = deproject_mask(mask, depth, fx, fy, cx, cy, min_depth, DEPTH_MAX_M)
        if pts is None or len(pts) == 0:
            return None
        return np.median(pts, axis=0)

    def _bundle_depth(self, bundle, caller):
        """
        Return (depth [m] HxW, camera_info) of DetectionBundle `bundle`.

        Returns (None, None), with a warning naming `caller`, when the bundle
        carries no depth, it does not decode, or it is not on the color grid.
        """
        depth_msg, info = bundle.depth_image, bundle.camera_info
        if not depth_msg.data or not info.width or not info.height:
            self.get_logger().warn(
                f'{caller}: bundle carries no depth/caminfo — is yolo_server new '
                'enough to attach them?')
            return None, None
        try:
            depth = decode_compressed_depth_image(depth_msg).astype(np.float32) / 1000.0
        except (ValueError, TypeError) as e:
            self.get_logger().warn(f'{caller}: depth decode failed: {e}')
            return None, None
        if (int(info.height), int(info.width)) != depth.shape:
            self.get_logger().warn(
                f'{caller}: depth {depth.shape[0]}x{depth.shape[1]} != caminfo '
                f'{info.height}x{info.width} — depth is not aligned to the color '
                'grid the boxes were measured in')
            return None, None
        return depth, info

    def bundle_mask_point(self, bundle, mask_msg, min_depth=DEPTH_MIN_M,
                          support_m=0.01):
        """
        Return one on-object (x, y, z) point of a mono8 mask on `bundle`'s color grid, or None.

        Back-projects the mask with the bundle's aligned depth (bundle camera
        optical frame) and takes the masked point nearest the mask's 3-D
        centroid, median-averaged with its neighbours within `support_m` [m].
        Unlike the centroid, that point lies on the part even when the part is
        bent or hollow.
        """
        if bundle is None or mask_msg is None:
            return None
        depth, info = self._bundle_depth(bundle, 'bundle_mask_point')
        if depth is None:
            return None
        mask = (np.frombuffer(bytes(mask_msg.data), dtype=np.uint8)
                .reshape(mask_msg.height, mask_msg.width) > 127)
        if mask.shape != depth.shape:
            self.get_logger().warn(
                f'bundle_mask_point: mask {mask.shape} != depth {depth.shape}')
            return None
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        pts = deproject_mask(mask, depth, fx, fy, cx, cy, min_depth, DEPTH_MAX_M)
        if pts is None or len(pts) == 0:
            return None
        nearest = pts[np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]
        near = pts[np.linalg.norm(pts - nearest, axis=1) <= support_m]
        return np.median(near, axis=0)

    def plan_grasp(self, cloud, frame, gripper_name, scene_cloud=None, arm=''):
        """
        Send an (N, 3) object cloud to the graspgen service; return its response or None.

        The response holds ranked `grasps` (PoseStamped[] in `frame`),
        `scores` and `gripper_width`. An optional (M, 3) `scene_cloud` in the
        same frame lets the server collision-filter the grasps; `arm` is
        recorded in the server's call log.
        """
        if cloud is None or len(cloud) < MIN_GRASP_POINTS:
            self.get_logger().error('plan_grasp: empty/too-small cloud')
            return None
        header = Header()
        header.frame_id = frame
        header.stamp = self.get_clock().now().to_msg()
        cloud_msg = point_cloud2.create_cloud_xyz32(
            header, np.asarray(cloud, dtype=np.float32))
        req = GraspGen.Request()
        req.object_cloud = cloud_msg
        if scene_cloud is not None and len(scene_cloud):
            req.scene_cloud = point_cloud2.create_cloud_xyz32(
                header, np.asarray(scene_cloud, dtype=np.float32))
        req.gripper_name = gripper_name
        req.arm = arm
        resp = self._call_service(self.graspgen_cli, req, 'GraspGen', timeout_sec=60.0)
        if resp is None or not resp.success or not resp.grasps:
            if resp is not None and resp.message:
                self.get_logger().error(f'plan_grasp: {resp.message}')
            return None
        return resp

    def move_frame_to(self, frame, pose,
                      outer_gh=None, duration_sec=3, do_plan=True,
                      slow_mode=False, label='', stable_stop_msgs=None,
                      lin_tol=0.01, ang_tol=0.05):
        """
        Drive URDF `frame` to `pose` (pelvis frame) with one /frame_task goal; return reached.

        `frame` is GRASP_FRAMES[arm] for grasping. Pass the skill's goal handle
        as `outer_gh` so a skill cancel cancels the frame_task goal too.

        `duration_sec` is a TIMEOUT. `slow_mode` runs the goal at the server's
        SLOW_MODE_SCALE (1/4) arm speed for delicate moves near an object or
        with a payload, and does not stretch the timeout: callers multiply it
        by SLOW_MODE_TIME_SCALE.

        `label` names what the move is for (e.g. 'contact descent') and is in
        every frame_task log line of the move.

        `stable_stop_msgs` (int, or None to disable): return early once the
        server's feedback stays unchanged for this many messages
        (_send_frame_move).

        `lin_tol` / `ang_tol` [m, rad]: the final pelvis-frame error within
        which the move counts as reached; loosen them for a rough staging move.
        """
        x, y, z = float(pose.position.x), float(pose.position.y), float(pose.position.z)
        quat = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        target = Pose()
        target.position = Point(x=x, y=y, z=z)
        target.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        return self._send_frame_move(frame, target, duration_sec, outer_gh, do_plan,
                                     slow_mode=slow_mode, label=label,
                                     stable_stop_msgs=stable_stop_msgs,
                                     lin_tol=lin_tol, ang_tol=ang_tol)

    def _on_safety_heartbeat(self, msg):
        """Record a /safety/heartbeat; a false beat latches the arm halt for the node's life."""
        if not msg.data:
            self._safety_latched = True
        self._last_beat = self._monotonic()

    def arm_halt_reason(self):
        """
        Return why arm motion must stop for the current skill goal, or None.

        SAFETY: while this returns a reason, no frame_task or named_config goal
        is sent and the one in flight is canceled, so a skill aborts rather
        than try another target. Halted after a false /safety/heartbeat (for
        the node's life: golem_safety's estop latches until restart), before
        the first beat, while the last beat is older than `heartbeat_timeout`,
        and for the rest of the goal after a move the server aborted or a halt
        that canceled a move in flight.
        """
        if self._safety_latched:
            return 'golem_safety estop latched'
        if self._last_beat is None:
            return 'safety heartbeat not received yet'
        silent = self._monotonic() - self._last_beat
        if silent > self.get_parameter('heartbeat_timeout').value:
            return f'safety heartbeat silent for {silent:.1f} s'
        return self._arm_halt_reason

    def _note_arm_move_status(self, response, what):
        """
        Latch an arm halt for the rest of the goal after a failed arm move `what`.

        Latched when the server aborted the move, or when the move has no
        result because a halt canceled it.
        """
        if response is not None and response.status == GoalStatus.STATUS_ABORTED:
            self._arm_halt_reason = (
                f'{what} was aborted by the server (estop, stale robot state, '
                'or an execution error)')
            self.get_logger().error(f'arm halted: {self._arm_halt_reason}')
        elif response is None:
            halt = self.arm_halt_reason()
            if halt is not None:
                self._arm_halt_reason = halt
                self.get_logger().error(f'{what} canceled: {halt}')

    def _send_frame_move(self, frame, pose, duration_sec, outer_gh, do_plan,
                         slow_mode=False, label='', stable_stop_msgs=None,
                         lin_tol=0.01, ang_tol=0.05):
        """
        Send ONE /frame_task goal driving `frame` to `pose` (pelvis frame); return reached.

        Reached means the goal succeeded (or stopped early as stable) AND the
        last streamed error is within (`lin_tol`, `ang_tol`).

        `stable_stop_msgs`: if set, cancel the goal and return once the
        streamed (lin, ang) feedback stays within FRAME_TASK_STABLE_EPS_* of
        its running reference for that many messages: the arm has settled and
        the server would only hold steady state until its timeout.
        """
        tag = f'{frame} | {label}' if label else frame
        halt = self.arm_halt_reason()
        if halt is not None:
            self.get_logger().error(f'frame_task[{tag}] not sent: {halt}')
            return False
        goal = FrameTask.Goal()
        goal.frame_names = [frame]
        goal.frame_targets = [pose]
        goal.plan = do_plan
        goal.slow_mode = bool(slow_mode)
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        # keeps the fractional seconds of the budget
        whole = int(duration_sec)
        goal.duration = Duration(
            sec=whole, nanosec=int(round((float(duration_sec) - whole) * 1e9)))

        self.get_logger().info(
            f'frame_task: {tag} -> ({x:.3f}, {y:.3f}, {z:.3f}) in {duration_sec}s'
            + (' [SLOW]' if slow_mode else '') + (' [planned]' if do_plan else ''))

        # the latest streamed errors (max over the driven frames), for the
        # throttled convergence log and the final reached test
        last = {}
        stable_event = threading.Event() if stable_stop_msgs else None
        stable = {'ref': None, 'n': 0}

        def _log_feedback(feedback_msg):
            """Log the streamed convergence and count unchanged feedbacks for the early stop."""
            fb = feedback_msg.feedback
            last['lin'] = max(fb.errors_linear) if fb.errors_linear else 0.0
            last['ang'] = max(fb.errors_angular) if fb.errors_angular else 0.0
            self.get_logger().info(
                f'frame_task[{tag}] converging: '
                f'lin={last["lin"] * 1000:.1f}mm ang={last["ang"]:.3f}rad',
                throttle_duration_sec=0.5)
            if stable_event is not None and not stable_event.is_set():
                # Within a band of the running reference, not exact equality:
                # steady-state feedback jitters ~0.1 mm. The reference moves only
                # when a reading leaves the band, so real motion keeps resetting
                # the count while a settled hold accrues it.
                ref = stable['ref']
                if (ref is not None
                        and abs(last['lin'] - ref[0]) <= FRAME_TASK_STABLE_EPS_LIN
                        and abs(last['ang'] - ref[1]) <= FRAME_TASK_STABLE_EPS_ANG):
                    stable['n'] += 1
                else:
                    stable['ref'], stable['n'] = (last['lin'], last['ang']), 1
                if stable['n'] >= stable_stop_msgs:
                    self.get_logger().info(
                        f'frame_task[{tag}] feedback stable (within '
                        f'{FRAME_TASK_STABLE_EPS_LIN * 1000:.1f}mm/'
                        f'{FRAME_TASK_STABLE_EPS_ANG:.3f}rad) for {stable_stop_msgs} '
                        f'msgs (lin={last["lin"] * 1000:.1f}mm '
                        f'ang={last["ang"]:.3f}rad) — stopping early')
                    stable_event.set()

        response = self._send_action(
            self.frame_task_cli, goal, feedback_cb=_log_feedback,
            result_timeout=float(duration_sec) + 10.0, outer_gh=outer_gh,
            stop_event=stable_event, halt_check=self.arm_halt_reason)
        self._note_arm_move_status(response, f'frame_task {tag!r}')
        stable_stopped = stable_event is not None and stable_event.is_set()
        ok = stable_stopped or (
            response is not None and response.status == GoalStatus.STATUS_SUCCEEDED)
        # The server's goal status is SUCCEEDED even when it times out short of
        # the target, so the last streamed error decides. The tolerances are
        # looser than the server's 1 mm / 2 mrad to absorb feedback jitter.
        if ok and last:
            ok = last['lin'] < lin_tol and last['ang'] < ang_tol
        if last:
            self.get_logger().info(
                f'frame_task[{tag}] done: lin={last["lin"] * 1000:.1f}mm '
                f'ang={last["ang"]:.3f}rad success={ok}')
        return ok

    def _transform_pose(self, pose, source_frame, target_frame):
        """
        Return `pose` (a Pose in `source_frame`) expressed in `target_frame`, or None.

        Uses the latest available transform (Time()), which avoids
        extrapolation-into-the-future errors on a lagging SLAM/odometry link.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=RclpyDuration(seconds=1.0))
        except TransformException as e:
            self.get_logger().warn(
                f'_transform_pose: TF {source_frame!r} -> {target_frame!r} failed: {e}')
            return None
        return matrix_to_pose(transform_to_matrix(tf.transform) @ pose_to_matrix(pose))

    def _world_frame_error(self, frame, target_pos, target_quat):
        """
        Return (lin [m], ang [rad]) between `frame`'s actual WORLD_FRAME pose and a target.

        The target is an xyz array and an (x, y, z, w) quaternion in
        WORLD_FRAME; (None, None) when TF fails.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                WORLD_FRAME, frame, Time(), timeout=RclpyDuration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(f'servo: lookup {WORLD_FRAME} -> {frame!r}: {e}')
            return None, None
        t, q = tf.transform.translation, tf.transform.rotation
        lin = float(np.linalg.norm(np.array([t.x, t.y, t.z]) - target_pos))
        ang = quat_geodesic((q.x, q.y, q.z, q.w), target_quat)
        return lin, ang

    def goto_named_config(self, name, duration_sec=0.0, plan=False,
                          slow_mode=False, result_timeout=60.0, outer_gh=None):
        """
        Drive the arms to frame_task_server's named configuration `name`; return reached.

        `duration_sec` 0 is the server's default move timeout; `plan` asks
        for a planned move instead of the direct IK descent; `slow_mode` runs
        it at SLOW_MODE_SCALE speed, so pass a larger `duration_sec` and
        `result_timeout`. Pass the skill goal handle as `outer_gh` so a skill
        cancel cancels the arm move.
        """
        halt = self.arm_halt_reason()
        if halt is not None:
            self.get_logger().error(f'named_config {name!r} not sent: {halt}')
            return False
        goal = NamedConfig.Goal()
        goal.plan = bool(plan)
        goal.config_name = name
        goal.slow_mode = bool(slow_mode)
        goal.duration.sec = int(duration_sec)
        goal.duration.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        response = self._send_action(self.named_config_cli, goal,
                                     result_timeout=result_timeout,
                                     outer_gh=outer_gh,
                                     halt_check=self.arm_halt_reason)
        self._note_arm_move_status(response, f'named_config {name!r}')
        ok = (response is not None
              and response.status == GoalStatus.STATUS_SUCCEEDED
              and response.result.success)
        self.get_logger().info(
            f'named_config {name!r}: {"reached" if ok else "FAILED"}')
        return ok

    def servo_frame_to_world(self, frame, world_pose, fallback_pose, outer_gh=None,
                             duration_sec=10.0, *, max_iter, lin_tol, ang_tol,
                             do_plan=True, slow_mode=False, label='',
                             stable_stop_msgs=None):
        """
        Drive `frame` to `world_pose` (WORLD_FRAME), compensating pelvis drift; return reached.

        Each iteration re-resolves the fixed world target into the LIVE pelvis
        frame, commands frame_task there, then measures the frame's actual
        world pose, and stops once within (`lin_tol` m, `ang_tol` rad).

        Returns True on convergence, and also best-effort after `max_iter`
        iterations or once the error diverges; False when the first move
        leaves the frame far from the target (unreachable: the caller tries
        its next candidate).

        With `world_pose` None (no world anchor: navigation or odometry down),
        or when the world TF drops mid-servo, it makes ONE direct pelvis-frame
        move to `fallback_pose` without drift compensation.

        `do_plan`, `slow_mode`, `label` and `stable_stop_msgs` are forwarded
        to every move. Pass do_plan=False for a short, committed straight-in
        move (pre-grasp -> contact), where collision-aware planning would fight
        the approach into the object. The caller scales `duration_sec` for
        slow mode; the per-iteration drift-fix budget is scaled here.
        """
        tag = f'{frame} | {label}' if label else frame
        if world_pose is None:
            self.get_logger().warn(
                f'servo[{tag}]: no {WORLD_FRAME} anchor; direct pelvis move '
                '(pelvis-drift compensation OFF)')
            return self.move_frame_to(frame, fallback_pose, outer_gh=outer_gh,
                                      duration_sec=duration_sec, do_plan=do_plan,
                                      slow_mode=slow_mode, label=label,
                                      stable_stop_msgs=stable_stop_msgs)
        tp = np.array([world_pose.position.x, world_pose.position.y,
                       world_pose.position.z])
        tq = (world_pose.orientation.x, world_pose.orientation.y,
              world_pose.orientation.z, world_pose.orientation.w)
        ok, lin, ang = False, None, None
        best_lin = None
        for i in range(int(max_iter)):
            if outer_gh is not None and outer_gh.is_cancel_requested:
                return False
            pelvis_pose = self._transform_pose(world_pose, WORLD_FRAME, 'pelvis')
            if pelvis_pose is None:
                self.get_logger().warn(
                    f'servo[{tag}]: lost {WORLD_FRAME} TF; pelvis fallback')
                return self.move_frame_to(frame, fallback_pose, outer_gh=outer_gh,
                                          duration_sec=duration_sec, do_plan=do_plan,
                                          slow_mode=slow_mode, label=label,
                                          stable_stop_msgs=stable_stop_msgs)
            # iteration 0 is the full approach (already scaled by the caller);
            # later ones are short drift fixes, scaled here for slow mode
            dur = duration_sec if i == 0 else (
                3.0 * (SLOW_MODE_TIME_SCALE if slow_mode else 1.0))
            iter_label = (f'{label} (drift fix {i})' if label and i else label)
            ok = self.move_frame_to(frame, pelvis_pose, outer_gh=outer_gh,
                                    duration_sec=dur, do_plan=do_plan,
                                    slow_mode=slow_mode, label=iter_label,
                                    stable_stop_msgs=stable_stop_msgs)
            lin, ang = self._world_frame_error(frame, tp, tq)
            # unmeasurable in the world frame: report the pelvis-frame result
            if lin is None:
                return ok
            self.get_logger().info(
                f'servo[{tag}] iter {i}: {WORLD_FRAME} err lin={lin * 1000:.1f}mm '
                f'ang={ang:.3f}rad (tol {lin_tol * 1000:.0f}mm/{ang_tol:.3f}rad, '
                f'reached={ok})')
            if lin <= lin_tol and ang <= ang_tol:
                return True
            # first move unconverged in the pelvis frame AND far off in the
            # world: unreachable, so the caller moves on to its next candidate
            if i == 0 and not ok and (lin > 0.05 or ang > 0.20):
                self.get_logger().warn(
                    f'servo[{tag}]: unreachable; caller tries next grasp')
                return False
            if best_lin is not None and lin > best_lin + SERVO_DIVERGE_M:
                self.get_logger().warn(
                    f'servo[{tag}]: diverging (lin {lin * 1000:.1f}mm vs best '
                    f'{best_lin * 1000:.1f}mm); stopping drift fixes, proceeding '
                    'best-effort')
                return True
            best_lin = lin if best_lin is None else min(best_lin, lin)
        # Out of iterations without meeting the tolerance: proceed best-effort
        # from the near-converged pose; real unreachability fails at iteration 0.
        self.get_logger().warn(
            f'servo[{tag}]: tol not met in {max_iter} iters; proceeding '
            f'best-effort (last lin={lin * 1000:.1f}mm ang={ang:.3f}rad)')
        return True

    def _on_gripper_state(self, arm, msg):
        """Cache the driver's published gripper aperture [mm] for `arm`."""
        self._gripper_state_mm[arm] = float(msg.position)

    def gripper_aperture(self, arm):
        """
        Return `arm`'s latest gripper opening [mm], or None.

        The driver's published state (which follows every close, including the
        force-based /close) wins; the last set_gripper result is the fallback
        before any state has arrived.
        """
        val = self._gripper_state_mm.get(arm)
        return val if val is not None else self._gripper_actual_mm.get(arm)

    def set_gripper(self, arm, position_mm, speed=1.0):
        """
        Command `arm`'s gripper to `position_mm` [mm]; return the driver's success.

        Every aperture command (open_gripper included) goes through here, and
        it first stops the arm's grip-hold loop, so an explicit position always
        wins over a hold that would keep re-closing on top of it.
        """
        self._stop_grip_hold(arm)
        req = SetGripperPosition.Request()
        req.position = float(position_mm)
        req.speed = float(speed)
        result = self._call_service(
            self.gripper_clis[arm], req, f'SetGripperPosition({arm})')
        if result is None:
            return False
        self._gripper_actual_mm[arm] = float(result.actual_position)
        self.get_logger().info(
            f'gripper {arm}: success={result.success} '
            f'actual={result.actual_position:.2f} mm — {result.message}')
        return result.success

    def set_gripper_force(self, arm, max_force_n=GRIP_FORCE_N):
        """Set `arm`'s grip-force limit [N] before a close; return the driver's success."""
        req = SetGripperForce.Request()
        req.max_force = float(max_force_n)
        result = self._call_service(
            self.gripper_force_clis[arm], req, f'SetGripperForce({arm})')
        if result is None:
            return False
        self.get_logger().info(
            f'gripper {arm} force limit {max_force_n:.1f} N: '
            f'success={result.success} — {result.message}')
        return result.success

    def _trigger_gripper(self, client, name):
        """Call a gripper Trigger service; return its success (False when unavailable)."""
        result = self._call_service(client, Trigger.Request(), name)
        if result is None:
            return False
        self.get_logger().info(f'{name}: success={result.success} — {result.message}')
        return result.success

    def open_gripper(self, arm, amount=1.0):
        """
        Open `arm`'s gripper to `amount` of its range (0 closed, 1 fully open); return success.

        A position command, so partial openings hold, followed by
        GRIPPER_SETTLE_SEC so the fingers stop before the arm moves. Opening
        ends the arm's grip-hold loop (through set_gripper).
        """
        amount = min(max(float(amount), 0.0), 1.0)
        if not self.set_gripper(arm, amount * GRIPPER_MAX_WIDTH_MM):
            return False
        self.get_clock().sleep_for(RclpyDuration(seconds=GRIPPER_SETTLE_SEC))
        return True

    def close_gripper(self, arm, force_n=GRIP_FORCE_N):
        """
        Close `arm`'s gripper on an object at `force_n` [N] and start its grip-hold loop.

        Sets the force limit (set_force), then triggers the driver's /close,
        which drives the fingers shut and holds against the object at that
        force whatever its width. Waits GRIPPER_SETTLE_SEC, since /close
        returns before the fingers finish, then starts the grip-hold loop
        (_grip_hold_loop), which keeps running after this call and after the
        skill returns, until the gripper is opened. Returns success.
        """
        if not self.set_gripper_force(arm, force_n):
            return False
        if not self._trigger_gripper(self.gripper_close_clis[arm], f'close({arm})'):
            return False
        self.get_clock().sleep_for(RclpyDuration(seconds=GRIPPER_SETTLE_SEC))
        # after the settle, so the loop's first tick cannot race the initial close
        self._start_grip_hold(arm)
        return True

    def _start_grip_hold(self, arm):
        """Start `arm`'s grip-hold loop, stopping a running one first (one loop per arm)."""
        self._stop_grip_hold(arm)
        stop = threading.Event()
        thread = threading.Thread(target=self._grip_hold_loop, args=(arm, stop),
                                  name=f'grip_hold_{arm}', daemon=True)
        self._grip_hold[arm] = (thread, stop)
        thread.start()
        self.get_logger().info(
            f'gripper {arm}: grip-hold loop started at {GRIP_HOLD_RATE_HZ:.0f} Hz '
            '(reset_overload + close); runs until the gripper is opened')

    def _stop_grip_hold(self, arm):
        """
        Stop `arm`'s grip-hold loop and wait (2 s at most) for its thread to end.

        Waiting means no tick is in flight when the caller then commands the
        gripper, so a stray re-close cannot race the open it is about to send.
        """
        entry = self._grip_hold.pop(arm, None)
        if entry is None:
            return
        thread, stop = entry
        stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            self.get_logger().warn(
                f'gripper {arm}: grip-hold thread did not stop within 2s; '
                'it will exit on its next tick')
        else:
            self.get_logger().info(f'gripper {arm}: grip-hold loop stopped')

    def _grip_hold_loop(self, arm, stop):
        """
        Re-assert the grip on `arm` every 1/GRIP_HOLD_RATE_HZ s of robot time until `stop`.

        This loop prevents gripper overload crashes: each tick calls
        reset_overload (clears a Dynamixel overload without releasing) then
        close (re-drives the fingers shut at the force close_gripper set), so
        an overload cannot leave the fingers limp and drop the object.

        Runs on its own thread, which is safe because the node spins a
        MultiThreadedExecutor over one reentrant group, so blocking on a
        service future here does not deadlock it. It calls _wait_future
        directly, not _call_service, so failures can be logged throttled
        (GRIP_HOLD_WARN_EVERY). A failed tick is never fatal; the next one
        tries again.
        """
        period = 1.0 / GRIP_HOLD_RATE_HZ
        calls = ((self.gripper_reset_clis[arm], 'reset_overload'),
                 (self.gripper_close_clis[arm], 'close'))
        fails = 0
        while not stop.is_set():
            ok, failed = True, ''
            for client, label in calls:
                if stop.is_set():
                    return
                result = None
                if client.service_is_ready():
                    result = self._wait_future(
                        client.call_async(Trigger.Request()),
                        GRIP_HOLD_CALL_TIMEOUT_SEC)
                if result is None or not result.success:
                    ok, failed = False, label
            if ok:
                fails = 0
            else:
                if fails % GRIP_HOLD_WARN_EVERY == 0:
                    self.get_logger().warn(
                        f'gripper {arm}: grip-hold {failed} failed '
                        f'({fails + 1} consecutive tick(s)); still retrying')
                fails += 1
            if self._sleep_robot(period, stop):
                break

    def _validated_arm(self, goal):
        """Return goal.arm normalized to 'left' or 'right', or None for any other value."""
        arm = goal.arm.strip().lower()
        if arm not in GRASP_FRAMES:
            return None
        return arm

    def _start_run_telemetry(self, skill, goal_handle):
        """
        Start a run_telemetry recorder for `skill`; return it, or None when not recording.

        None when `record_runs` is off, `skill` is not in RECORDED_SKILLS, or
        a recorder is already active (pick_place's in-process grasp shares the
        outer one). The caller hands the return value to _finish_run_telemetry.
        """
        if not self._record_runs or skill not in RECORDED_SKILLS:
            return None
        goal = getattr(goal_handle, 'request', None)
        with self._run_lock:
            if self._active_run is not None:
                return None
            arm = (getattr(goal, 'arm', '') or '').strip().lower()
            out_dir = run_dir(self._run_log_root, skill,
                              getattr(goal, 'target_object', ''))
            request = describe_goal(goal)
            # grasp and pick_place goals
            if hasattr(goal, 'top_down'):
                request['grasp_method'] = method_name(
                    goal.top_down, self.get_parameter('grasp_method').value)
            try:
                rec = RunRecorder(
                    self, skill, out_dir, WORLD_FRAME, GRASP_FRAMES,
                    arm=arm if arm in GRASP_FRAMES else None,
                    request=request).start()
            # recording must never break a skill
            except Exception as e:
                self.get_logger().warn(f'run telemetry: start failed: {e}')
                return None
            self._active_run = rec
            return rec

    def _note_run_arm(self, arm):
        """Tell the active run recorder, if any, which arm the skill chose."""
        rec = self._active_run
        if rec is not None:
            rec.note_arm(arm)

    def _note_run_target(self, xyz):
        """Tell the active run recorder, if any, the task target (a world Pose or (x, y, z))."""
        rec = self._active_run
        if rec is not None:
            rec.note_target(xyz)

    def _finish_run_telemetry(self, rec, success=None, message=None):
        """
        Stop `rec` (what _start_run_telemetry returned) and write its result.json.

        None does nothing. Only the starter clears the active slot, so a
        nested call cannot stop the outer recorder.
        """
        if rec is None:
            return
        try:
            rec.finish(success=success, message=message)
        finally:
            with self._run_lock:
                if self._active_run is rec:
                    self._active_run = None
