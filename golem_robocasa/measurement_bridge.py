"""Publish the RoboCasa task's signals on ROS and serve the elastic band's toggle.

The sim loop steps the env's own MjData, so once per control step
(``env.control_timestep``, as RoboCasa's own ``env.step`` does) it calls
``env.update_state()`` (fixture state: temperature, timer, door, rack) and then
``tick()``, which reads ``_check_success`` and publishes:

  /robocasa/task_name   std_msgs/String   the task env name, latched
  /robocasa/task_goal   std_msgs/String   the language instruction, latched
  /robocasa/success     std_msgs/Bool     success held for debounce_steps control steps
  /robocasa/reward      std_msgs/Float32  float(success) of this control step

These are privileged task signals the real robot does not have: no robot-stack
node may depend on them.

With GOLEM_SIM_EXPERIMENT=1 (off by default) it also serves a ground-truth
experiment surface, for experiment harnesses only; no robot-stack node may
consume or call it, since the real robot has no such interface:

  /sim/ground_truth     std_msgs/String   JSON at the control rate: sim time,
                        every non-robot free body's world pose ("objects"),
                        the pose of every body attached under one ("links",
                        e.g. a cable's segments), the robot's whole-body CoM
                        and pelvis pose, and both ankle_roll_link poses
                        (poses are [x, y, z, qw, qx, qy, qz], world frame, m)
  /sim/reset_objects    std_srvs/Trigger  put every non-robot free body, and
                        the joints of the bodies attached under it, back at
                        their authored poses, at rest, on the next control step
"""
import json
import os

import mujoco
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger


class MeasurementBridge(Node):
    """Run the robocasa_measurement node: task signals, the band toggle, the experiment surface."""

    def __init__(self, env, debounce_steps=15, elastic_band=None, task_name=""):
        """Publish env's task signals; serve /elastic_band/toggle when elastic_band is given.

        debounce_steps is how many consecutive successful control steps make a
        success (RoboCasa teleop uses ~15), so a transient frame does not count.
        """
        super().__init__("robocasa_measurement")
        self.env = env
        self.debounce_steps = int(debounce_steps)
        self._streak = 0
        self.lang = ""
        self.task_name = task_name

        self.elastic_band = elastic_band
        if elastic_band is not None:
            self.create_service(Trigger, "/elastic_band/toggle", self._on_elastic_band_toggle)

        # transient_local, so a late subscriber still gets the task name and goal.
        from rclpy.qos import QoSDurabilityPolicy, QoSProfile

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._task_pub = self.create_publisher(String, "/robocasa/task_name", latched)
        self._goal_pub = self.create_publisher(String, "/robocasa/task_goal", latched)
        self._succ_pub = self.create_publisher(Bool, "/robocasa/success", 10)
        self._rew_pub = self.create_publisher(Float32, "/robocasa/reward", 10)

        self._gt_pub = None
        self._reset_pending = False
        if os.environ.get("GOLEM_SIM_EXPERIMENT", "0").strip().lower() not in (
                "0", "off", "false", "no", ""):
            self._init_experiment()

    def _init_experiment(self):
        """Serve /sim/ground_truth and /sim/reset_objects for every non-robot free body."""
        model = self.env.sim.model._model
        pfx = self.env.robots[0].robot_model.naming_prefix

        def body(i):
            """Return body i's name, or ""."""
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""

        free = [j for j in range(model.njnt)
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        objects = [j for j in free if not body(model.jnt_bodyid[j]).startswith(pfx)]
        self._gt_objects = {body(model.jnt_bodyid[j]): int(model.jnt_bodyid[j])
                            for j in objects}
        object_roots = set(self._gt_objects.values())

        def root_of(b):
            """Return the child of the world body that body b hangs under."""
            while model.body_parentid[b] != 0:
                b = model.body_parentid[b]
            return b

        self._gt_links = {body(b): b for b in range(1, model.nbody)
                          if b not in object_roots and root_of(b) in object_roots}
        # (qpos address, qpos width, dof address, dof width) of every joint of an
        # object or of a body under it; the authored pose is model.qpos0.
        widths = {mujoco.mjtJoint.mjJNT_FREE: (7, 6), mujoco.mjtJoint.mjJNT_BALL: (4, 3),
                  mujoco.mjtJoint.mjJNT_SLIDE: (1, 1), mujoco.mjtJoint.mjJNT_HINGE: (1, 1)}
        self._reset_joints = [
            (int(model.jnt_qposadr[j]), widths[model.jnt_type[j]][0],
             int(model.jnt_dofadr[j]), widths[model.jnt_type[j]][1])
            for j in range(model.njnt) if root_of(model.jnt_bodyid[j]) in object_roots]
        robot_roots = [int(model.jnt_bodyid[j]) for j in free
                       if body(model.jnt_bodyid[j]).startswith(pfx)]
        self._gt_robot_root = robot_roots[0] if robot_roots else None
        self._gt_feet = {}
        for side in ("left", "right"):
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  f"{pfx}{side}_ankle_roll_link")
            if i >= 0:
                self._gt_feet[side] = i
        self._gt_pub = self.create_publisher(String, "/sim/ground_truth", 10)
        self.create_service(Trigger, "/sim/reset_objects", self._on_reset_objects)
        self.get_logger().warn(
            f"GOLEM_SIM_EXPERIMENT=1: /sim/ground_truth + /sim/reset_objects for "
            f"{sorted(self._gt_objects)} (experiment harnesses only)")

    def _on_reset_objects(self, request, response):
        """Ask tick() to reset the objects, which it does holding the sim lock."""
        self._reset_pending = True
        response.success = True
        response.message = (f"resetting {len(self._gt_objects)} objects "
                            f"({len(self._reset_joints)} joints) on the next step")
        return response

    def _reset_objects(self):
        """Put every object joint back at its authored pose, at rest."""
        model, data = self.env.sim.model._model, self.env.sim.data._data
        for qadr, nq, dadr, nv in self._reset_joints:
            data.qpos[qadr:qadr + nq] = model.qpos0[qadr:qadr + nq]
            data.qvel[dadr:dadr + nv] = 0.0
        mujoco.mj_forward(model, data)

    def _publish_ground_truth(self):
        """Publish /sim/ground_truth: poses [x, y, z, qw, qx, qy, qz] (m, world) as JSON."""
        data = self.env.sim.data._data

        def pose(b):
            """Return body b's world pose as [x, y, z, qw, qx, qy, qz]."""
            return [float(v) for v in data.xpos[b]] + [float(v) for v in data.xquat[b]]

        msg = {
            "t": float(data.time),
            "objects": {name: pose(b) for name, b in self._gt_objects.items()},
            "links": {name: pose(b) for name, b in self._gt_links.items()},
            "feet": {side: pose(b) for side, b in self._gt_feet.items()},
        }
        if self._gt_robot_root is not None:
            msg["robot_com"] = [float(v) for v in data.subtree_com[self._gt_robot_root]]
            msg["pelvis"] = pose(self._gt_robot_root)
        self._gt_pub.publish(String(data=json.dumps(msg)))

    def _on_elastic_band_toggle(self, request, response):
        """Toggle the band and answer with its new state."""
        enabled = self.elastic_band.toggle()
        response.success = True
        response.message = "enabled" if enabled else "disabled"
        return response

    def publish_goal(self):
        """Publish the task name and the episode's language goal, and return the goal."""
        self._task_pub.publish(String(data=self.task_name))
        try:
            self.lang = self.env.get_ep_meta().get("lang", "") or ""
        except Exception as e:  # pragma: no cover - container only
            self.get_logger().warn(f"get_ep_meta failed: {e}")
            self.lang = ""
        self._goal_pub.publish(String(data=self.lang))
        self.get_logger().info(f"task {self.task_name!r} goal: {self.lang!r}")
        return self.lang

    def tick(self):
        """Publish this control step's success and reward and return the debounced success.

        Call after env.update_state(), holding the sim lock: it also applies a
        pending object reset and publishes ground truth when enabled.
        """
        try:
            success = bool(self.env._check_success())
        except Exception as e:  # pragma: no cover - container only
            self.get_logger().warn(f"_check_success failed: {e}")
            success = False
        self._streak = self._streak + 1 if success else 0
        held = self._streak >= self.debounce_steps
        self._succ_pub.publish(Bool(data=held))
        self._rew_pub.publish(Float32(data=float(success)))
        if self._reset_pending:
            self._reset_pending = False
            self._reset_objects()
        if self._gt_pub is not None:
            try:
                self._publish_ground_truth()
            except Exception as e:  # pragma: no cover - container only
                self.get_logger().warn(f"ground truth publish failed: {e}")
        return held
