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
"""
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger


class MeasurementBridge(Node):
    """Run the robocasa_measurement node: the task signals and the band toggle."""

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

        Call after env.update_state(), holding the sim lock.
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
        return held
