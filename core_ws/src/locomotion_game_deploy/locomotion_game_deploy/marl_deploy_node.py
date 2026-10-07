"""LocoMotionGameDeploy node: the two-agent MARL policy (legs + IK arms) on the H1-2.

Flow
----
1. Idle until the first /lowstate.
2. Pre-pose: ramp all 27 joint targets (cosine blend over ``prepose_ramp_s``)
   from the measured pose to the policy's default pose at the training gains,
   then wait for the joints to settle there.
3. With ``engage_wait_for_confirm`` (default true; the sim launch turns it
   off) hold the pre-pose until an operator calls ~/confirm_engage.
4. Engage: the controller starts an episode at the measured state and runs at
   ``control_hz`` (50 Hz, the training policy rate). In the sim the elastic
   band is released ``band_release_delay_s`` later and the episode restarts
   then, so the policy's first free step is its episode's first.

While engaged the robot navigates on /cmd_vel, or reaches for an arm goal:
a PoseArray on ~/arm_goal (left then right wrist_yaw_link pose, pelvis frame)
stops it, settles and reaches; ~/end_arm_goal settles and returns to walking.
/cmd_vel is ignored during an arm goal: the policy only ever saw zero there.

This node owns all 27 joints: legs on /safety/lowcmd_lower_in, torso and arms
on /safety/lowcmd_upper_in. frame_task_server and lowerbody_controller_node
must not run beside it (the bringup gates them on lowerbody:=marl_ik).

Safety
------
* /safety/heartbeat watchdog, as in h12_lowerbody_rl: armed on the first beat;
  a False beat (the safety layer's estop latched) or ``heartbeat_timeout`` of
  silence latches this node silent.
* /lowstate older than ``lowstate_timeout`` (wall clock): no policy step and no publish
  that tick (stepping on a frozen state would integrate garbage).
* A non-finite observation or target latches the node silent instead of
  sending it: the safety layer would latch its estop (all motors limp) on it.
Silent means the safety layer keeps holding the last targets until it is
stopped; it has no staleness watchdog on these channels.

Interfaces
----------
sub  /lowstate                       (unitree_hg/LowState)
sub  /cmd_vel                        (geometry_msgs/Twist)
sub  /safety/heartbeat               (std_msgs/Bool)
sub  ~/arm_goal                      (geometry_msgs/PoseArray)  2 poses, pelvis frame
pub  /safety/lowcmd_lower_in         (unitree_hg/LowCmd)  rows 0-11
pub  /safety/lowcmd_upper_in         (unitree_hg/LowCmd)  rows 12-26
pub  ~/state                         (std_msgs/String, latched)
pub  ~/wrist_error                   (std_msgs/Float32MultiArray)  m, per arm, while engaged
srv  ~/confirm_engage                (std_srvs/Trigger)
srv  ~/end_arm_goal                  (std_srvs/Trigger)
"""

import os
import time

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseArray, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray, String
from std_srvs.srv import Trigger
from unitree_hg.msg import LowCmd, LowState

from locomotion_game_deploy.controller import (
    ARM_START,
    NUM_JOINTS,
    NUM_LEGS,
    JointCommand,
    MarlController,
    RobotState,
)
from locomotion_game_deploy.kinematics import ArmKinematics
from locomotion_game_deploy.networks import TorchScriptNetwork, ZeroNetwork
from locomotion_game_deploy.rotations import xyzw_to_wxyz

MOTOR_MODE_PR = 1
SAFETY_HEARTBEAT_TOPIC = "/safety/heartbeat"
PREPOSE_TOL = 0.15          # rad; max |q - default| over all 27 joints to count as posed
PREPOSE_SETTLE_TICKS = 10   # consecutive in-tolerance ticks (0.2 s at 50 Hz)
PREPOSE_MAX_TICKS = 400     # after the ramp; posed or not, stop waiting (8 s at 50 Hz)


def _share(*parts: str) -> str:
    return os.path.join(get_package_share_directory("locomotion_game_deploy"), *parts)


class MarlDeployNode(Node):
    def __init__(self):
        super().__init__("marl_deploy_node")
        self.declare_parameter("config", _share("policies", "marl_ik", "marl_ik.yaml"))
        self.declare_parameter("urdf_path", "/home/code/CL_Assets/ros_assets/h1_2_magpie.urdf")
        # "torch": the exported networks next to the config. "zero": stand-ins (legs hold the
        # default pose, arms pure IK, no odometry), for plumbing tests without a checkpoint.
        self.declare_parameter("policy_mode", "torch")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("prepose_ramp_s", 2.0)
        self.declare_parameter("engage_wait_for_confirm", True)
        self.declare_parameter("disable_elastic_band", True)
        self.declare_parameter("band_release_delay_s", 1.0)
        self.declare_parameter("heartbeat_timeout", 0.5)
        self.declare_parameter("lowstate_timeout", 0.1)

        cfg_path = self.get_parameter("config").value
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
        # export_marl.py's per-checkpoint values (rest pose, source files) override the template
        export_path = os.path.join(cfg_dir, "export.yaml")
        if os.path.isfile(export_path):
            with open(export_path) as f:
                cfg.update(yaml.safe_load(f) or {})
            self.get_logger().info(f"export: {cfg.get('source')}")
        control_hz = float(self.get_parameter("control_hz").value)
        if abs(1.0 / control_hz - float(cfg["control_dt"])) > 1e-6:
            raise ValueError(f"control_hz {control_hz} doesn't match the policy's control_dt {cfg['control_dt']}")

        torch.set_num_threads(1)
        # leg odometry needs the feet, and the IMU's frame when the IMU isn't on the pelvis
        legs_odom = cfg.get("legs_odometry", {}) if cfg.get("odometry") == "legs" else {}
        extra_frames = list(legs_odom.get("foot_frames", []))
        if legs_odom.get("imu_frame", "pelvis") != "pelvis":
            extra_frames.append(legs_odom["imu_frame"])
        kin = ArmKinematics(
            self.get_parameter("urdf_path").value,
            joint_names=cfg["joint_names"],
            ee_frames=cfg["arms"]["ee_frames"],
            arm_joint_names=[cfg["joint_names"][ARM_START:ARM_START + 7], cfg["joint_names"][ARM_START + 7:]],
            extra_frames=extra_frames,
        )
        mode = str(self.get_parameter("policy_mode").value)
        if mode == "torch":
            legs = TorchScriptNetwork(os.path.join(cfg_dir, cfg["legs"]["policy"]),
                                      cfg["legs"]["num_obs"], cfg["legs"]["num_actions"])
            arms = TorchScriptNetwork(os.path.join(cfg_dir, cfg["arms"]["policy"]),
                                      cfg["arms"]["num_obs"], cfg["arms"]["num_actions"])
            estimator = TorchScriptNetwork(os.path.join(cfg_dir, cfg["estimator"]["path"]),
                                           cfg["estimator"]["num_inputs"], 6)
        elif mode == "zero":
            self.get_logger().warn("policy_mode=zero: stand-in networks, for plumbing tests only")
            legs, arms, estimator = (ZeroNetwork(cfg["legs"]["num_actions"]),
                                     ZeroNetwork(cfg["arms"]["num_actions"]), ZeroNetwork(6))
        else:
            raise ValueError(f"policy_mode must be 'torch' or 'zero', not {mode!r}")
        self.controller = MarlController(cfg, kin, legs, arms, estimator)

        self._lowstate: LowState | None = None
        self._lowstate_time: float | None = None   # monotonic receipt time
        self._confirm_wait = bool(self.get_parameter("engage_wait_for_confirm").value)
        self._confirmed = False
        self._ramp_ticks = max(1, int(round(float(self.get_parameter("prepose_ramp_s").value) * control_hz)))
        self._prepose_ticks = 0
        self._prepose_pass = 0
        self._prepose_start: np.ndarray | None = None
        self._engaged = False
        self._band_released = not bool(self.get_parameter("disable_elastic_band").value)
        self._band_release_at: float | None = None
        self._band_inflight = False
        self._lowstate_timeout = float(self.get_parameter("lowstate_timeout").value)
        self._hb_timeout = float(self.get_parameter("heartbeat_timeout").value)
        self._hb_last: float | None = None
        self._dead = False                  # latched: never publishes again
        self._state_name = ""

        lowstate_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LowState, "/lowstate", self._on_lowstate, lowstate_qos)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(Bool, SAFETY_HEARTBEAT_TOPIC, self._on_heartbeat, 10)
        self.create_subscription(PoseArray, "~/arm_goal", self._on_arm_goal, 10)
        self._lower_pub = self.create_publisher(LowCmd, "/safety/lowcmd_lower_in", 10)
        self._upper_pub = self.create_publisher(LowCmd, "/safety/lowcmd_upper_in", 10)
        self._state_pub = self.create_publisher(String, "~/state", latched)
        self._err_pub = self.create_publisher(Float32MultiArray, "~/wrist_error", 10)
        self.create_service(Trigger, "~/confirm_engage", self._on_confirm)
        self.create_service(Trigger, "~/end_arm_goal", self._on_end_arm_goal)
        self._band_cli = self.create_client(Trigger, "/elastic_band/toggle")
        self.create_timer(1.0 / control_hz, self._tick)
        self._set_state("idle")
        self.get_logger().info(
            f"marl_deploy_node ready: {cfg_path} ({mode}), {control_hz:.0f} Hz, "
            f"confirm={'required' if self._confirm_wait else 'auto'}")

    # -- callbacks -------------------------------------------------------------------
    def _on_lowstate(self, msg: LowState) -> None:
        self._lowstate = msg
        self._lowstate_time = time.monotonic()

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.controller.set_velocity_command(np.array([msg.linear.x, msg.linear.y, msg.angular.z]))

    def _on_arm_goal(self, msg: PoseArray) -> None:
        n = self.controller.num_arms
        if len(msg.poses) != n:
            self.get_logger().error(f"arm goal needs {n} poses (left, right), got {len(msg.poses)}; ignored")
            return
        poses = np.array([[p.position.x, p.position.y, p.position.z,
                           *xyzw_to_wxyz(np.array([p.orientation.x, p.orientation.y,
                                                   p.orientation.z, p.orientation.w]))]
                          for p in msg.poses])
        if not np.isfinite(poses).all() or np.any(np.linalg.norm(poses[:, 3:], axis=1) < 1e-6):
            self.get_logger().error("arm goal has a non-finite position or a zero quaternion; ignored")
            return
        if not self._engaged:
            self.get_logger().warn("arm goal before the policy is engaged; ignored")
            return
        self.controller.request_arm_goal(poses)
        self.get_logger().info(f"arm goal: left {poses[0, :3].round(3)}, right {poses[1, :3].round(3)} (pelvis)")

    def _on_heartbeat(self, msg: Bool) -> None:
        if self._dead:
            return
        if not msg.data:
            self._latch("safety layer reports estop latched (heartbeat False)")
            return
        self._hb_last = time.monotonic()

    def _on_confirm(self, _req, resp):
        self._confirmed = True
        resp.success = True
        resp.message = "already engaged" if self._engaged else "confirmed: engaging once the pre-pose settles"
        self.get_logger().info(f"operator confirm: {resp.message}")
        return resp

    def _on_end_arm_goal(self, _req, resp):
        self.controller.end_arm_goal()
        resp.success = True
        resp.message = f"mode {self.controller.mode.value}"
        return resp

    # -- helpers -------------------------------------------------------------------------
    def _latch(self, reason: str) -> None:
        self._dead = True
        self._set_state(f"fault: {reason}")
        self.get_logger().error(
            f"LATCHED SILENT: {reason}. No more commands on /safety/lowcmd_*_in; "
            "restart this node once the cause is cleared.")

    def _set_state(self, name: str) -> None:
        if name != self._state_name:
            self._state_name = name
            self._state_pub.publish(String(data=name))

    def _robot_state(self, msg: LowState) -> RobotState:
        motors = msg.motor_state
        return RobotState(
            q=np.array([motors[i].q for i in range(NUM_JOINTS)], dtype=np.float64),
            dq=np.array([motors[i].dq for i in range(NUM_JOINTS)], dtype=np.float64),
            tau=np.array([motors[i].tau_est for i in range(NUM_JOINTS)], dtype=np.float64),
            quat=np.asarray(msg.imu_state.quaternion, dtype=np.float64),
            gyro=np.asarray(msg.imu_state.gyroscope, dtype=np.float64),
            acc=np.asarray(msg.imu_state.accelerometer, dtype=np.float64),
        )

    def _publish(self, cmd: JointCommand, mode_machine: int) -> None:
        if not (np.isfinite(cmd.q).all() and np.isfinite(cmd.kp).all() and np.isfinite(cmd.kd).all()):
            self._latch(f"non-finite joint command {cmd.q.round(3).tolist()}")
            return
        for pub, rows in ((self._lower_pub, range(0, NUM_LEGS)), (self._upper_pub, range(NUM_LEGS, NUM_JOINTS))):
            msg = LowCmd()
            msg.mode_pr = 0
            msg.mode_machine = mode_machine   # the safety layer copies the header of whichever channel spoke last
            for i in rows:
                m = msg.motor_cmd[i]
                m.mode = MOTOR_MODE_PR
                m.q = float(cmd.q[i])
                m.dq = 0.0
                m.tau = 0.0
                m.kp = float(cmd.kp[i])
                m.kd = float(cmd.kd[i])
            pub.publish(msg)

    def _release_band(self) -> None:
        if self._band_released or self._band_inflight:
            return
        if not self._band_cli.service_is_ready():
            self.get_logger().warn("waiting for /elastic_band/toggle", throttle_duration_sec=5.0)
            return
        self._band_inflight = True
        self._band_cli.call_async(Trigger.Request()).add_done_callback(self._on_band_toggled)

    def _on_band_toggled(self, fut) -> None:
        self._band_inflight = False
        result = fut.result()
        if result is None:
            self.get_logger().warn("elastic band toggle failed; retrying")
            return
        self._band_released = True
        self.get_logger().info(f"elastic band released ({result.message}); episode restarts")
        if self._lowstate is not None:
            self.controller.reset(self._robot_state(self._lowstate))

    # -- main loop ---------------------------------------------------------------------------
    def _tick(self) -> None:
        if self._dead:
            return
        now = time.monotonic()
        if self._hb_last is not None and now - self._hb_last > self._hb_timeout:
            self._latch(f"no {SAFETY_HEARTBEAT_TOPIC} for {self._hb_timeout:.1f} s (safety node down?)")
            return
        if self._lowstate is None:
            return
        if now - self._lowstate_time > self._lowstate_timeout:
            self.get_logger().warn(
                f"/lowstate is {now - self._lowstate_time:.2f} s old; not stepping", throttle_duration_sec=1.0)
            return
        msg = self._lowstate
        state = self._robot_state(msg)
        if not (np.isfinite(state.q).all() and np.isfinite(state.quat).all()):
            self._latch("non-finite /lowstate")
            return

        if not self._engaged:
            self._prepose(state, msg.mode_machine)
            return

        if self._band_release_at is not None and not self._band_released and now >= self._band_release_at:
            self._release_band()

        cmd = self.controller.step(state)
        for name, obs in self.controller.last_obs.items():
            if not np.isfinite(obs).all():
                self._latch(f"non-finite {name} observation")
                return
        self._publish(cmd, msg.mode_machine)
        self._set_state(self.controller.mode.value)
        self._err_pub.publish(Float32MultiArray(data=self.controller.wrist_errors().astype(np.float32).tolist()))

    def _prepose(self, state: RobotState, mode_machine: int) -> None:
        ctrl = self.controller
        if self._prepose_start is None:
            self._prepose_start = state.q.copy()
            self._set_state("preposing")
        target = ctrl.default.copy()
        if self._prepose_ticks < self._ramp_ticks:
            frac = self._prepose_ticks / self._ramp_ticks
            alpha = 0.5 * (1.0 - np.cos(np.pi * frac))
            target = (1.0 - alpha) * self._prepose_start + alpha * target
        self._publish(JointCommand(q=target, kp=ctrl.kp.copy(), kd=ctrl.kd.copy()), mode_machine)
        self._prepose_ticks += 1
        err = float(np.max(np.abs(state.q - ctrl.default)))
        self._prepose_pass = self._prepose_pass + 1 if err < PREPOSE_TOL else 0
        settled = self._prepose_pass >= PREPOSE_SETTLE_TICKS
        timed_out = self._prepose_ticks >= self._ramp_ticks + PREPOSE_MAX_TICKS
        if not settled and not timed_out:
            return
        if self._confirm_wait and not self._confirmed:
            self._set_state("waiting_confirm")
            self.get_logger().info(
                f"pre-posed ({'settled' if settled else 'NOT SETTLED'}, max err {err:.3f} rad); holding. "
                "Engage with: ros2 service call /marl_deploy_node/confirm_engage std_srvs/srv/Trigger",
                throttle_duration_sec=5.0)
            return
        self.get_logger().info(f"pre-posed ({'settled' if settled else 'timeout'}, max err {err:.3f} rad): engaging")
        ctrl.reset(state)
        self._engaged = True
        if not self._band_released:
            self._band_release_at = time.monotonic() + float(self.get_parameter("band_release_delay_s").value)


def main():
    rclpy.init()
    node = MarlDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
