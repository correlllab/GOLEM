"""Serve the H1-2 sim's MAGPIE grippers on the real magpie_control driver's ROS surface.

One MagpieHandBridge serves one gripper. h12_mujoco.py spins one per side on
its background MultiThreadedExecutor, and its physics loop calls tick() after
every step; every MjData or MjModel access holds the sim lock, since mj_step
runs on the main thread. The state rate and the DeliGrasp settles count sim
time, so they keep the real driver's rates per robot second at any
GOLEM_SIM_RATE.

Per side, as the real driver (magpie_control's gripper_node.py, launched in the
<side> namespace) names them; the real driver's gripper/reset_overload service
is not simulated:
  /<side>/gripper/state                       magpie_msgs/msg/GripperState (10 Hz of sim time)
  /<side>/gripper/open                        std_srvs/srv/Trigger
  /<side>/gripper/close                       std_srvs/srv/Trigger
  /<side>/gripper/set_position                magpie_msgs/srv/SetGripperPosition
  /<side>/gripper/set_force                   magpie_msgs/srv/SetGripperForce
  /<side>/gripper/calibrate                   std_srvs/srv/Trigger
  /<side>/gripper/reset_parameters            std_srvs/srv/Trigger
  /<side>/gripper/deligrasp                   magpie_msgs/action/DeliGrasp
"""
import math
import threading
import time

import mujoco
import numpy as np
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from magpie_msgs.action import DeliGrasp
    from magpie_msgs.msg import GripperState
    from magpie_msgs.srv import SetGripperForce, SetGripperPosition
except ModuleNotFoundError as exc:
    if exc.name == "magpie_msgs":
        raise ModuleNotFoundError(
            "magpie_msgs is not available in this shell. Source the workspace "
            "before launching the sim, e.g. "
            "`source /opt/ros/humble/setup.bash && "
            "source ~/GOLEM/core_ws/install/setup.bash`."
        ) from exc
    raise


# Aperture: a linear map between the MJCF finger_joint1 angle (rad) and the
# reported aperture (mm), spanning the real MAGPIE's ~0-110 mm. 0 rad is open,
# CLOSED_RAD (finger_joint1's upper limit in the gripper MJCF) tips touching.
OPEN_MM = 110.0
CLOSED_MM = 0.0
CLOSED_RAD = 2.05

# Force: set_force bounds each finger's crank torque (actuator forcerange) at
# max_force * FINGER_LEVER_M, so the fingers squeeze with at most that pad
# normal force, as the real MAGPIE's force-limited close does. FINGER_LEVER_M
# (m) is the crank torque per newton of pad force through the 4-bar, measured
# on the robosuite gripper model closing on a fixed probe: within +/-4% over
# 12-60 mm openings. The same lever turns the measured crank torque back into
# the reported force.
FINGER_LEVER_M = 0.044
# N, MAGPIE's maximum grasp force (two AX-12A servos); requests are clipped to it.
MAX_FORCE_N = 32.0
# s of wall time between looks at the sim clock while a DeliGrasp settle waits.
SIM_POLL_S = 0.005
# s of sim time: a state publish due within this of the sim time is published now.
_EPS_T = 1e-9


def mm_to_rad(mm: float) -> float:
    """Return the finger_joint1 angle (rad) of an aperture (mm), clipped to the range."""
    f = (OPEN_MM - mm) / (OPEN_MM - CLOSED_MM)
    return float(np.clip(f, 0.0, 1.0)) * CLOSED_RAD


def rad_to_mm(rad: float) -> float:
    """Return the aperture (mm) of a finger_joint1 angle (rad), clipped to the range."""
    f = float(np.clip(rad, 0.0, CLOSED_RAD)) / CLOSED_RAD
    return OPEN_MM - f * (OPEN_MM - CLOSED_MM)


class MagpieHandBridge(Node):
    """Serve one simulated MAGPIE gripper's state, services and DeliGrasp action."""

    # Actuator and sensor names inside each gripper MJCF, under the
    # gripper0_<side>_ name_prefix.
    _NAMES = {
        "act_left":  "left_finger_actuator",
        "act_right": "right_finger_actuator",
        "pos_left":  "left_finger_pos",
        "pos_right": "right_finger_pos",
        "vel_left":  "left_finger_vel",
        "vel_right": "right_finger_vel",
        "trq_left":  "left_finger_torque",
        "trq_right": "right_finger_torque",
    }

    def __init__(self, model, data, side: str, sim_lock: threading.Lock,
                 publish_rate_hz: float = 10.0, name_prefix=""):
        """Open the gripper and serve it on /<side>/gripper/*, state at publish_rate_hz of sim time (tick()).

        Raises for a side other than left or right, or a missing actuator or sensor.
        """
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        super().__init__(f"magpie_hand_{side}")

        self.model = model
        self.data = data
        self.sim_lock = sim_lock
        self.side = side
        names = {k: name_prefix + v for k, v in self._NAMES.items()}

        self.act_left  = self._actuator_id(names["act_left"])
        self.act_right = self._actuator_id(names["act_right"])
        self.pos_left  = self._sensor_adr(names["pos_left"])
        self.pos_right = self._sensor_adr(names["pos_right"])
        self.vel_left  = self._sensor_adr(names["vel_left"])
        self.vel_right = self._sensor_adr(names["vel_right"])
        self.trq_left  = self._sensor_adr(names["trq_left"])
        self.trq_right = self._sensor_adr(names["trq_right"])

        # Open at boot, as the real driver is, with the force limit at MAGPIE's
        # maximum until set_force lowers it.
        with self.sim_lock:
            self._set_force_limit_locked(MAX_FORCE_N)
            self._write_aperture_rad_locked(0.0)

        # Reentrant, so the DeliGrasp loop, which sleeps between iterations,
        # does not block the other callbacks.
        self._cb_group = ReentrantCallbackGroup()

        # The real driver's names are gripper/... under a <side> namespace; this
        # node has none, so the side is part of each name.
        ns = f"{side}/gripper"
        self.pub_state = self.create_publisher(GripperState, f"{ns}/state", 10)
        self._publish_period = 1.0 / publish_rate_hz   # s of sim time
        self._next_state_t = -math.inf   # sim time the next state publish is due

        self.create_service(Trigger, f"{ns}/open", self._on_open,
                            callback_group=self._cb_group)
        self.create_service(Trigger, f"{ns}/close", self._on_close,
                            callback_group=self._cb_group)
        self.create_service(SetGripperPosition, f"{ns}/set_position",
                            self._on_set_position, callback_group=self._cb_group)
        self.create_service(SetGripperForce, f"{ns}/set_force",
                            self._on_set_force, callback_group=self._cb_group)
        self.create_service(Trigger, f"{ns}/calibrate", self._on_calibrate,
                            callback_group=self._cb_group)
        self.create_service(Trigger, f"{ns}/reset_parameters",
                            self._on_reset_parameters, callback_group=self._cb_group)

        self.action_server = ActionServer(
            self, DeliGrasp, f"{ns}/deligrasp",
            execute_callback=self._on_deligrasp,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"MagpieHandBridge[{side}] up: /{ns}/"
            "{state,open,close,set_position,set_force,calibrate,reset_parameters,deligrasp}")

    def _actuator_id(self, name: str) -> int:
        """Return the named actuator's id, raising RuntimeError if it is missing."""
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"actuator '{name}' not found in MJCF")
        return int(aid)

    def _sensor_adr(self, name: str) -> int:
        """Return the named sensor's sensordata address, raising RuntimeError if it is missing."""
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"sensor '{name}' not found in MJCF")
        return int(self.model.sensor_adr[sid])

    def _write_aperture_rad_locked(self, target_rad: float) -> None:
        """Command both fingers to target_rad, mirrored; the caller holds sim_lock.

        left_finger_actuator closes positive (ctrlrange [-0.035, 2.70]),
        right_finger_actuator negative ([-2.70, 0.035]).
        """
        lo_l, hi_l = self.model.actuator_ctrlrange[self.act_left]
        lo_r, hi_r = self.model.actuator_ctrlrange[self.act_right]
        self.data.ctrl[self.act_left]  = float(np.clip( target_rad, lo_l, hi_l))
        self.data.ctrl[self.act_right] = float(np.clip(-target_rad, lo_r, hi_r))

    def _read_aperture_mm_locked(self) -> tuple[float, float]:
        """Return (mean aperture mm, [right mm, left mm]); the caller holds sim_lock.

        finger_positions is [right, left], as the real driver publishes it.
        """
        rad_left  = float(self.data.sensordata[self.pos_left])
        rad_right = float(self.data.sensordata[self.pos_right])
        mm_left  = rad_to_mm( rad_left)
        mm_right = rad_to_mm(-rad_right)
        return 0.5 * (mm_left + mm_right), [mm_right, mm_left]

    def _read_force_n_locked(self) -> float:
        """Return the mean pad force (N) of the two fingers; the caller holds sim_lock."""
        torque_l = float(self.data.sensordata[self.trq_left])
        torque_r = float(self.data.sensordata[self.trq_right])
        return float(0.5 * (abs(torque_l) + abs(torque_r)) / FINGER_LEVER_M)

    def _read_vel_locked(self) -> float:
        """Return the faster finger's crank speed (rad/s); the caller holds sim_lock."""
        v_l = float(self.data.sensordata[self.vel_left])
        v_r = float(self.data.sensordata[self.vel_right])
        return max(abs(v_l), abs(v_r))

    def _set_force_limit_locked(self, force_n: float) -> float:
        """Limit each pad to force_n (N, clipped to [0, MAX_FORCE_N]) and return the limit applied.

        The caller holds sim_lock.
        """
        force_n = float(np.clip(force_n, 0.0, MAX_FORCE_N))
        tau = force_n * FINGER_LEVER_M
        for act in (self.act_left, self.act_right):
            self.model.actuator_forcelimited[act] = 1
            self.model.actuator_forcerange[act] = [-tau, tau]
        return force_n

    def tick(self, sim_t: float) -> None:
        """Publish /<side>/gripper/state when it is due at sim time sim_t (s): publish_rate_hz of sim time.

        Call it from the physics loop after each step, without holding sim_lock. After a jump
        in sim time the next publish is a period after sim_t, not a burst of the ones missed.
        """
        if sim_t < self._next_state_t - _EPS_T:
            return
        self._next_state_t += self._publish_period
        if self._next_state_t <= sim_t:
            self._next_state_t = sim_t + self._publish_period
        self._publish_state()

    def _sleep_sim(self, seconds: float, sleep=time.sleep) -> None:
        """Return once seconds of sim time have passed, looking at the sim clock every SIM_POLL_S of wall time."""
        with self.sim_lock:
            end = float(self.data.time) + seconds
        while True:
            with self.sim_lock:
                if float(self.data.time) >= end:
                    return
            sleep(SIM_POLL_S)

    def _publish_state(self) -> None:
        """Publish /<side>/gripper/state."""
        try:
            with self.sim_lock:
                aperture_mm, finger_positions = self._read_aperture_mm_locked()
                force_n  = self._read_force_n_locked()
                vel_max  = self._read_vel_locked()
            msg = GripperState()
            msg.position = float(aperture_mm)
            msg.force = float(force_n)
            msg.temperature = 25.0  # deg C; the sim has no thermal model
            msg.is_moving = bool(vel_max > 0.01)
            msg.contact_detected = bool(force_n > 0.5)
            msg.finger_positions = [float(p) for p in finger_positions]
            self.pub_state.publish(msg)
        except Exception as e:
            self.get_logger().warning(f"state publish failed: {e}")

    def _on_open(self, request, response):
        """Open the gripper fully."""
        with self.sim_lock:
            self._write_aperture_rad_locked(0.0)
        response.success = True
        response.message = f"{self.side} hand opening"
        return response

    def _on_close(self, request, response):
        """Close the gripper fully, within the force limit."""
        with self.sim_lock:
            self._write_aperture_rad_locked(CLOSED_RAD)
        response.success = True
        response.message = f"{self.side} hand closing"
        return response

    def _on_set_position(self, request, response):
        """Command the aperture request.position (mm); the sim ignores request.speed."""
        try:
            target_mm = float(request.position)
            target_rad = mm_to_rad(target_mm)
            with self.sim_lock:
                self._write_aperture_rad_locked(target_rad)
                actual_mm, _ = self._read_aperture_mm_locked()
            response.success = True
            response.actual_position = float(actual_mm)
            response.message = (
                f"{self.side} target {target_mm:.1f} mm "
                f"({target_rad:.3f} rad); actual {actual_mm:.1f} mm "
                f"(speed {request.speed:.2f} ignored in sim)"
            )
        except Exception as e:
            response.success = False
            response.actual_position = 0.0
            response.message = f"set_position failed: {e}"
        return response

    def _on_set_force(self, request, response):
        """Set the force limit to request.max_force (N), clipped to MAX_FORCE_N."""
        try:
            with self.sim_lock:
                applied = self._set_force_limit_locked(request.max_force)
            response.success = True
            response.message = (
                f"{self.side} force limit {applied:.2f} N "
                f"(requested {request.max_force:.2f} N, max {MAX_FORCE_N:.0f} N)")
        except Exception as e:
            response.success = False
            response.message = f"set_force failed: {e}"
        return response

    def _on_calibrate(self, request, response):
        """Succeed without doing anything: the sim gripper needs no calibration."""
        response.success = True
        response.message = f"{self.side} hand: sim calibration is implicit"
        return response

    def _on_reset_parameters(self, request, response):
        """Restore the force limit to MAX_FORCE_N."""
        try:
            with self.sim_lock:
                self._set_force_limit_locked(MAX_FORCE_N)
            response.success = True
            response.message = f"{self.side} force limit restored to {MAX_FORCE_N:.0f} N"
        except Exception as e:
            response.success = False
            response.message = f"reset_parameters failed: {e}"
        return response

    def _on_deligrasp(self, goal_handle):
        """Run a DeliGrasp goal: go to goal_aperture, then close until the force target is met.

        After each settle of SETTLE_S of sim time, while the measured force is
        below the target, it closes additional_closure mm further and raises
        the force limit by additional_force N, at most MAX_ITERS times. Stops
        on reaching the force, on cancellation, or after MAX_ITERS. The goal
        and result fields are the real driver's.
        """
        params = goal_handle.request.params
        feedback = DeliGrasp.Feedback()
        force_log: list[float] = []

        MAX_ITERS = 8
        SETTLE_S = 0.20

        try:
            current_aperture = float(params.goal_aperture)
            current_force_target = float(params.initial_force)

            feedback.phase = "approach"
            feedback.current_aperture = current_aperture
            feedback.current_force = 0.0
            goal_handle.publish_feedback(feedback)

            with self.sim_lock:
                self._set_force_limit_locked(current_force_target)
                self._write_aperture_rad_locked(mm_to_rad(current_aperture))

            self._sleep_sim(SETTLE_S)

            feedback.phase = "initial_close"
            goal_handle.publish_feedback(feedback)

            for it in range(MAX_ITERS):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = DeliGrasp.Result()
                    result.success = False
                    result.message = "cancelled"
                    result.final_aperture, _ = self._read_state_snapshot()
                    result.final_force = 0.0
                    result.force_log = force_log
                    return result

                with self.sim_lock:
                    current_force = self._read_force_n_locked()
                    actual_mm, _ = self._read_aperture_mm_locked()
                force_log.append(current_force)

                feedback.current_aperture = float(actual_mm)
                feedback.current_force = float(current_force)
                goal_handle.publish_feedback(feedback)

                if current_force >= current_force_target:
                    break

                current_aperture = max(
                    0.0, current_aperture - float(params.additional_closure))
                current_force_target += float(params.additional_force)
                feedback.phase = (
                    "additional_close" if it > 0 else "force_increase")
                feedback.current_aperture = current_aperture
                goal_handle.publish_feedback(feedback)

                with self.sim_lock:
                    self._set_force_limit_locked(current_force_target)
                    self._write_aperture_rad_locked(mm_to_rad(current_aperture))
                self._sleep_sim(SETTLE_S)

            if params.complete_grasp:
                feedback.phase = "complete"
                goal_handle.publish_feedback(feedback)

            with self.sim_lock:
                final_force = self._read_force_n_locked()
                final_aperture, _ = self._read_aperture_mm_locked()

            result = DeliGrasp.Result()
            result.success = True
            result.message = (
                f"deligrasp done: aperture {final_aperture:.1f} mm, "
                f"force {final_force:.2f} N over {len(force_log)} iters"
            )
            result.final_aperture = float(final_aperture)
            result.final_force = float(final_force)
            result.force_log = force_log
            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f"deligrasp failed: {e}")
            result = DeliGrasp.Result()
            result.success = False
            result.message = f"deligrasp failed: {e}"
            result.final_aperture = 0.0
            result.final_force = 0.0
            result.force_log = force_log
            goal_handle.abort()
            return result

    def _read_state_snapshot(self) -> tuple[float, float]:
        """Return (aperture mm, force N), taking sim_lock."""
        with self.sim_lock:
            aperture_mm, _ = self._read_aperture_mm_locked()
            force_n = self._read_force_n_locked()
        return aperture_mm, force_n
