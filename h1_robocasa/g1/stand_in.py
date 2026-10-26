"""The sim's stand-in for Unitree's controller: HOMIE keeps the G1 standing.

On the real G1, Unitree's controller balances the robot and holds the waist and
arms until a program takes over: rt/arm_sdk shares the waist and arms with it,
and rt/lowcmd takes every motor once Unitree's controller is released. The sim
has no Unitree controller, so HOMIE's walking policy (g1_locomotion's policy/) stands in
for it from the first physics step: it runs at 50 Hz sim time with the standing
command (0, 0, 0) at 0.74 m and drives the legs, while the waist and arms are
held at 0. Both use HOMIE's training gains, as in mujoco_deploy_g1.py.
"""
import os
import sys

import numpy as np

# HOMIE's policy is g1_locomotion's policy/ (core_ws/src/g1/g1_locomotion/policy),
# not a Python package; the containers mount it read-only next to this
# directory, at /home/code/g1_locomotion
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POLICY_DIRS = (os.path.join(_ROOT, "g1_locomotion"),
                os.path.join(_ROOT, "core_ws", "src", "g1", "g1_locomotion", "policy"))
POLICY_DIR = next((d for d in _POLICY_DIRS if os.path.isdir(d)), _POLICY_DIRS[0])
sys.path.insert(0, POLICY_DIR)
from homie_policy import HomiePolicy  # noqa: E402


class StandIn:
    """HOMIE on the legs and the waist and arms held at 0, for the body motors
    named by ``names`` (any order; the sim passes the 29 in DDS slot order).

    torques() is the PD torque for the current targets; call it before every
    physics step. Call after_step() after every physics step: every
    decimation-th call (50 Hz at dt 0.002) runs the policy on the state after
    that step and updates the leg targets, as mujoco_deploy_g1.py does.
    """

    def __init__(self, names, dt, cmd=(0.0, 0.0, 0.0), height=None):
        self.policy = HomiePolicy()
        config = self.policy.config
        if not np.isclose(dt, config["simulation_dt"]):
            raise ValueError(f"the policy needs dt {config['simulation_dt']}, the model has {dt}")
        self.decimation = int(config["control_decimation"])
        self.cmd, self.height = self.policy.clip_command(
            cmd, config["height_cmd"] if height is None else height)

        index = {name: i for i, name in enumerate(names)}
        missing = [n for n in self.policy.policy_joints if n not in index]
        if missing:
            raise KeyError(f"policy joints missing from the motors: {missing}")
        self._observed = np.array([index[n] for n in self.policy.policy_joints])
        self._legs = np.array([index[n] for n in self.policy.leg_joints])
        self.kp, self.kd = self.policy.gains(names)
        # the legs start at HOMIE's default angles; every other joint is held at 0
        self.target = np.zeros(len(names))
        self.target[self._legs] = self.policy.default_angles
        self._steps = 0
        self._warned = False

    def torques(self, q, dq):
        """PD torque on every motor toward the current targets (zero velocity)"""
        return self.kp * (self.target - q) - self.kd * dq

    def after_step(self, q, dq, quat_wxyz, gyro):
        """q, dq of the motors after the step, the pelvis orientation and its
        angular velocity in the pelvis frame (imu_in_pelvis has no rotation
        relative to the pelvis, so this is what the IMU measures)"""
        self._steps += 1
        if self._steps % self.decimation:
            return
        try:
            self.target[self._legs] = self.policy.step(
                self.cmd, self.height, gyro, quat_wxyz, q[self._observed], dq[self._observed])
        except ValueError as e:  # a state that is not finite: keep the last targets
            if not self._warned:
                self._warned = True
                print(f"[stand_in] policy step skipped, holding the last leg targets: {e}")
