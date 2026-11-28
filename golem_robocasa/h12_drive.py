"""The real H1-2's drive behaviour for the RoboCasa sim: hip-roll series elasticity and the drivers' velocity lag.

On by default (h12_robot.make_interface); GOLEM_H12_DRIVE_MODEL=0 gives the ideal motors of before.

Why: on the robot a hip roll follows its target about 0.05 at 10 Hz, 0.10-0.13 at 20 Hz (phase about -90 deg) and
0.03-0.08 at 30 Hz while walking, in sport mode and under the lmg_deploy policies alike, and the motor drivers report
the velocity of hip yaw/pitch/roll, knee and torso through a ~21 Hz 2nd-order low-pass (~12 ms late). A rigid joint
under the same PD follows 20 Hz targets 5-20x less, so a policy that damps hip roll on that lagged velocity rings at
20 Hz on the robot (lmg_deploy's Coop150k policy did, 2026-11-24) and, with ideal motors, not in the sim. The model:

  * each hip roll gets a rotor of inertia J on which the motor's PD acts, coupled to the link by a gear spring k and
    damper c; rt/lowstate reports the rotor (as the robot's encoder does), the link gets the spring torque:
        J th'' = tau_pd - k (th - q) - c (th' - q'),   tau_pd = clip(kp (q_cmd - th) + kd (dq_cmd - th') + tau, frc)
  * the latched rt/lowcmd reaches the motors DELAY_STEPS physics steps late (4 ms, fitted with the drive; the robot's
    ~6 ms from /lowcmd to the motor less the sim's own step);
  * rt/lowstate's dq of LAG_SLOTS goes through a 2nd-order Butterworth at LAG_HZ, sampled every physics step;
  * tau_est of the modelled joints is tau_pd, as the robot reports its PD's torque.

The parameters were fitted to the robot's bags (Locomanipulation_game evals/mujoco/drive20.json, robocasa plant: the
same joint classes as this MJCF) by matching the hip roll's command -> reported angle and the policy-rate target ->
reported velocity responses at 10-30 Hz: the sim's hip roll then follows 20 Hz targets 0.10 at -86 deg (the robot
0.11 at -94 deg; the rigid joint 0.004-0.03), and Coop150k rings there as on the robot (hip-roll 15-25 Hz targets 96
mrad, the robot's 78-99) while FastEnv60k does not (24, the robot's 18).
"""
import numpy as np

# body slots (sim_names.ROS_MOTOR_ORDER): the left and right hip rolls
DRIVE_SLOTS = (2, 8)
# the fitted rotor (kg m^2), gear spring (N m/rad) and its damping (N m s/rad); the joints' armature then the link's own
DRIVE_J, DRIVE_K, DRIVE_C = 0.08, 2000.0, 10.0
LINK_ARMATURE = 0.01
# physics steps from the latched rt/lowcmd to the motors. The fit (in the evals' MuJoCo, where the policy reads the
# state with no latency of its own) put 2 ms on the commands and 6 ms on the readings; here lmg_deploy's own reading
# (a /lowstate sample behind, then inference) and golem_safety's relay supply part of that, so 4 ms is added.
DELAY_STEPS = 2
# the drivers' reported-velocity low-pass: hip yaw/pitch/roll and knee of both legs, and the torso
LAG_SLOTS = (0, 1, 2, 3, 6, 7, 8, 9, 12)
LAG_HZ = 21.0
# rotor integration substeps per physics step (the spring never enters MuJoCo's integrator)
SUBSTEPS = 10


class VelocityLowpass:
    """A 2nd-order Butterworth per channel (lmg_deploy.livox_imu.GyroLowpass's filter, exact ZOH per dt)."""

    ZETA = 1.0 / np.sqrt(2.0)

    def __init__(self, cutoff_hz: float):
        """Set the cutoff (Hz)."""
        w = 2.0 * np.pi * float(cutoff_hz)
        self._a = np.array([[0.0, 1.0], [-w * w, -2.0 * self.ZETA * w]])
        self._b = np.array([0.0, w * w])
        self._sigma, self._wd, self._w2 = self.ZETA * w, w * np.sqrt(1.0 - self.ZETA ** 2), w * w
        self._x = None

    def reset(self):
        """Restart at the next sample."""
        self._x = None

    def __call__(self, u: np.ndarray, dt: float) -> np.ndarray:
        """Return the filtered value after sample u held over dt (s); the first sample starts the filter at rest on u."""
        u = np.asarray(u, dtype=np.float64)
        if self._x is None:
            self._x = np.stack([u, np.zeros_like(u)])
            return u.copy()
        e, c, sn = np.exp(-self._sigma * dt), np.cos(self._wd * dt), np.sin(self._wd * dt)
        k = self._sigma / self._wd
        ad = e * np.array([[c + k * sn, sn / self._wd], [-self._w2 * sn / self._wd, c - k * sn]])
        bd = np.linalg.solve(self._a, (ad - np.eye(2)) @ self._b)
        self._x = ad @ self._x + np.outer(bd, u)
        return self._x[0].copy()


class SeriesDrive:
    """The rotors of DRIVE_SLOTS."""

    def __init__(self, slots=DRIVE_SLOTS, J=DRIVE_J, k=DRIVE_K, c=DRIVE_C):
        """Set the rotors' parameters; reset() places them."""
        self.slots = np.asarray(slots, dtype=int)
        n = len(self.slots)
        self.J, self.k, self.c = np.full(n, float(J)), np.full(n, float(k)), np.full(n, float(c))
        self.th, self.thd, self.tau_pd = np.zeros(n), np.zeros(n), np.zeros(n)

    def reset(self, q: np.ndarray, dq: np.ndarray):
        """Place the rotors on the links (q, dq over every body slot), the springs unloaded."""
        self.th, self.thd = q[self.slots].astype(float).copy(), dq[self.slots].astype(float).copy()
        self.tau_pd = np.zeros(len(self.slots))

    def step(self, cmd, q: np.ndarray, dq: np.ndarray, frc: np.ndarray, dt: float) -> np.ndarray:
        """Advance the rotors over dt under cmd (a unitree_common.MotorCommand over every body slot, or None: motors
        off) with the links (q, dq) held; return the mean spring torque on the modelled links. frc is each slot's
        torque limit (N m), applied to the motor's PD."""
        s_ = self.slots
        ql, qdl, lim = q[s_], dq[s_], frc[s_]
        h = dt / SUBSTEPS
        spring = np.zeros(len(s_))
        for _ in range(SUBSTEPS):
            if cmd is None:
                self.tau_pd = np.zeros(len(s_))
            else:
                pd = cmd.kp[s_] * (cmd.q[s_] - self.th) + cmd.kd[s_] * (cmd.dq[s_] - self.thd) + cmd.tau[s_]
                self.tau_pd = np.where(cmd.on[s_], np.clip(pd, -lim, lim), 0.0)
            sp = self.k * (self.th - ql) + self.c * (self.thd - qdl)
            self.thd = self.thd + h * (self.tau_pd - sp) / self.J
            self.th = self.th + h * self.thd
            spring += sp
        return spring / SUBSTEPS


def _interface_class():
    """Return DriveSimInterface, imported late so this module's numbers load without the Unitree SDK."""
    from collections import deque

    import unitree_interface as ui

    class DriveSimInterface(ui.SimInterface):
        """SimInterface with the drive model: rt/lowcmd reaches the motors DELAY_STEPS steps late, the hip rolls are
        series-elastic (SeriesDrive) and rt/lowstate reports their rotors, LAG_SLOTS' dq comes through the drivers'
        low-pass, and the hip rolls' tau_est is their PD's torque."""

        def __init__(self, model, data, resolver=None):
            """Set up the drive on SimInterface's model: the hip rolls' armature moves to the rotor, and their
            actuatorfrcrange is widened so the spring torque reaches the link unclamped (the PD's is clamped)."""
            super().__init__(model, data, resolver=resolver)
            r = self.resolver
            joints = [int(np.nonzero(model.jnt_qposadr == a)[0][0]) for a in r.motor_qpos]
            self.frc = np.array([model.jnt_actfrcrange[j, 1] if model.jnt_actfrclimited[j] else np.inf for j in joints])
            self.drive = SeriesDrive()
            for s in self.drive.slots:
                model.jnt_actfrcrange[joints[s]] = (-1e4, 1e4)
                model.dof_armature[r.motor_qvel[s]] = LINK_ARMATURE
            self.lag = VelocityLowpass(LAG_HZ)
            self.lag_slots = np.asarray(LAG_SLOTS, dtype=int)
            # the delay line: the motors act on the oldest entry; empty slots are no command (motors off)
            self.cmds = deque([None] * DELAY_STEPS, maxlen=DELAY_STEPS + 1)
            self._placed = False
            print(f"[h12_drive] drive model on: hip rolls J {DRIVE_J} k {DRIVE_K} c {DRIVE_C}, rt/lowcmd "
                  f"{DELAY_STEPS} steps late, dq of slots {LAG_SLOTS} low-passed at {LAG_HZ} Hz")

        def reported(self):
            """Return what the encoders read: the links' q and dq, the rotors' on the drive's slots."""
            d, r = self.data, self.resolver
            q, dq = d.qpos[r.motor_qpos].copy(), d.qvel[r.motor_qvel].copy()
            if self._placed:
                q[self.drive.slots], dq[self.drive.slots] = self.drive.th, self.drive.thd
            return q, dq

        def write_ctrl(self):
            """Write the torques of the command DELAY_STEPS steps old: the PD on what the encoders read, the
            springs' on the hip rolls (advancing their rotors)."""
            d, r = self.data, self.resolver
            if not self._placed:
                self.drive.reset(d.qpos[r.motor_qpos], d.qvel[r.motor_qvel])
                self._placed = True
            q, dq = self.reported()
            cmd = None
            if self.commanded:
                self._lowcmd_torques(q, dq, float(d.time))   # the timeout's bookkeeping and log
                cmd = None if self._lowcmd_timed_out else self._lowcmd
            self.cmds.append(cmd)
            cmd = self.cmds[0]
            tau = cmd.torques(q, dq) if cmd is not None else np.zeros(len(q))
            tau[self.drive.slots] = self.drive.step(cmd, d.qpos[r.motor_qpos], d.qvel[r.motor_qvel], self.frc,
                                                    float(self.model.opt.timestep))
            d.ctrl[r.motor_ctrl] = tau

        def lowstate_motors(self):
            """Return what rt/lowstate reports, as the robot does: the hip rolls' rotors and PD torque, and LAG_SLOTS'
            dq through the drivers' low-pass (sampled here, once per physics step). SimInterface.after_step, which
            the sim tests' recorder wraps, publishes it."""
            q, dq = self.reported()
            dq[self.lag_slots] = self.lag(dq[self.lag_slots], float(self.model.opt.timestep))
            tau = self.data.sensordata[self.resolver.motor_tau].copy()
            tau[self.drive.slots] = self.drive.tau_pd
            return q, dq, tau

        def physics_reset(self):
            """Restamp the latch; the rotors, the delay line and the low-pass start over at the next step."""
            super().physics_reset()
            self._placed = False
            self.cmds.clear()
            self.cmds.extend([None] * DELAY_STEPS)
            self.lag.reset()

    return DriveSimInterface


def make_interface(model, data, resolver):
    """Return the H1-2's SimInterface with the drive model."""
    return _interface_class()(model, data, resolver=resolver)
