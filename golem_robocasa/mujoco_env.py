"""Model the elastic band both sim runners hang the robot from until a policy drives it, and log a fall.

The band is a spring-damper pulling a body (the torso), or a hook point on
it, toward an anchor above its spawn. golem_sim.sim_loop applies it every
physics step while it is enabled (apply_band); SPACE in the viewer or
/elastic_band/toggle (MeasurementBridge) releases it, but not while BandLowering
is still bringing the robot down: that release takes effect once the lowering
has ended (ElasticBand.toggle). BandLowering moves the
anchor as the training env's hoist does: up until the robot hangs, then down
onto its feet once the legs (on the G1 then the waist too) are held
(golem_sim.py --no_band_lowering keeps it fixed). FallLogger logs the
band's release and the robot's first fall, which the closed-loop tests read.
"""
import threading

import mujoco
import numpy as np


class ElasticBand:
    """Compute the band's force on the body it holds, and toggle it."""

    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100, rope=False,
                 hook=None):
        """Anchor the band at point (m, world), enabled.

        length is the spring's rest length (m), stiffness N/m, damping N*s/m on
        the hooked point's whole linear velocity. A rope band never pulls the
        body down: its vertical force is clamped at 0, as a hoist's rope goes
        slack. hook is the point (m, the body's frame) the band pulls, which
        apply_band then also turns by its torque; None pulls the body's origin
        through its centre of mass, with no torque.
        """
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.rope = rope
        self.hook = None if hook is None else np.asarray(hook, dtype=np.float64)
        self.enabled = True
        self._torque = False   # apply_band wrote a torque it must clear at the release
        # True while a BandLowering brings the robot down (a stage under way); a release asked
        # for then waits for the lowering to end, release_pending set meanwhile
        self.lowering = False
        self.release_pending = False
        # toggle() runs on the service's and the viewer's threads, set_lowering() on the physics
        # loop's
        self._lock = threading.Lock()
        self._released = threading.Event()   # set while the band is released

    def hook_point(self, data, body):
        """Return the world point (m) the band pulls on body in data: its hook, or the body's origin."""
        if self.hook is None:
            return data.xpos[body]
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ self.hook

    def evaluate_force(self, x, v):
        """Return the world-frame force (N) on the body at position x moving at velocity v.

        Damping acts on the whole velocity, not only along the band: with
        length 0 the spring pulls in every direction, and radial-only damping
        would leave sideways and fore-aft swing undamped.
        """
        offset = x - self.point
        displacement = np.linalg.norm(offset)
        if displacement > 1e-9:
            spring = -self.stiffness * (displacement - self.length) * offset / displacement
        else:
            spring = np.zeros(3)
        force = spring - self.damping * v
        if self.rope:
            force[2] = max(force[2], 0.0)
        return force

    def toggle(self) -> bool:
        """Enable or disable the band, log it, and return whether it is now enabled.

        A release asked for while the band is lowering the robot (lowering) does not let go:
        the band stays enabled, release_pending is set, and set_lowering(False) releases it
        once the lowering has ended. A toggle while that release waits changes nothing.
        """
        with self._lock:
            if self.enabled and self.lowering:
                if not self.release_pending:
                    self.release_pending = True
                    print("Elastic band release requested while it lowers the robot; it lets go once lowered",
                          flush=True)
                return True
            self._set(not self.enabled)
            return self.enabled

    def set_lowering(self, lowering: bool):
        """Note whether a BandLowering is bringing the robot down; at its end, make a waiting release."""
        with self._lock:
            self.lowering = lowering
            if not lowering and self.release_pending:
                self._set(False)

    def wait_released(self, timeout: float) -> bool:
        """Wait up to timeout s (wall) for the band to be released; return whether it is."""
        return self._released.wait(timeout)

    def _set(self, enabled: bool):
        """Enable or disable the band and log it; the caller holds _lock."""
        self.enabled = enabled
        self.release_pending = False
        if enabled:
            self._released.clear()
        else:
            self._released.set()
        print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")

    # GLFW_KEY_SPACE: mujoco.viewer passes raw GLFW key codes, and mujoco.glfw
    # is not importable as an attribute in mujoco 3.3.
    _KEY_SPACE = 32

    def key_callback(self, key):
        """Toggle the band on SPACE (the passive viewer's key_callback)."""
        if key == self._KEY_SPACE:
            self.toggle()


def apply_band(model, data, band, body, vel):
    """Write the band's force on body into xfrc_applied for the next physics step.

    Its pull while enabled, else zero: MuJoCo keeps xfrc_applied between steps,
    so call it before every mj_step. A hooked band (ElasticBand.hook) also
    writes the pull's torque about the body's centre of mass, and clears it
    once at the release. vel is a 6-vector scratch buffer.
    """
    if band.enabled:
        # [angular(3), linear(3)] of the body at its centre of mass, world frame
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        if band.hook is None:
            data.xfrc_applied[body, :3] = band.evaluate_force(data.xpos[body], vel[3:])
            return
        arm = band.hook_point(data, body) - data.xipos[body]   # m, world: centre of mass -> hook
        force = band.evaluate_force(data.xipos[body] + arm, vel[3:] + np.cross(vel[:3], arm))
        data.xfrc_applied[body, :3] = force
        data.xfrc_applied[body, 3:] = np.cross(arm, force)
        band._torque = True
    else:
        data.xfrc_applied[body, :3] = 0.0
        if band._torque:
            data.xfrc_applied[body, 3:] = 0.0
            band._torque = False


class BandLowering:
    """Move the band's anchor as the training env's hoist does: hang the robot, then lower it onto its feet.

    From the first step the anchor rises lift m, linearly over lift_duration s
    of sim time, so the robot hangs with its feet clear of the floor. Then each
    of stages, a (slots, drop, delay, duration) in turn, lowers it: once every
    motor in slots is on with kp > 0 in the robot's latched rt/lowcmd, it waits
    delay s, then comes down drop m over duration s (not before the stage
    before it has ended), and stays there until the band's toggle releases it.
    While a stage is under way (from its motors' hold, its delay included, to its
    end) the band is lowering (ElasticBand.set_lowering): a release asked for then
    waits for the stage's end, or the end of the stage queued after it. With no
    stage under way (before the first is held, between one's end and the next's
    hold, after the last) a release takes effect at once.
    The band is a rope (ElasticBand.rope), so an anchor below the hook holds
    the robot sideways and does not pull it down: the feet carry the weight. A
    release stops it for good. It moves band.point in place, so the viewer's
    overlay follows.
    """

    def __init__(self, band, lift, lift_duration, stages):
        """Move band's anchor (an ElasticBand at the robot's spawn); lift m over lift_duration s, then stages."""
        self.band = band
        self.lift, self.lift_duration = lift, lift_duration
        self.stages = [(np.asarray(slots, dtype=int), drop, delay, duration)
                       for slots, drop, delay, duration in stages]
        self.drop = sum(stage[1] for stage in self.stages)   # m, all the stages
        self.base = float(band.point[2])   # m, world: the anchor at the spawn
        self.t0 = None      # s of sim time of the first step
        self.starts = []    # s of sim time each stage started down, as their slots are held in turn
        self.done = False

    def step(self, t, latch):
        """Place the anchor for sim time t (s); latch is the robot's SimInterface (unitree_common.LowCmdLatch)."""
        if self.done:
            return
        if not self.band.enabled:
            self.done = True
            print(f"[elastic-band] t={t:.2f}s: released with the anchor at z {self.band.point[2]:.3f} m",
                  flush=True)
            return
        if self.t0 is None:
            self.t0 = t
        if len(self.starts) < len(self.stages):
            slots, drop, delay, duration = self.stages[len(self.starts)]
            if latch.holds(slots):
                after = self.starts[-1] + self.stages[len(self.starts) - 1][3] if self.starts else t
                self.starts.append(max(t + delay, after))
                print(f"[elastic-band] t={t:.2f}s: motors {slots[0]}-{slots[-1]} are held; the anchor comes down "
                      f"{drop:.3f} m over {duration:.1f}s from t={self.starts[-1]:.2f}s", flush=True)
        z = self.base + self.lift * min(1.0, (t - self.t0) / self.lift_duration)
        for start, (_, drop, _, duration) in zip(self.starts, self.stages):
            z -= drop * min(1.0, max(0.0, (t - start) / duration))
        self.band.point[2] = z
        # under way until the last scheduled stage ends
        self.band.set_lowering(bool(self.starts) and t < self.starts[-1] + self.stages[len(self.starts) - 1][3])
        if len(self.starts) == len(self.stages) and t >= self.starts[-1] + self.stages[-1][3]:
            self.done = True
            print(f"[elastic-band] t={t:.2f}s: anchor lowered to z {z:.3f} m; "
                  "it stays there until the band is released", flush=True)


class FallLogger:
    """Log the band's release once, and once the robot's first fall, with how long it stood after the release.

    The robot has fallen when body's z axis . world up drops below upright_min
    or its height below height_min (m); label names the body's height in the
    log line.
    """

    def __init__(self, body, label, upright_min, height_min):
        """Watch body (an id; -1 watches no fall)."""
        self.body, self.label = body, label
        self.upright_min, self.height_min = upright_min, height_min
        self.release_t = None   # s of sim time, once the band is released
        self.fell = False

    def step(self, data, band):
        """Check the state mj_step just produced; band is the sim's ElasticBand or None."""
        if band is not None and self.release_t is None and not band.enabled:
            self.release_t = float(data.time)
            print(f"[fall-logger] elastic band released at t={data.time:.2f}s", flush=True)
        if self.body < 0 or self.fell:
            return
        upz = float(data.xmat[self.body][8])   # body z-axis . world up (1: upright)
        height = float(data.xpos[self.body][2])
        if upz < self.upright_min or height < self.height_min:
            self.fell = True
            since = (f"{data.time - self.release_t:.2f}s after band release" if self.release_t is not None
                     else f"{data.time:.2f}s (band still on)")
            print(f"[fall-logger] *** ROBOT FELL at sim t={data.time:.2f}s — stood {since} "
                  f"(uprightness={upz:.2f}, {self.label}_h={height:.2f}) ***", flush=True)


def draw_band(handle, anchor, body_pos):
    """Draw the band in the passive viewer: a sphere at anchor and a capsule to body_pos.

    Replaces whatever the viewer's user scene held.
    """
    scn = handle.user_scn
    scn.ngeom = 0
    color = np.array([0.5, 0.6, 1.0, 0.8])
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.02, 0.0, 0.0]), anchor, np.eye(3).flatten(), color)
        scn.ngeom += 1
    if scn.ngeom < scn.maxgeom:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(), color)
        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.01, anchor, body_pos)
        scn.ngeom += 1
