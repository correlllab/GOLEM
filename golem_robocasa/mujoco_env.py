"""Model the elastic band both sim runners hang the robot from until a policy drives it, and log a fall.

The band is a spring-damper pulling a body (the torso) toward a fixed anchor
above its spawn. h12_mujoco.sim_loop and g1.g1_mujoco.sim_loop apply it every
physics step while it is enabled (apply_band); SPACE in the viewer or
/elastic_band/toggle (MeasurementBridge) releases it. FallLogger logs the
band's release and the robot's first fall, which the closed-loop tests read.
"""
import mujoco
import numpy as np


class ElasticBand:
    """Compute the band's force on the body it holds, and toggle it."""

    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        """Anchor the band at point (m, world), enabled.

        length is the spring's rest length (m), stiffness N/m, damping N*s/m on
        the body's whole linear velocity.
        """
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.enabled = True

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
        return spring - self.damping * v

    def toggle(self) -> bool:
        """Enable or disable the band, log it, and return whether it is now enabled."""
        self.enabled = not self.enabled
        print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

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
    so call it before every mj_step. vel is a 6-vector scratch buffer.
    """
    if band.enabled:
        # [angular(3), linear(3)] of the body, world frame
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
        data.xfrc_applied[body, :3] = band.evaluate_force(data.xpos[body], vel[3:])
    else:
        data.xfrc_applied[body, :3] = 0.0


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
