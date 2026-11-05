"""Model the elastic band both sim runners hang the robot from until a policy drives it.

The band is a spring-damper pulling a body (the torso) toward a fixed anchor
above its spawn. h12_mujoco.sim_loop and g1.g1_mujoco.sim_loop apply it every
physics step while it is enabled; SPACE in the viewer or /elastic_band/toggle
(MeasurementBridge) releases it. With rot_stiffness > 0 it also holds the
body's orientation, which makes it a rigid hoist: a linear-only band leaves
yaw free, and the arms' reaction and contact torques then spin the hanging
robot.
"""
import mujoco
import numpy as np


class ElasticBand:
    """Compute the band's force and torque on the body it holds, and toggle it."""

    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100,
                 orientation=None, rot_stiffness=0.0, rot_damping=0.0):
        """Anchor the band at point (m, world), enabled.

        length is the spring's rest length (m), stiffness N/m, damping N*s/m on
        the body's whole linear velocity. orientation is the held orientation
        (wxyz, world), rot_stiffness N*m/rad and rot_damping N*m*s/rad; the
        rotational hold is off while rot_stiffness is 0 or orientation is None.
        """
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness
        self.damping = damping
        self.orientation = None if orientation is None else np.asarray(orientation, dtype=np.float64)
        self.rot_stiffness = float(rot_stiffness)
        self.rot_damping = float(rot_damping)
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

    def evaluate_torque(self, quat, omega):
        """Return the world-frame torque (N*m) holding the body at the band's orientation.

        quat is the body's orientation (wxyz, world), omega its angular velocity
        (rad/s, world). Zero while the rotational hold is off.
        """
        if self.rot_stiffness <= 0.0 or self.orientation is None:
            return np.zeros(3)
        # q_err = q_target * conj(q): the world-frame rotation from q to the target.
        err = np.zeros(4)
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, np.asarray(quat, dtype=np.float64))
        mujoco.mju_mulQuat(err, self.orientation, neg)
        if err[0] < 0.0:
            err = -err
        rotvec = np.zeros(3)
        mujoco.mju_quat2Vel(rotvec, err, 1.0)
        return self.rot_stiffness * rotvec - self.rot_damping * np.asarray(omega)

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
