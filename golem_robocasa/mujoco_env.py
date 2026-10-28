"""Elastic-band balance tether used by both robots' sim runners
(h12_mujoco.sim_loop, g1.g1_mujoco.sim_loop).

A vertical spring-damper that pulls a body (the torso) toward a fixed anchor
point above its spawn, holding the free-floating biped upright until the ROS
walking policy takes over. Toggle with SPACE in the viewer or the
/elastic_band/toggle service; draw_band draws it in the viewer.

Optionally (rot_stiffness > 0) it also holds the body's orientation at
`orientation` with a rotational spring-damper, which makes the tether a rigid
hoist fixture: a linear-only band leaves yaw free, and the arm's reaction and
contact torques then spin the hanging robot.
"""
import mujoco
import numpy as np


class ElasticBand:
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100,
                 orientation=None, rot_stiffness=0.0, rot_damping=0.0):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness  # N/m
        self.damping = damping      # N*s/m, on the full linear velocity
        # Target orientation (wxyz quaternion, world) for the rotational hold.
        self.orientation = None if orientation is None else np.asarray(orientation, dtype=np.float64)
        self.rot_stiffness = float(rot_stiffness)   # N*m/rad
        self.rot_damping = float(rot_damping)       # N*m*s/rad
        self.enabled = True

    def evaluate_force(self, x, v):
        """World-frame force on the attached body at position x with linear velocity v.

        Damping acts on the whole velocity, not just its component along the band:
        with length=0 the spring pulls in every direction, so radial-only damping
        leaves lateral/fore-aft swing undamped.
        """
        offset = x - self.point
        displacement = np.linalg.norm(offset)
        if displacement > 1e-9:
            spring = -self.stiffness * (displacement - self.length) * offset / displacement
        else:
            spring = np.zeros(3)
        return spring - self.damping * v

    def evaluate_torque(self, quat, omega):
        """World-frame torque on the body with orientation `quat` (wxyz) and
        angular velocity `omega` (world), or zeros when the rotational hold is
        off."""
        if self.rot_stiffness <= 0.0 or self.orientation is None:
            return np.zeros(3)
        # q_err = q_target * conj(q): the world-frame rotation from q to target.
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
        self.enabled = not self.enabled
        print(f"Elastic band {'enabled' if self.enabled else 'disabled'}")
        return self.enabled

    # GLFW_KEY_SPACE; mujoco.viewer passes raw GLFW key codes, and the
    # mujoco.glfw submodule isn't importable as an attribute in mujoco 3.3.
    _KEY_SPACE = 32

    def key_callback(self, key):
        if key == self._KEY_SPACE:
            self.toggle()

    def print_instructions(self):
        print('Elastic Band Controls:')
        print('  SPACE - Toggle elastic band on/off')


def draw_band(handle, anchor, body_pos):
    """Draw the elastic band (anchor sphere + capsule to the torso at body_pos) in
    the passive viewer's user scene, replacing whatever the scene held."""
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
