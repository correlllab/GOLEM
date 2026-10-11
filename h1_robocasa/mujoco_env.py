"""Elastic-band balance tether used by the sim runner (h12_mujoco.sim_loop).

A vertical spring-damper that pulls a body (the torso) toward a fixed anchor
point above its spawn, holding the free-floating biped upright until the ROS
walking policy takes over. Toggle with SPACE in the viewer or the
/elastic_band/toggle service.
"""
import mujoco
import numpy as np


class ElasticBand:
    def __init__(self, point=np.array([0, 0, 3]), length=0, stiffness=200, damping=100):
        self.point = point.astype(np.float64)
        self.length = length
        self.stiffness = stiffness  # N/m
        self.damping = damping      # N*s/m, on the full linear velocity
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
