"""Check that the RoboCasa sims stop at start, with one message, without a GPU to render on through EGL.

sim_common.egl_renderer() is the probe and sim_common.require_gpu() the stop. The probe is made
to fail three ways: a failing probe passed in, MuJoCo not on EGL (MUJOCO_GL=osmesa) and Mesa's
software EGL in place of NVIDIA's, the last two in h12_mujoco.py run as a subprocess for each
robot. Run it in the robocasa image, on its GPU:
ROS_DOMAIN_ID=231 tests/robocasa/run_in_container.sh -p test_gpu_required.py
"""
import os
import subprocess
import sys
import unittest

import sim_common

SIM = os.path.join(os.path.dirname(os.path.abspath(sim_common.__file__)), "h12_mujoco.py")
# glvnd's vendor file for Mesa's EGL: with only it listed, EGL renders on Mesa's software
# rasteriser even where NVIDIA's EGL is installed.
MESA_EGL_VENDOR = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
START_TIMEOUT_S = 120.0   # wall s: the refusal comes before either robot's kitchen is built


class RequireGpuTests(unittest.TestCase):
    """Check require_gpu() with a probe passed in."""

    def test_a_failing_probe_stops_the_sim_with_the_message(self):
        """A probe that raises stops the sim with GPU_REQUIRED and the probe's reason."""
        def probe():
            """Fail as the probe does without an EGL device."""
            raise RuntimeError("no EGL device: EGL_NOT_INITIALIZED")

        with self.assertRaises(SystemExit) as caught:
            sim_common.require_gpu(probe)
        self.assertEqual(caught.exception.code,
                         f"{sim_common.GPU_REQUIRED}: no EGL device: EGL_NOT_INITIALIZED")
        self.assertTrue(sim_common.GPU_REQUIRED.startswith("the RoboCasa sims need an NVIDIA GPU"))

    def test_a_working_probe_returns_its_renderer(self):
        """A probe that finds a GPU gives its GL renderer back, and the sim goes on."""
        self.assertEqual(sim_common.require_gpu(lambda: "NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2"),
                         "NVIDIA GeForce RTX 5070 Ti/PCIe/SSE2")

    def test_the_probe_refuses_mujoco_off_egl(self):
        """The probe raises, naming MUJOCO_GL, when MuJoCo is not set to render through EGL."""
        for value in ("osmesa", "glfw", ""):
            with self.subTest(MUJOCO_GL=value):
                with self.assertRaisesRegex(RuntimeError, "MUJOCO_GL"):
                    sim_common.egl_renderer({"MUJOCO_GL": value})


class SimStartTests(unittest.TestCase):
    """Run each robot's sim with the probe made to fail and check it stops at start."""

    def start(self, robot, **env):
        """Run h12_mujoco.py for robot, headless in the Blank task, with env added; return the CompletedProcess."""
        r = subprocess.run([sys.executable, "-u", SIM, f"--{robot}", "--headless", "--task", "Blank"],
                           capture_output=True, text=True, env=dict(os.environ, **env),
                           timeout=START_TIMEOUT_S, cwd=os.path.dirname(SIM))
        print(f"\n  {robot}: exit {r.returncode}: {r.stderr.strip().splitlines()[-1:] or r.stdout[-300:]}")
        return r

    def assert_refused(self, r, reason):
        """Assert the sim exited 1 with GPU_REQUIRED and reason on its last stderr line, before building a scene."""
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        self.assertEqual(r.returncode, 1, (r.stdout + r.stderr)[-2000:])
        self.assertTrue(last.startswith(f"{sim_common.GPU_REQUIRED}: "), last)
        self.assertIn(reason, last)
        self.assertNotIn("RoboCasa task", r.stdout)

    def test_without_egl_both_sims_stop_at_start(self):
        """MUJOCO_GL=osmesa stops either robot's sim at start, naming MUJOCO_GL."""
        for robot in ("h12", "g1"):
            with self.subTest(robot=robot):
                self.assert_refused(self.start(robot, MUJOCO_GL="osmesa"), "MUJOCO_GL is 'osmesa'")

    def test_on_mesas_software_egl_both_sims_stop_at_start(self):
        """With only Mesa's EGL, which renders in software, either robot's sim stops at start."""
        if not os.path.exists(MESA_EGL_VENDOR):
            self.skipTest(f"no {MESA_EGL_VENDOR}")
        for robot in ("h12", "g1"):
            with self.subTest(robot=robot):
                r = self.start(robot, MUJOCO_GL="egl", __EGL_VENDOR_LIBRARY_FILENAMES=MESA_EGL_VENDOR)
                # Mesa either gives a software renderer or no EGL device at all.
                self.assert_refused(r, "")


if __name__ == "__main__":
    unittest.main()
