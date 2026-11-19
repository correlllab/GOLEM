"""Check that the RoboCasa sims stop at start, with one message, without a GPU to render on through EGL.

GOLEM_RENDER=cpu (the Mac images) is the one way past the check: it renders through OSMesa
and casts the lidar on the CPU (RenderModeTests).

sim_common.egl_renderer() is the probe and sim_common.require_gpu() the stop. The probe is made
to fail three ways: a failing probe passed in, MuJoCo not on EGL (MUJOCO_GL=osmesa) and Mesa's
software EGL in place of NVIDIA's, the last two in h12_mujoco.py run as a subprocess for each
robot; with the probe failing, --help and a usage error still answer. Run it in the robocasa
image, on its GPU:
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


class RenderModeTests(unittest.TestCase):
    """Check GOLEM_RENDER: gpu (the default) keeps the GPU check; cpu needs OSMesa."""

    def test_unset_or_empty_is_gpu(self):
        """GOLEM_RENDER unset, empty or blank is gpu, the default."""
        for env in ({}, {"GOLEM_RENDER": ""}, {"GOLEM_RENDER": "  "}):
            with self.subTest(env=env):
                self.assertEqual(sim_common.render_mode(env), "gpu")

    def test_a_bad_value_stops_the_sim(self):
        """A GOLEM_RENDER other than gpu or cpu stops the sim, naming the knob."""
        with self.assertRaises(SystemExit) as caught:
            sim_common.render_mode({"GOLEM_RENDER": "metal"})
        self.assertIn("GOLEM_RENDER must be gpu or cpu", caught.exception.code)

    def test_gpu_mode_still_requires_the_gpu(self):
        """By default require_renderer() runs the GPU check, and a failing probe still stops the sim."""
        def probe():
            """Fail as the probe does with MuJoCo off EGL."""
            raise RuntimeError("MUJOCO_GL is 'osmesa', not 'egl'")

        with self.assertRaises(SystemExit) as caught:
            sim_common.require_renderer({"MUJOCO_GL": "osmesa"}, probe)
        self.assertTrue(caught.exception.code.startswith(sim_common.GPU_REQUIRED))

    def test_cpu_mode_needs_osmesa(self):
        """GOLEM_RENDER=cpu passes with MUJOCO_GL=osmesa and refuses another MUJOCO_GL."""
        def probe():
            """Fail: cpu mode must not probe the GPU."""
            raise AssertionError("probed the GPU")

        env = {"GOLEM_RENDER": "CPU", "MUJOCO_GL": "osmesa"}
        self.assertEqual(sim_common.require_renderer(env, probe), "cpu")
        with self.assertRaises(SystemExit) as caught:
            sim_common.require_renderer({"GOLEM_RENDER": "cpu", "MUJOCO_GL": "egl"}, probe)
        self.assertIn("MUJOCO_GL must be osmesa", caught.exception.code)


class SimStartTests(unittest.TestCase):
    """Run each robot's sim with the probe made to fail and check it stops at start."""

    def start(self, robot, *args, **env):
        """Run h12_mujoco.py for robot with args (default: headless in the Blank task) and env added.

        Returns the CompletedProcess.
        """
        args = args or ("--headless", "--task", "Blank")
        # GOLEM_RENDER is the test's own: the Mac images set cpu (docker-compose.mac.yml),
        # and these tests check the default, gpu.
        base = {k: v for k, v in os.environ.items() if k != "GOLEM_RENDER"}
        r = subprocess.run([sys.executable, "-u", SIM, f"--{robot}", *args],
                           capture_output=True, text=True, env=dict(base, **env),
                           timeout=START_TIMEOUT_S, cwd=os.path.dirname(SIM))
        print(f"\n  {robot}: exit {r.returncode}: {r.stderr.strip().splitlines()[-1:] or r.stdout[-300:]}")
        return r

    def assert_refused(self, r, *reasons):
        """Assert the sim exited 1 with GPU_REQUIRED and one of reasons on its last stderr line, before building a scene."""
        last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        self.assertEqual(r.returncode, 1, (r.stdout + r.stderr)[-2000:])
        self.assertTrue(last.startswith(f"{sim_common.GPU_REQUIRED}: "), last)
        self.assertTrue(any(reason in last for reason in reasons), (reasons, last))
        self.assertNotIn("RoboCasa task", r.stdout)

    def test_help_and_usage_errors_need_no_gpu(self):
        """--help and a usage error answer as argparse does with the probe failing, for either robot."""
        for robot in ("h12", "g1"):
            with self.subTest(robot=robot, case="--help"):
                r = self.start(robot, "--help", MUJOCO_GL="osmesa")
                self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-2000:])
                self.assertIn("usage:", r.stdout)
                self.assertNotIn(sim_common.GPU_REQUIRED, r.stderr)
            with self.subTest(robot=robot, case="usage error"):
                r = self.start(robot, "--layout", "one", MUJOCO_GL="osmesa")
                self.assertEqual(r.returncode, 2, (r.stdout + r.stderr)[-2000:])
                self.assertIn("--layout: invalid int value", r.stderr)
                self.assertNotIn(sim_common.GPU_REQUIRED, r.stderr)

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
                self.assert_refused(r, ", a software renderer", ": no EGL device: ")


if __name__ == "__main__":
    unittest.main()
