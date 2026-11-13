"""Check that the H1-2 sim's grippers report the pads' gap, as the real MAGPIE driver reports the jaw gap.

magpie_hand_bridge maps the finger_joint1 crank angle to an aperture through a table measured on
the gripper MJCF. These tests sweep the crank of each gripper MJCF alone in MuJoCo and compare the
gap between the pads' inner faces with the table, so a change to the gripper model that moves the
pads fails here. They need rclpy (the bridge is a Node), so they run in the robocasa container:
ROS_DOMAIN_ID=221 tests/robocasa/run_in_container.sh -p test_magpie_aperture.py
"""
import importlib.util
import os
import unittest

import numpy as np

HAVE_RCLPY = importlib.util.find_spec("rclpy") is not None
GRIPPERS = os.path.normpath(os.path.join(
    os.getcwd(), "..", "CL_Assets", "robosuite_assets", "grippers"))
# mm the table may differ from the model's pad gap: its interpolation error is under 0.1 mm
GAP_TOL_MM = 0.5
# s of sim time each crank target settles for
SETTLE_S = 0.6


def pad_gap_mm(model, data):
    """Return the mean gap (mm) between the left and right pads' inner faces, along the mount's y."""
    y = data.xmat[model.body("mount").id].reshape(3, 3)[:, 1]
    gaps = []
    for i in range(1, 5):
        faces = []
        for side, sign in (("left", -1.0), ("right", 1.0)):
            g = model.geom(f"{side}_pad{i}").id
            axis = data.geom_xmat[g].reshape(3, 3)[:, 1]
            # the pad faces the other finger with the side of its own y nearer the gripper's middle
            faces.append(data.geom_xpos[g] + sign * model.geom_size[g][1] * axis)
        gaps.append(float(np.dot(faces[0] - faces[1], y)) * 1000.0)
    return float(np.mean(gaps))


@unittest.skipUnless(HAVE_RCLPY, "needs rclpy (the robocasa container)")
class MagpieApertureTests(unittest.TestCase):
    """Check the bridge's crank-to-aperture table against the gripper MJCFs."""

    @classmethod
    def setUpClass(cls):
        """Load the bridge module (test_sim_rate's loader, with magpie_msgs stand-ins)."""
        from test_sim_rate import _magpie_hand_bridge
        cls.bridge = _magpie_hand_bridge()

    def sweep(self, xml):
        """Return [(crank rad, pad gap mm)] of `xml` closing from open to past the pads' contact."""
        import mujoco
        model = mujoco.MjModel.from_xml_path(xml)
        data = mujoco.MjData(model)
        left = model.actuator("left_finger_actuator").id
        right = model.actuator("right_finger_actuator").id
        joint = model.joint("finger_joint1").qposadr[0]
        rows = []
        for target in np.linspace(0.0, self.bridge.CLOSED_RAD, 42):
            data.ctrl[left], data.ctrl[right] = target, -target
            for _ in range(int(SETTLE_S / model.opt.timestep)):
                mujoco.mj_step(model, data)
            rows.append((float(data.qpos[joint]), pad_gap_mm(model, data)))
        return rows

    def test_table_matches_both_grippers(self):
        """The reported aperture is the pads' gap within GAP_TOL_MM over the whole stroke."""
        for name in ("magpie_right_gripper.xml", "magpie_left_gripper.xml"):
            with self.subTest(gripper=name):
                for crank, gap in self.sweep(os.path.join(GRIPPERS, name)):
                    self.assertAlmostEqual(self.bridge.rad_to_mm(crank), gap, delta=GAP_TOL_MM,
                                           msg=f"crank {crank:.3f} rad")

    def test_an_empty_close_reads_zero(self):
        """Closing on nothing stops with the pads together, which reads about 0 mm."""
        crank, gap = self.sweep(os.path.join(GRIPPERS, "magpie_right_gripper.xml"))[-1]
        self.assertLess(gap, GAP_TOL_MM)
        self.assertLess(self.bridge.rad_to_mm(crank), GAP_TOL_MM)

    def test_mm_to_rad_inverts_rad_to_mm(self):
        """A commanded aperture maps to the crank angle that reports it, open to closed."""
        for mm in (108.1, 80.0, 45.0, 10.0, 0.0):
            self.assertAlmostEqual(self.bridge.rad_to_mm(self.bridge.mm_to_rad(mm)), mm, places=6)
        self.assertEqual(self.bridge.mm_to_rad(200.0), 0.0)
        self.assertAlmostEqual(self.bridge.mm_to_rad(-5.0), self.bridge.APERTURE_TABLE_RAD[-1])


if __name__ == "__main__":
    unittest.main()
