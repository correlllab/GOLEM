"""Validate mimic signs and finite limits using the real USD schema, without GPU."""
import importlib.util
from pathlib import Path
import unittest

try:
    from pxr import PhysxSchema, Usd, UsdPhysics
except ImportError:
    PhysxSchema = None

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('magpie_spawn', ROOT / 'CL_isaaclab_sim/tasks/common_config/magpie_spawn.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipIf(PhysxSchema is None, 'Requires Isaac Sim USD/PhysX schema packages')
class MagpieCouplingTests(unittest.TestCase):
    def test_asset_couplings_remap_references_without_editing_source(self):
        asset = ROOT / 'CL_Assets/isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd'
        if not asset.is_file():
            self.skipTest('CL_Assets checkout is required')
        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim('/Robot', 'Xform')
        root.GetReferences().AddReference(str(asset))
        module.add_magpie_couplings(root)
        followers = [p for p in Usd.PrimRange(root) if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, 'rotX')]
        self.assertEqual(len(followers), 8)
        for prim in followers:
            mimic = PhysxSchema.PhysxMimicJointAPI(prim, 'rotX')
            target = stage.GetPrimAtPath(mimic.GetReferenceJointRel().GetTargets()[0])
            self.assertTrue(str(target.GetPath()).startswith('/Robot/'))
            self.assertEqual(target.GetName(), prim.GetName()[:-1] + '1')
            gearing = 1. if prim.GetName().endswith('2') else -1.
            self.assertEqual(mimic.GetGearingAttr().Get(), gearing)
            self.assertEqual(mimic.GetOffsetAttr().Get(), 0.)
            reference = UsdPhysics.RevoluteJoint(target)
            follower = UsdPhysics.RevoluteJoint(prim)
            bounds = sorted([-gearing * reference.GetLowerLimitAttr().Get(),
                             -gearing * reference.GetUpperLimitAttr().Get()])
            self.assertAlmostEqual(follower.GetLowerLimitAttr().Get(), bounds[0])
            self.assertAlmostEqual(follower.GetUpperLimitAttr().Get(), bounds[1])
        # Applying twice is safe and does not duplicate schema instances.
        module.add_magpie_couplings(root)
        self.assertEqual(len([p for p in Usd.PrimRange(root) if p.HasAPI(PhysxSchema.PhysxMimicJointAPI, 'rotX')]), 8)

    def test_wrong_asset_fails_loudly(self):
        stage = Usd.Stage.CreateInMemory()
        with self.assertRaisesRegex(ValueError, 'missing linkage joints'):
            module.add_magpie_couplings(stage.DefinePrim('/Robot', 'Xform'))
