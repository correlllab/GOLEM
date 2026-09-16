"""Inspect composed robot links, joints, limits and sensors without starting Kit."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pxr import Usd, UsdPhysics

path = sys.argv[1] if len(sys.argv) > 1 else '/home/code/CL_Assets/isaac_assets/robots/h1_2_magpie/h1_2_magpie.usd'
stage = Usd.Stage.Open(path)
print('ASSET', path, 'default', stage.GetDefaultPrim().GetPath(), flush=True)
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI) or 'Joint' in p.GetTypeName():
        print(str(p.GetPath()), p.GetTypeName(), {a.GetName(): a.Get() for a in p.GetAttributes() if any(k in a.GetName() for k in ('Limit','axis','stiffness','damping','mimic'))}, p.GetAppliedSchemas(), flush=True)

app.close()
