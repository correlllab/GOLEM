"""Inspect robot USD composition without starting Kit or physics."""
import glob
from pxr import Usd

for path in glob.glob('/home/code/CL_Assets/isaac_assets/robots/h1_2_handless/**/*.usd', recursive=True):
    stage = Usd.Stage.Open(path)
    print(path, flush=True)
    print(stage.GetRootLayer().ExportToString()[:2000], flush=True)
    print('prims', len(list(stage.Traverse())), flush=True)
    print([(str(p.GetPath()), p.GetTypeName()) for p in stage.Traverse() if 'Joint' in p.GetTypeName() or any(s in p.GetName().lower() for s in ('camera', 'lidar', 'mid360'))], flush=True)
