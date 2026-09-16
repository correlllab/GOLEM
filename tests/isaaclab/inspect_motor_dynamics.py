"""Read authored wrist mass and drive properties without starting GPU physics."""
from pxr import Usd
stage = Usd.Stage.Open('/home/code/CL_Assets/isaac_assets/robots/h1_2-26dof-inspire-base-fix-usd/h1_2_26dof_with_inspire_rev_1_0.usd')
for prim in stage.Traverse():
    if any(name in prim.GetName() for name in ('left_wrist', 'L_hand', 'L_base')):
        print(str(prim.GetPath()), prim.GetTypeName())
        for attr in prim.GetAttributes():
            if any(k in attr.GetName().lower() for k in ('mass', 'inertia', 'armature', 'stiffness', 'damping', 'maxforce', 'axis')):
                print(' ', attr.GetName(), attr.Get())
