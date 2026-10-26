import os
import argparse
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

def view_urdf(filename):
    # package dir as parent folder
    model_path = os.path.dirname(os.path.abspath(filename))

    # load model
    model, visual, collision = pin.buildModelsFromUrdf(
        filename,
        package_dirs=model_path
    )
    viz = MeshcatVisualizer(model, visual, collision)
    viz.initViewer()
    viz.loadViewerModel()

    while True:
        # display robot at neutral configuration
        viz.display(pin.neutral(model))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize URDF model using Meshcat')
    parser.add_argument('filename', type=str, help='URDF filename')
    args = parser.parse_args()
    view_urdf(args.filename)
