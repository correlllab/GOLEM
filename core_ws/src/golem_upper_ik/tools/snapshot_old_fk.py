#!/usr/bin/env python3
'''
Writes what the old H1-2 arm controller's robot model computes, for golem_upper_ik's
robot-model test (test/test_robot_model.py).

From h12_ros2_controller's RobotModel and RobotDynamics, built from the URDF its
frame_task_server loads, at 50 configurations drawn with a fixed seed, it writes
test/fixtures/fk_h12.json:

  q               the 27 body joints, rad, uniform within the URDF limits
  imu_quat        a unit quaternion (w, x, y, z): the IMU orientation for full_q and gravity
  dq              joint velocities, rad/s, uniform in [-1, 1], for the end-effector twists
  frames          per frame, its placement in the free-flyer model at q (the base at the
                  origin, unrotated): [x, y, z, then the rotation row by row]
  reduced_frames  the same in the reduced (IK) model, at q's IK joints with the other
                  joints frozen at 0
  full_q          the free-flyer configuration RobotModel.full_q makes of q and imu_quat
  gravity         RobotDynamics.get_gravity_compensation(q, imu_quat), N*m per body joint
  twist           per end effector, RobotDynamics.compute_frame_twist(frame, dq) with the
                  measured state at q: (vx, vy, vz, wx, wy, wz)
  com             RobotDynamics.get_com(q), m

The frames are the end effectors, the planner's frames and the IMU frame. The test imports
no old package. h12_ros2_controller must import, so run this in the ros container after
the workspace is built and sourced:

  python3 /home/code/core_ws/src/golem_upper_ik/tools/snapshot_old_fk.py
'''
import json
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = PACKAGE / 'test' / 'fixtures' / 'fk_h12.json'
SEED = 20261027
COUNT = 50
# UpperController's left_ee_name and right_ee_name, set in its constructor
EE_FRAMES = ['left_wrist_yaw_link', 'right_wrist_yaw_link']
# the frame RobotModel.full_q takes the IMU orientation in
IMU_FRAME = 'torso_link'


def placement(transform):
    transform = np.asarray(transform)
    return [float(x) for x in np.concatenate([transform[:3, 3],
                                              transform[:3, :3].reshape(-1)])]


def floats(values):
    return [float(x) for x in np.asarray(values).reshape(-1)]


def snapshot():
    from h12_ros2_controller.core.planner import PlannerConfig
    from h12_ros2_controller.core.robot_model import RobotModel
    from h12_ros2_controller.utility.joint_definition import BODY_JOINTS, ENABLED_JOINTS
    from h12_ros2_controller.utility.path_definition_ros import URDF_MAGPIE_PIN_PATH

    model = RobotModel(URDF_MAGPIE_PIN_PATH)
    model.init_reduced_model(ENABLED_JOINTS)
    planner_frames = list(PlannerConfig().frame_names)
    frames = EE_FRAMES + planner_frames + [IMU_FRAME]

    rng = np.random.default_rng(SEED)
    lower = model.model_body.lowerPositionLimit
    upper = model.model_body.upperPositionLimit
    data = {
        'urdf': Path(URDF_MAGPIE_PIN_PATH).name,
        'seed': SEED,
        'body_joints': list(BODY_JOINTS),
        'enabled_joints': list(ENABLED_JOINTS),
        'ee_frames': EE_FRAMES,
        'planner_frames': planner_frames,
        'imu_frame': IMU_FRAME,
        'q': [], 'imu_quat': [], 'dq': [],
        'frames': {frame: [] for frame in frames},
        'reduced_frames': {frame: [] for frame in frames},
        'full_q': [], 'gravity': [],
        'twist': {frame: [] for frame in EE_FRAMES},
        'com': [],
    }
    for _ in range(COUNT):
        q = rng.uniform(lower, upper)
        imu_quat = rng.normal(size=4)
        imu_quat /= np.linalg.norm(imu_quat)
        dq = rng.uniform(-1.0, 1.0, size=q.shape)

        data['q'].append(floats(q))
        data['imu_quat'].append(floats(imu_quat))
        data['dq'].append(floats(dq))
        for frame in frames:
            data['frames'][frame].append(placement(model.get_frame_transformation(frame, q)))
            data['reduced_frames'][frame].append(placement(
                model.get_frame_transformation_reduced(frame, q[model.reduced_mask])))
        data['full_q'].append(floats(model.full_q(q, imu_quat=imu_quat)))
        data['gravity'].append(floats(model.dynamics.get_gravity_compensation(q, imu_quat)))
        model._q = q.copy()  # compute_frame_twist takes q from the measured state
        for frame in EE_FRAMES:
            data['twist'][frame].append(floats(model.dynamics.compute_frame_twist(frame, dq)))
        data['com'].append(floats(model.dynamics.get_com(q)))
    return data


def write(data):
    # one top-level key per line, each value on one line: the arrays are long
    lines = [f'{json.dumps(key)}: {json.dumps(value, separators=(",", ":"))}'
             for key, value in data.items()]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text('{\n' + ',\n'.join(lines) + '\n}\n')
    print(f'wrote {OUTPUT.relative_to(PACKAGE)}')


if __name__ == '__main__':
    write(snapshot())
