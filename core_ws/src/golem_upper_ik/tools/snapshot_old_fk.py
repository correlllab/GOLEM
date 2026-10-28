#!/usr/bin/env python3
'''
Writes what the old arm controllers' robot models compute, for golem_upper_ik's robot-model
tests (test/test_robot_model.py).

From h12_ros2_controller's (h12) and g1_control's (g1) RobotModel and RobotDynamics, each
built from the URDF its frame_task_server loads, at 50 configurations drawn with a fixed
seed, it writes test/fixtures/fk_<robot>.json:

  q               the body joints, rad, uniform within the URDF limits
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
  collision_free_reduced
                  (h12 only: the G1 has no self-collision model) whether the reduced model
                  at q's IK joints is free of self-collision

The frames are the end effectors, the planner's frames and the IMU frame. The tests import
no old package. Both old packages must import, so run this in the G1-mode ros container
after the workspace is built and sourced:

  python3 /home/code/core_ws/src/golem_upper_ik/tools/snapshot_old_fk.py
'''
import json
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT = PACKAGE / 'test' / 'fixtures'
SEED = 20261027
COUNT = 50


def placement(transform):
    transform = np.asarray(transform)
    return [float(x) for x in np.concatenate([transform[:3, 3],
                                              transform[:3, :3].reshape(-1)])]


def floats(values):
    return [float(x) for x in np.asarray(values).reshape(-1)]


def old_h12():
    from h12_ros2_controller.core.planner import PlannerConfig
    from h12_ros2_controller.core.robot_model import RobotModel
    from h12_ros2_controller.utility.joint_definition import BODY_JOINTS, ENABLED_JOINTS
    from h12_ros2_controller.utility.path_definition_ros import (
        SRDF_MAGPIE_SPHERE_PATH,
        URDF_MAGPIE_PIN_PATH,
        URDF_MAGPIE_SPHERE_PATH,
    )

    model = RobotModel(URDF_MAGPIE_PIN_PATH)
    model.init_reduced_model(ENABLED_JOINTS)
    model.init_collision_model(URDF_MAGPIE_SPHERE_PATH, SRDF_MAGPIE_SPHERE_PATH)
    return {
        'model': model,
        'urdf': URDF_MAGPIE_PIN_PATH,
        'body_joints': list(BODY_JOINTS),
        'enabled_joints': list(ENABLED_JOINTS),
        # UpperController's left_ee_name and right_ee_name, set in its constructor
        'ee_frames': ['left_wrist_yaw_link', 'right_wrist_yaw_link'],
        'planner_frames': list(PlannerConfig().frame_names),
        # the frame RobotModel.full_q takes the IMU orientation in
        'imu_frame': 'torso_link',
        'collision': True,
    }


def old_g1():
    from g1_control.core.planner import PlannerConfig
    from g1_control.core.robot_model import RobotModel
    from g1_control.utility.joint_definition import (
        BODY_JOINTS,
        ENABLED_JOINTS,
        LEFT_EE_FRAME,
        RIGHT_EE_FRAME,
    )
    from g1_control.utility.path_definition_ros import URDF_PIN_PATH

    # UpperController's default for controller.imu_frame, which every G1 config sets
    imu_frame = 'imu_in_pelvis'
    model = RobotModel(URDF_PIN_PATH, imu_frame=imu_frame)
    model.init_reduced_model(ENABLED_JOINTS)
    return {
        'model': model,
        'urdf': URDF_PIN_PATH,
        'body_joints': list(BODY_JOINTS),
        'enabled_joints': list(ENABLED_JOINTS),
        'ee_frames': [LEFT_EE_FRAME, RIGHT_EE_FRAME],
        'planner_frames': list(PlannerConfig().frame_names),
        'imu_frame': imu_frame,
        'collision': False,
    }


def snapshot(old):
    model = old['model']
    frames = list(dict.fromkeys(old['ee_frames'] + old['planner_frames'] + [old['imu_frame']]))
    rng = np.random.default_rng(SEED)
    lower = model.model_body.lowerPositionLimit
    upper = model.model_body.upperPositionLimit
    data = {
        'urdf': Path(old['urdf']).name,
        'seed': SEED,
        'body_joints': old['body_joints'],
        'enabled_joints': old['enabled_joints'],
        'ee_frames': old['ee_frames'],
        'planner_frames': old['planner_frames'],
        'imu_frame': old['imu_frame'],
        'q': [], 'imu_quat': [], 'dq': [],
        'frames': {frame: [] for frame in frames},
        'reduced_frames': {frame: [] for frame in frames},
        'full_q': [], 'gravity': [],
        'twist': {frame: [] for frame in old['ee_frames']},
    }
    if old['collision']:
        data['collision_free_reduced'] = []
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
        for frame in old['ee_frames']:
            data['twist'][frame].append(floats(model.dynamics.compute_frame_twist(frame, dq)))
        if old['collision']:
            data['collision_free_reduced'].append(
                bool(model.check_collision_free_reduced(q[model.reduced_mask])))
    return data


def write(robot, data):
    # one top-level key per line, each value on one line: the arrays are long
    lines = [f'{json.dumps(key)}: {json.dumps(value, separators=(",", ":"))}'
             for key, value in data.items()]
    target = OUTPUT / f'fk_{robot}.json'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target.write_text('{\n' + ',\n'.join(lines) + '\n}\n')
    print(f'wrote {target.relative_to(PACKAGE)}')


if __name__ == '__main__':
    write('h12', snapshot(old_h12()))
    write('g1', snapshot(old_g1()))
