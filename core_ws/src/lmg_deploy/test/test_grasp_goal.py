"""Test the grasp-frame goal maths: the wrist <-> grasp-frame transform from the URDF, the errors, the settle judge.

    python -m pytest test/test_grasp_goal.py
"""

import numpy as np
import pytest

from lmg_deploy import rotations as R
from lmg_deploy.grasp_goal import GRASP_FRAMES, GoalJudge, GraspFrames, pose_error
from lmg_deploy.kinematics import ArmKinematics
from shipped_policy import URDF, shipped_cfg

RNG = np.random.default_rng(3)


def _kin():
    """Return the H1-2 kinematics with the grasp frames, updated at the default pose."""
    cfg = shipped_cfg()
    names = cfg["joint_names"]
    kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]],
                        extra_frames=list(GRASP_FRAMES))
    kin.update(np.array(cfg["default_joint_pos"]))
    return kin, cfg


def test_grasp_pose_from_the_wrist_is_the_urdfs_grasp_frame_at_any_joints():
    """The fixed wrist -> grasp-frame transform, read once, gives the URDF grasp frame's pose at other joint angles."""
    kin, cfg = _kin()
    frames = GraspFrames(kin)
    for _ in range(10):
        q = np.array(cfg["default_joint_pos"]) + RNG.uniform(-0.5, 0.5, 27)
        kin.update(q)
        for arm, name in enumerate(GRASP_FRAMES):
            pos, quat = frames.grasp_pose(arm, *kin.wrist_pose(arm))
            ref_pos, ref_quat = kin.frame_pose(name)
            np.testing.assert_allclose(pos, ref_pos, atol=1e-9)
            np.testing.assert_allclose(R.quat_unique(quat), R.quat_unique(ref_quat), atol=1e-9)
    # the pinch point sits 23 cm along the wrist (the URDF's grasp_frame_joint)
    kin.update(np.array(cfg["default_joint_pos"]))
    pos, _ = frames.grasp_pose(0, *kin.wrist_pose(0))
    assert np.linalg.norm(pos - kin.wrist_pose(0)[0]) == pytest.approx(0.232, abs=0.001)


def test_wrist_goal_puts_the_grasp_frame_on_the_target():
    """wrist_goal() is grasp_pose()'s inverse: the grasp frame of the wrist goal is the target."""
    kin, _ = _kin()
    frames = GraspFrames(kin)
    for arm in range(2):
        target_pos, target_quat = RNG.normal(size=3), R.quat_unique(R.quat_from_rotvec(RNG.normal(size=3)))
        wrist = frames.wrist_goal(arm, target_pos, target_quat)
        pos, quat = frames.grasp_pose(arm, wrist[:3], wrist[3:])
        np.testing.assert_allclose(pos, target_pos, atol=1e-12)
        np.testing.assert_allclose(quat, target_quat, atol=1e-12)


def test_grasp_frames_must_be_in_the_kinematics():
    """GraspFrames refuses kinematics without the grasp frames."""
    cfg = shipped_cfg()
    names = cfg["joint_names"]
    kin = ArmKinematics(URDF, names, cfg["arms"]["ee_frames"], [names[13:20], names[20:]])
    kin.update(np.zeros(27))
    with pytest.raises(ValueError, match="grasp"):
        GraspFrames(kin)


def test_pose_error_is_the_distance_and_the_rotation_angle():
    """pose_error: metres between the origins, radians of the rotation between the orientations, either sign of q."""
    q = R.quat_from_rotvec(np.array([0.0, 0.0, 0.3]))
    m, rad = pose_error(np.zeros(3), np.array([1.0, 0, 0, 0]), np.array([0.03, 0.04, 0.0]), -q)
    assert m == pytest.approx(0.05) and rad == pytest.approx(0.3)


def test_goal_judge_needs_every_arm_within_the_tolerance_for_the_settle_time():
    """The goal is reached after settle_s with every arm within the tolerance; leaving it restarts the wait."""
    judge = GoalJudge(0.01, 0.1, 0.3)
    inside, outside = [(0.005, 0.05), (0.0, 0.0)], [(0.005, 0.05), (0.02, 0.0)]
    assert not judge.reached(0.0, inside)
    assert not judge.reached(0.2, inside)
    assert not judge.reached(0.25, outside)
    assert not judge.reached(0.3, inside)
    assert not judge.reached(0.5, inside)
    assert judge.reached(0.6, inside)
