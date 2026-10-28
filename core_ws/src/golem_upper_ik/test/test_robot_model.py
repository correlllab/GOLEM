'''
RobotModel and RobotDynamics against the old H1-2 controller, and on the G1.

test/fixtures/fk_h12.json holds what h12_ros2_controller's RobotModel and RobotDynamics
compute at 50 seeded random configurations (written by tools/snapshot_old_fk.py); these
tests import no old package.
'''
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest

FIXTURE = Path(__file__).parent / 'fixtures' / 'fk_h12.json'
# Pinocchio's own float noise is around 1e-15; the old and new models are built the same way
FK_ATOL = 1e-9
# The Magpie gripper's finger and rocker hinges are continuous joints, frozen at Pinocchio's
# neutral configuration (angle 0). The old model froze them at [cos, sin] = [0, 0], which is
# no rotation at all, so its fingers and rockers (8 links of 50 g) sat elsewhere: at the
# fixture's configurations the gravity torques differ by up to 6.4e-3 N*m and the centre of
# mass by 1.1e-5 m. The frames before them (the end effectors, the grasp frames, the torso)
# are bit-identical.
GRAVITY_ATOL = 1e-2  # N*m
COM_ATOL = 1e-4  # m
# (left, right) arm joint slots
ARM_SLOTS = {
    'h12': (range(13, 20), range(20, 27)),
    'g1': (range(15, 22), range(22, 29)),
}


def profile_of(robot):
    if robot == 'g1':
        pytest.importorskip('g1_description')
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def model_of(robot, **kwargs):
    from golem_upper_ik.core.robot_model import RobotModel
    return RobotModel(profile_of(robot), **kwargs)


@pytest.fixture(scope='module')
def fixture():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope='module')
def h12_model():
    return model_of('h12')


def placement(transform):
    '''[x, y, z, R row-major], as the fixture stores a frame placement'''
    transform = np.asarray(transform)
    return np.concatenate([transform[:3, 3], transform[:3, :3].reshape(-1)])


def test_fixture_matches_the_profile(fixture):
    profile = profile_of('h12')
    assert fixture['urdf'] == Path(profile.urdf_path).name
    assert tuple(fixture['body_joints']) == profile.body_joints
    assert tuple(fixture['enabled_joints']) == profile.enabled_joints
    assert tuple(fixture['ee_frames']) == profile.ee_frames
    assert tuple(fixture['planner_frames']) == profile.planner_frames
    assert fixture['imu_frame'] == profile.imu_frame
    assert len(fixture['q']) == 50


def test_h12_fk_matches_old_controller(fixture, h12_model):
    model = h12_model
    frames = fixture['ee_frames'] + fixture['planner_frames'] + [fixture['imu_frame']]
    assert sorted(fixture['frames']) == sorted(frames)
    for index, q in enumerate(np.asarray(fixture['q'])):
        for frame in frames:
            np.testing.assert_allclose(
                placement(model.get_frame_transformation(frame, q)),
                fixture['frames'][frame][index], rtol=0, atol=FK_ATOL,
                err_msg=f'{frame} at q {index}')
            # the IK's model: the IK joints of q, the others frozen at 0
            np.testing.assert_allclose(
                placement(model.get_frame_transformation_reduced(frame, q[model.reduced_mask])),
                fixture['reduced_frames'][frame][index], rtol=0, atol=FK_ATOL,
                err_msg=f'{frame} at q {index}, reduced model')
        np.testing.assert_allclose(
            model.full_q(q, imu_quat=fixture['imu_quat'][index]), fixture['full_q'][index],
            rtol=0, atol=FK_ATOL, err_msg=f'full_q at q {index}')


def test_h12_dynamics_matches_old_controller(fixture, h12_model):
    model = h12_model
    try:
        for index, q in enumerate(np.asarray(fixture['q'])):
            model._q = q.copy()  # the measured state the twist is taken at
            for frame in fixture['ee_frames']:
                np.testing.assert_allclose(
                    model.dynamics.compute_frame_twist(frame, np.asarray(fixture['dq'][index])),
                    fixture['twist'][frame][index], rtol=0, atol=FK_ATOL,
                    err_msg=f'{frame} twist at q {index}')
            np.testing.assert_allclose(
                model.dynamics.get_gravity_compensation(q, fixture['imu_quat'][index]),
                fixture['gravity'][index], rtol=0, atol=GRAVITY_ATOL,
                err_msg=f'gravity at q {index}')
            np.testing.assert_allclose(model.dynamics.get_com(q), fixture['com'][index],
                                       rtol=0, atol=COM_ATOL, err_msg=f'com at q {index}')
    finally:
        model._q = np.zeros(model.num_motor)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_models_follow_the_profile(robot):
    profile = profile_of(robot)
    model = model_of(robot)

    # every non-body joint is frozen; the rest are the body joints in slot order
    assert tuple(model.model_body.names[1:]) == profile.body_joints
    assert tuple(model.model.names[2:]) == profile.body_joints  # after the free-flyer
    assert model.model_body.nq == len(profile.body_joints)
    assert model.num_motor == len(profile.body_joints)
    # the reduced model solves for the IK joints only
    assert tuple(model.model_body_reduced.names[1:]) == profile.enabled_joints
    assert [profile.body_joints[i] for i in np.flatnonzero(model.reduced_mask)] == list(
        profile.enabled_joints)
    assert model.imu_frame == profile.imu_frame
    # the old controllers' LEFT_ARM_INDEX and RIGHT_ARM_INDEX
    left, right = ARM_SLOTS[robot]
    assert model.left_arm_index == list(left)
    assert model.right_arm_index == list(right)


def test_h12_model_has_collision(h12_model):
    model = h12_model
    assert model.has_collision is True
    assert len(model.collision_model_body_reduced.collisionPairs) > 0
    # 'home' hangs the arms clear of the body; some random arm configurations do not
    assert model.check_collision_free_reduced(model.profile.named_configs['home'])
    rng = np.random.default_rng(0)
    lower = model.model_body_reduced.lowerPositionLimit
    upper = model.model_body_reduced.upperPositionLimit
    free = [model.check_collision_free_reduced(rng.uniform(lower, upper)) for _ in range(200)]
    assert any(free) and not all(free)


def test_g1_model_builds_without_collision():
    model = model_of('g1')
    assert model.has_collision is False
    q = np.zeros(model.model_body.nq)
    # without a self-collision model every configuration counts as collision-free
    assert model.check_collision_free(q)
    assert model.check_collision_free_reduced(q[model.reduced_mask])
    assert model.check_valid(q)


def test_rebuild_reduced_relocks_at_measured():
    model = model_of('h12')
    torso = model.profile.body_joints.index('torso_joint')
    q = np.zeros(model.num_motor)
    q[model.reduced_mask] = model.profile.named_configs['prep']
    q[torso] = 0.3
    model._q = q.copy()  # measured: the torso turned 0.3 rad

    frame = 'left_wrist_yaw_link'
    at_zero = q.copy()
    at_zero[torso] = 0.0
    built = model.get_frame_transformation_reduced(frame, q[model.reduced_mask])
    np.testing.assert_allclose(built, model.get_frame_transformation(frame, at_zero),
                               atol=1e-12)
    assert not np.allclose(built, model.get_frame_transformation(frame, q), atol=1e-3)

    model.rebuild_reduced_model()
    relocked = model.get_frame_transformation_reduced(frame, q[model.reduced_mask])
    np.testing.assert_allclose(relocked, model.get_frame_transformation(frame, q), atol=1e-12)

    # the reduced collision model is re-locked with it: the torso's spheres turn with the torso
    def sphere_position(robot_model, geometry_model, name, q_model):
        geometry_data = pin.GeometryData(geometry_model)
        pin.updateGeometryPlacements(robot_model, pin.Data(robot_model), geometry_model,
                                     geometry_data, q_model)
        return geometry_data.oMg[geometry_model.getGeometryId(name)].translation

    sphere = next(geometry.name
                  for geometry in model.collision_model_body_reduced.geometryObjects
                  if geometry.name.startswith('torso_link'))
    np.testing.assert_allclose(
        sphere_position(model.model_body_reduced, model.collision_model_body_reduced, sphere,
                        q[model.reduced_mask]),
        sphere_position(model.model_body, model.collision_model_body, sphere, q), atol=1e-12)

    # an explicit reference re-locks there instead
    model.rebuild_reduced_model(at_zero)
    np.testing.assert_allclose(
        model.get_frame_transformation_reduced(frame, q[model.reduced_mask]),
        model.get_frame_transformation(frame, at_zero), atol=1e-12)


def test_imu_frame_defaults_to_the_profile_and_can_be_overridden():
    profile = profile_of('h12')
    model = model_of('h12', imu_frame='pelvis')
    assert model.imu_frame == 'pelvis'
    # the IMU on the pelvis: the pelvis takes the IMU's orientation, whatever the torso does
    q = np.zeros(model.num_motor)
    q[profile.body_joints.index('torso_joint')] = 0.5
    quat = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])  # w, x, y, z
    np.testing.assert_allclose(model.full_q(q, imu_quat=quat)[3:7],
                               [quat[1], quat[2], quat[3], quat[0]], atol=1e-12)
    with pytest.raises(ValueError, match='no_such_frame'):
        model_of('h12', imu_frame='no_such_frame')


def test_state_staleness_comes_from_the_state_source(h12_model):
    model = h12_model
    assert model.state_staleness_ms == 0.0
    q = np.arange(model.num_motor, dtype=float) * 0.01
    source = SimpleNamespace(state={'q': q, 'dq': np.zeros(model.num_motor)},
                             staleness_ms=12.5)
    model.state_subscriber = source
    try:
        assert model.state_staleness_ms == 12.5
        np.testing.assert_array_equal(model.state['q'], q)
        np.testing.assert_array_equal(model.state_reduced['q'], q[model.reduced_mask])
    finally:
        model.state_subscriber = None


def test_reduced_limit_violations_name_the_joint(h12_model):
    model = h12_model
    q = np.zeros(model.model_body_reduced.nq)
    assert model.reduced_limit_violations(q) == []
    upper = model.model_body_reduced.upperPositionLimit
    q[3] = upper[3] + 0.1
    violations = model.reduced_limit_violations(q)
    assert len(violations) == 1
    violation = violations[0]
    assert violation['name'] == model.profile.enabled_joints[3]
    assert violation['index'] == 3
    assert violation['bound'] == pytest.approx(upper[3])
    assert violation['overshoot'] == pytest.approx(0.1)
    assert model.reduced_limit_violations(q, tol=0.2) == []
