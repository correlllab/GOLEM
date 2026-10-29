'''
RobotModel and RobotDynamics against the old controllers, on both robots.

test/fixtures/fk_<robot>.json holds what the robot's previous arm controller's RobotModel
and RobotDynamics computed at 50 seeded random configurations (test/fixtures/README.md);
these tests import no old package.
'''
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

FIXTURES = Path(__file__).parent / 'fixtures'
# Pinocchio's own float noise is around 1e-15; the old and new models are built the same way
FK_ATOL = 1e-9
# The Magpie gripper's finger and rocker hinges are continuous joints, frozen at Pinocchio's
# neutral configuration (angle 0). The old H1-2 model froze them at [cos, sin] = [0, 0],
# which is no rotation at all, so its fingers and rockers (8 links of 50 g) sat elsewhere:
# at the fixture's configurations the gravity torques differ by up to 6.4e-3 N*m. The frames
# before them (the end effectors, the grasp frames, the torso) are bit-identical. The G1 has
# no continuous joint: its dynamics are identical.
GRAVITY_ATOL = {'h12': 1e-2, 'g1': FK_ATOL}  # N*m
# each arm's seven IK joints, by /lowcmd slot: (left, right)
ARM_SLOTS = {
    'h12': (range(13, 20), range(20, 27)),
    'g1': (range(15, 22), range(22, 29)),
}


def profile_of(robot):
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def model_of(robot, **kwargs):
    from golem_upper_ik.core.robot_model import RobotModel
    return RobotModel(profile_of(robot), **kwargs)


def fixture_of(robot):
    return json.loads((FIXTURES / f'fk_{robot}.json').read_text())


def fixed_state(q, dq=None, imu_quaternion=None):
    from golem_upper_ik.core.robot_model import FixedState
    return FixedState(q, dq=dq, imu_quaternion=imu_quaternion)


def placement(transform):
    '''[x, y, z, R row-major], as the fixtures store a frame placement'''
    transform = np.asarray(transform)
    return np.concatenate([transform[:3, 3], transform[:3, :3].reshape(-1)])


def frames_of(fixture):
    return list(dict.fromkeys(
        fixture['ee_frames'] + fixture['planner_frames'] + [fixture['imu_frame']]))


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_fixture_matches_the_profile(robot):
    fixture = fixture_of(robot)
    profile = profile_of(robot)
    assert fixture['urdf'] == Path(profile.urdf_path).name
    assert tuple(fixture['body_joints']) == profile.body_joints
    assert tuple(fixture['enabled_joints']) == profile.enabled_joints
    assert tuple(fixture['ee_frames']) == profile.ee_frames
    assert tuple(fixture['planner_frames']) == profile.planner_frames
    assert fixture['imu_frame'] == profile.imu_frame
    assert len(fixture['q']) == 50
    assert sorted(fixture['frames']) == sorted(frames_of(fixture))


def check_fk(model, fixture):
    for index, q in enumerate(np.asarray(fixture['q'])):
        for frame in frames_of(fixture):
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


def test_h12_fk_matches_old_controller(models):
    check_fk(models('h12'), fixture_of('h12'))


def test_g1_fk_matches_old_controller(models):
    check_fk(models('g1'), fixture_of('g1'))


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_dynamics_matches_old_controller(models, robot):
    fixture = fixture_of(robot)
    model = models(robot)
    try:
        for index, q in enumerate(np.asarray(fixture['q'])):
            model.state_subscriber = fixed_state(q)  # the twist is taken at the measured q
            for frame in fixture['ee_frames']:
                np.testing.assert_allclose(
                    model.dynamics.compute_frame_twist(frame, np.asarray(fixture['dq'][index])),
                    fixture['twist'][frame][index], rtol=0, atol=FK_ATOL,
                    err_msg=f'{frame} twist at q {index}')
            np.testing.assert_allclose(
                model.dynamics.get_gravity_compensation(q, fixture['imu_quat'][index]),
                fixture['gravity'][index], rtol=0, atol=GRAVITY_ATOL[robot],
                err_msg=f'gravity at q {index}')
    finally:
        model.state_subscriber = None


def test_h12_collision_matches_old_controller(models):
    fixture = fixture_of('h12')
    model = models('h12')
    verdicts = [bool(model.check_collision_free_reduced(q[model.reduced_mask]))
                for q in np.asarray(fixture['q'])]
    assert verdicts == fixture['collision_free_reduced']
    assert any(verdicts) and not all(verdicts)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_models_follow_the_profile(models, robot):
    profile = profile_of(robot)
    model = models(robot)

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
    left, right = ARM_SLOTS[robot]
    assert model.left_arm_index == list(left)
    assert model.right_arm_index == list(right)
    assert model.has_collision is (profile.collision is not None)


def test_h12_model_has_collision(models):
    model = models('h12')
    assert model.has_collision is True
    assert len(model.collision_model_body_reduced.collisionPairs) > 0
    assert model.check_collision_free_reduced(model.profile.named_configs['home'])


def pair_distances(model, q_reduced):
    '''Each enabled collision pair's distance at q_reduced, m (negative: they overlap)'''
    geometry = model.collision_model_body_reduced
    data = model.collision_data_body_reduced
    pin.computeDistances(model.model_body_reduced, model.data_body_reduced, geometry, data,
                         q_reduced)
    return np.array([result.min_distance for result in data.distanceResults])


def test_g1_model_has_collision(models):
    '''
    Every named configuration is collision-free and keeps every G1 config's d_min, the
    H1-2's 0.02 m, except arms_down: its hands and wrists sit inside the hip-pitch spheres
    (robots/g1.py)
    '''
    from golem_upper_ik.config import load_controller_config
    model = models('g1')
    assert model.has_collision is True
    # 24 spheres; the SRDF leaves 210 of their 240 pairs on the reduced model (the legs and
    # the waist locked)
    assert len(model.collision_model_body_reduced.geometryObjects) == 24
    assert len(model.collision_model_body_reduced.collisionPairs) == 210
    d_min = max(load_controller_config('g1', name)['controller']['d_min']
                for name in ('g1_split', 'sim_split'))
    assert d_min == 0.02
    for name, q in model.profile.named_configs.items():
        distances = pair_distances(model, q)
        if name == 'arms_down':
            assert not model.check_collision_free_reduced(q)
            assert distances.min() < -0.04, distances.min()
        else:
            assert model.check_collision_free_reduced(q), name
            assert distances.min() >= d_min, (name, distances.min())


def test_rebuild_reduced_relocks_at_measured(models):
    model = models('h12')
    torso = model.profile.body_joints.index('torso_joint')
    q = np.zeros(model.num_motor)
    q[model.reduced_mask] = model.profile.named_configs['prep']
    q[torso] = 0.3
    at_zero = q.copy()
    at_zero[torso] = 0.0
    frame = 'left_wrist_yaw_link'
    try:
        model.state_subscriber = fixed_state(q)  # measured: the torso turned 0.3 rad

        built = model.get_frame_transformation_reduced(frame, q[model.reduced_mask])
        np.testing.assert_allclose(built, model.get_frame_transformation(frame, at_zero),
                                   atol=1e-12)
        assert not np.allclose(built, model.get_frame_transformation(frame, q), atol=1e-3)

        model.rebuild_reduced_model()
        relocked = model.get_frame_transformation_reduced(frame, q[model.reduced_mask])
        np.testing.assert_allclose(relocked, model.get_frame_transformation(frame, q),
                                   atol=1e-12)

        # the reduced collision model is re-locked with it: the torso's spheres turn too
        urdf, _ = model.profile.collision
        sphere_model, sphere_geometry, _ = model._load_urdf_body(urdf)
        sphere = next(geometry.name
                      for geometry in model.collision_model_body_reduced.geometryObjects
                      if geometry.name.startswith('torso_link'))

        def sphere_position(robot_model, geometry_model, q_model):
            geometry_data = pin.GeometryData(geometry_model)
            pin.updateGeometryPlacements(robot_model, pin.Data(robot_model), geometry_model,
                                         geometry_data, q_model)
            return geometry_data.oMg[geometry_model.getGeometryId(sphere)].translation

        np.testing.assert_allclose(
            sphere_position(model.model_body_reduced, model.collision_model_body_reduced,
                            q[model.reduced_mask]),
            sphere_position(sphere_model, sphere_geometry, q), atol=1e-12)

        # an explicit reference re-locks there instead
        model.rebuild_reduced_model(at_zero)
        np.testing.assert_allclose(
            model.get_frame_transformation_reduced(frame, q[model.reduced_mask]),
            model.get_frame_transformation(frame, at_zero), atol=1e-12)
    finally:
        model.state_subscriber = None
        model.rebuild_reduced_model(np.zeros(model.num_motor))


def test_imu_frame_defaults_to_the_profile_and_can_be_overridden(models):
    profile = profile_of('g1')
    on_pelvis = models('g1')
    on_torso = model_of('g1', imu_frame='imu_in_torso')
    assert on_pelvis.imu_frame == profile.imu_frame == 'imu_in_pelvis'
    assert on_torso.imu_frame == 'imu_in_torso'

    def pelvis_turn(model, waist_yaw):
        '''rad the pelvis is turned by waist_yaw, the IMU reading the same orientation'''
        q = np.zeros(model.num_motor)
        level = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        before = model.full_q(q, imu_quat=level)[3:7]
        q[profile.body_joints.index('waist_yaw_joint')] = waist_yaw
        after = model.full_q(q, imu_quat=level)[3:7]
        return 2.0 * np.arccos(min(1.0, abs(float(np.dot(before, after)))))

    # an IMU on the pelvis ignores the waist; one on the torso turns with it, so the
    # pelvis it implies turns back by the waist's angle
    assert pelvis_turn(on_pelvis, 0.5) == pytest.approx(0.0, abs=1e-6)
    assert pelvis_turn(on_torso, 0.5) == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError, match='no_such_frame'):
        model_of('g1', imu_frame='no_such_frame')


def test_state_source_protocol(models):
    from golem_upper_ik.core.robot_model import FixedState, StateSource
    model = models('h12')
    assert isinstance(FixedState(np.zeros(model.num_motor)), StateSource)
    # without a source the model is at rest at q = 0, and nothing is stale
    assert model.state_subscriber is None
    assert set(model.state) == {'q', 'dq'}
    np.testing.assert_array_equal(model.state['q'], 0.0)
    assert model.state_staleness_ms == 0.0

    q = np.arange(model.num_motor, dtype=float) * 0.01
    source = FixedState(q, dq=-q)
    source.staleness_ms = 12.5
    model.state_subscriber = source
    try:
        assert model.state_staleness_ms == 12.5
        np.testing.assert_array_equal(model.state['q'], q)
        reduced = model.state_reduced
        assert set(reduced) == {'q', 'dq'}
        np.testing.assert_array_equal(reduced['q'], q[model.reduced_mask])
        np.testing.assert_array_equal(reduced['dq'], -q[model.reduced_mask])
        # one value per body joint, or the read is refused
        model.state_subscriber = FixedState(q[:-1])
        with pytest.raises(ValueError, match='one value per body joint'):
            model.state_reduced
    finally:
        model.state_subscriber = None


class ChangingSource:
    '''A source whose every read is a newer sample, the next q a step further'''

    def __init__(self, count):
        self.count = count
        self.reads = 0
        self.staleness_ms = 0.0

    @property
    def state(self):
        self.reads += 1
        level = np.array([1.0, 0.0, 0.0, 0.0])
        return {'q': np.full(self.count, 0.1 * self.reads), 'dq': np.zeros(self.count),
                'imu_state': type('Imu', (), {'quaternion': level})()}

    def shutdown(self):
        pass


def test_update_kinematics_reads_the_state_once(models):
    model = models('h12')
    frame = 'left_wrist_yaw_link'
    source = ChangingSource(model.num_motor)
    model.state_subscriber = source
    try:
        model.update_kinematics()
        assert source.reads == 1
        full = model.data.oMf[model.model.getFrameId(frame)].homogeneous.copy()
        body = model.data_body.oMf[model.model_body.getFrameId(frame)].homogeneous.copy()
        reduced = model.data_body_reduced.oMf[
            model.model_body_reduced.getFrameId(frame)].homogeneous.copy()
    finally:
        model.state_subscriber = None

    # every model is at the one sample read: q = 0.1 everywhere, the IMU level
    q = np.full(model.num_motor, 0.1)
    data = model.model.createData()
    pin.framesForwardKinematics(model.model, data,
                                model.full_q(q, imu_quat=[1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(full, data.oMf[model.model.getFrameId(frame)].homogeneous,
                               atol=1e-12)
    data = model.model_body.createData()
    pin.framesForwardKinematics(model.model_body, data, q)
    np.testing.assert_allclose(body, data.oMf[model.model_body.getFrameId(frame)].homogeneous,
                               atol=1e-12)
    np.testing.assert_allclose(
        reduced, model.get_frame_transformation_reduced(frame, q[model.reduced_mask]),
        atol=1e-12)


def test_shutdown_stops_the_state_source(models):
    model = models('g1')
    stopped = []
    source = ChangingSource(model.num_motor)
    source.shutdown = lambda: stopped.append(True)
    model.state_subscriber = source
    model.shutdown()
    assert stopped == [True]
    assert model.state_subscriber is None


def test_reduced_limit_violations_name_the_joint(models):
    model = models('h12')
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


def test_unused_paths_are_gone(models):
    '''RobotModel has only what the controllers and the planner call: none of these'''
    model = models('h12')
    for name in ('check_valid', 'check_within_limits', 'check_collision_free', 'total_mass',
                 'zero_q_body_reduced', 'init_reduced_model', 'init_collision_model',
                 'init_reduced', 'init_collision', 'collision_model_body'):
        assert not hasattr(model, name), name
    for name in ('get_com', 'get_zmp', 'get_com_velocity', 'get_com_reduced', 'get_joint_jacobian',
                 'get_frame_twist', 'get_frame_wrench', 'get_motion_compensation',
                 'get_angular_centroidal_momentum'):
        assert not hasattr(model.dynamics, name), name


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_profile_frames_exist_in_every_model(models, robot):
    model = models(robot)
    profile = model.profile
    frames = set(profile.ee_frames + profile.planner_frames + (profile.imu_frame, 'torso_link'))
    if robot == 'g1':
        # the G1's sensors and the other IMU, which controller.imu_frame may name
        frames |= {'imu_in_torso', 'd435_link', 'mid360_link'}
    for frame in sorted(frames):
        for name in ('model', 'model_body', 'model_body_reduced'):
            assert getattr(model, name).existFrame(frame), (name, frame)


def pelvis_quaternion(model, q, imu_wxyz):
    from pyquaternion import Quaternion
    x, y, z, w = model.full_q(q, imu_quat=imu_wxyz)[3:7]
    return Quaternion(w, x, y, z)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_imu_orientation_gives_the_pelvis_orientation(models, robot):
    from pyquaternion import Quaternion
    model = models(robot)
    profile = model.profile
    first_held = profile.body_joints.index(profile.held_joints[0])
    imu = Quaternion(axis=[0.3, 1.0, 0.2], angle=0.4)
    # the held joints at 0: the IMU frame sits level on the pelvis, so the pelvis is the IMU
    q = np.zeros(model.num_motor)
    pelvis = pelvis_quaternion(model, q, imu.elements)
    assert Quaternion.absolute_distance(pelvis, imu) < 1e-6
    # the first held joint (the H1-2's torso, the G1's waist yaw) turned 0.5 rad
    q[first_held] = 0.5
    if robot == 'h12':
        # the IMU on torso_link turns with it: an IMU reading the turn has a level pelvis
        turned = Quaternion(axis=[0.0, 0.0, 1.0], angle=0.5)
        pelvis = pelvis_quaternion(model, q, turned.elements)
        assert Quaternion.absolute_distance(pelvis, Quaternion()) < 1e-6
    else:
        # the IMU on the pelvis does not: the pelvis is still the IMU
        pelvis = pelvis_quaternion(model, q, imu.elements)
        assert Quaternion.absolute_distance(pelvis, imu) < 1e-6


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_named_configs_are_within_limits(models, robot):
    model = models(robot)
    profile = model.profile
    for name, q in profile.named_configs.items():
        assert model.check_within_limits_reduced(q), (name, model.reduced_limit_violations(q))
    assert set(profile.startup_sequence) <= set(profile.named_configs)


def test_g1_named_configs_are_where_they_claim(models):
    model = models('g1')
    profile = model.profile
    data = model.model_body_reduced.createData()

    def palms(name):
        pin.framesForwardKinematics(model.model_body_reduced, data, profile.named_configs[name])
        return [data.oMf[model.model_body_reduced.getFrameId(frame)].translation.copy()
                for frame in profile.ee_frames]

    left, right = palms('home')  # forearms forward at waist height
    assert left[0] > 0.15 and right[0] > 0.15
    left, _ = palms('arms_down')  # straight down, under the shoulder
    assert abs(left[0]) < 0.05 and left[2] < 0.0
    left, right = palms('t_pose')  # straight out to the sides at shoulder height
    assert left[1] > 0.45 and right[1] < -0.45 and left[2] > 0.25
