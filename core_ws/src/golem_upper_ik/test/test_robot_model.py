"""Check RobotModel and RobotDynamics on both robots."""
import numpy as np
import pinocchio as pin
import pytest

# each arm's seven IK joints, by /lowcmd slot: (left, right)
ARM_SLOTS = {
    'h12': (range(13, 20), range(20, 27)),
    'g1': (range(15, 22), range(22, 29)),
}


def profile_of(robot):
    """Return robot's RobotProfile."""
    from golem_upper_ik.profile import load_profile
    return load_profile(robot)


def model_of(robot, **kwargs):
    """Return a new RobotModel of robot."""
    from golem_upper_ik.core.robot_model import RobotModel
    return RobotModel(profile_of(robot), **kwargs)


def fixed_state(q, dq=None, imu_quaternion=None):
    """Return a FixedState at q, dq and an IMU quaternion."""
    from golem_upper_ik.core.robot_model import FixedState
    return FixedState(q, dq=dq, imu_quaternion=imu_quaternion)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_models_follow_the_profile(models, robot):
    """The models keep the body joints in slot order and solve for the IK joints only."""
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
    """The H1-2 has a self-collision model, and home is collision-free."""
    model = models('h12')
    assert model.has_collision is True
    assert len(model.collision_model_body_reduced.collisionPairs) > 0
    assert model.check_collision_free_reduced(model.profile.named_configs['home'])


def pair_distances(model, q_reduced):
    """Return each enabled collision pair's distance at q_reduced, m (negative: they overlap)."""
    geometry = model.collision_model_body_reduced
    data = model.collision_data_body_reduced
    pin.computeDistances(model.model_body_reduced, model.data_body_reduced, geometry, data,
                         q_reduced)
    return np.array([result.min_distance for result in data.distanceResults])


def test_g1_model_has_collision(models):
    """Every G1 named config but arms_down keeps every pair d_min apart; arms_down collides.

    The G1's spheres and pairs are counted, and home's closest pair (an upper arm against the
    torso) pins CL_Assets' g1_sphere.urdf.
    """
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
    # home's closest pairs are the upper arms against the torso. The distance pins CL_Assets'
    # g1_sphere.urdf: a changed sphere model moves it
    first, second, distance = model.closest_collision_pairs(model.profile.named_configs['home'],
                                                            count=1)[0]
    assert distance == pytest.approx(0.0249, abs=1e-3)
    assert 'torso_link' in first + second, (first, second)


def test_rebuild_reduced_relocks_at_measured(models):
    """Re-locking puts the frozen torso at its measured angle, spheres included."""
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
            """Return the torso sphere's position in a geometry model at q_model."""
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
    """An IMU on the pelvis ignores the waist, one on the torso turns with it; bad frames fail."""
    profile = profile_of('g1')
    on_pelvis = models('g1')
    on_torso = model_of('g1', imu_frame='imu_in_torso')
    assert on_pelvis.imu_frame == profile.imu_frame == 'imu_in_pelvis'
    assert on_torso.imu_frame == 'imu_in_torso'

    def pelvis_turn(model, waist_yaw):
        """Return how far (rad) the pelvis turns with waist_yaw, the IMU reading the same."""
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
    """A FixedState is a StateSource; the model reads its state and refuses a mis-sized one."""
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
    """Report a newer sample on every read, each q a step further."""

    def __init__(self, count):
        """Start before the first read, never stale."""
        self.count = count
        self.reads = 0
        self.staleness_ms = 0.0

    @property
    def state(self):
        """Return the next sample: q 0.1 per read so far, the IMU level."""
        self.reads += 1
        level = np.array([1.0, 0.0, 0.0, 0.0])
        return {'q': np.full(self.count, 0.1 * self.reads), 'dq': np.zeros(self.count),
                'imu_state': type('Imu', (), {'quaternion': level})()}

    def shutdown(self):
        """Do nothing."""
        pass


def test_update_kinematics_reads_the_state_once(models):
    """update_kinematics puts every model at one sample of the state."""
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
    """RobotModel.shutdown stops and detaches its state source."""
    model = models('g1')
    stopped = []
    source = ChangingSource(model.num_motor)
    source.shutdown = lambda: stopped.append(True)
    model.state_subscriber = source
    model.shutdown()
    assert stopped == [True]
    assert model.state_subscriber is None


def test_reduced_limit_violations_name_the_joint(models):
    """A joint past its limit is reported by name, bound and overshoot, within tol not."""
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


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_profile_frames_exist_in_every_model(models, robot):
    """Every frame the profile and the configs name is in every model."""
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
    """Return the pelvis orientation full_q gives for q and an IMU orientation (w, x, y, z)."""
    from pyquaternion import Quaternion
    x, y, z, w = model.full_q(q, imu_quat=imu_wxyz)[3:7]
    return Quaternion(w, x, y, z)


@pytest.mark.parametrize('robot', ['h12', 'g1'])
def test_imu_orientation_gives_the_pelvis_orientation(models, robot):
    """The pelvis orientation follows the IMU, through the held joint when the IMU is above it."""
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
    """Every named config is within the joint limits."""
    model = models(robot)
    profile = model.profile
    for name, q in profile.named_configs.items():
        assert model.check_within_limits_reduced(q), (name, model.reduced_limit_violations(q))
    assert set(profile.startup_sequence) <= set(profile.named_configs)


def test_g1_named_configs_are_where_they_claim(models):
    """The G1's home, arms_down and t_pose put the palms where robot.yaml says."""
    model = models('g1')
    profile = model.profile
    data = model.model_body_reduced.createData()

    def palms(name):
        """Return the palms' positions at a named config, m."""
        pin.framesForwardKinematics(model.model_body_reduced, data, profile.named_configs[name])
        return [data.oMf[model.model_body_reduced.getFrameId(frame)].translation.copy()
                for frame in profile.ee_frames]

    left, right = palms('home')  # forearms forward at waist height
    assert left[0] > 0.15 and right[0] > 0.15
    left, _ = palms('arms_down')  # straight down, under the shoulder
    assert abs(left[0]) < 0.05 and left[2] < 0.0
    left, right = palms('t_pose')  # straight out to the sides at shoulder height
    assert left[1] > 0.45 and right[1] < -0.45 and left[2] > 0.25
