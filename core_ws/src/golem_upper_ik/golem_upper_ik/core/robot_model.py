'''
RobotModel: one robot's Pinocchio models, built from its profile (golem_upper_ik.profile).

- model, data: the free-flyer model, for dynamics and the pelvis orientation
- model_body, data_body: the body joints only, without the free-flyer, for the IK
- model_body_reduced, data_body_reduced: model_body with every joint but the IK joints
  frozen, for the IK and the planner
- collision_model_body_reduced, collision_data_body_reduced: the reduced model's
  self-collision model, when the profile names a sphere URDF and SRDF (has_collision);
  without one, every configuration counts as collision-free

Every URDF joint that is not a body joint (the grippers', the hands') is frozen out of every
model, so the joints left are the profile's body joints in /lowcmd slot order and /lowstate
values feed the models directly. Joint arrays (q, dq, tau) hold one value per body joint in
slot order; the _reduced ones one per IK joint, in the profile's enabled_joints order.

The measured state comes from a StateSource (state_subscriber).
'''
from collections.abc import Mapping
import os
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pinocchio as pin
from pyquaternion import Quaternion

from golem_upper_ik.core.robot_dynamics import (
    FREEFLYER_NQ,
    FREEFLYER_NV,
    FREEFLYER_QUAT,
    RobotDynamics,
)
from golem_upper_ik.profile import RobotProfile

# the measured-state arrays the package reads, one value per body joint in slot order
STATE_KEYS = ('q', 'dq')


@runtime_checkable
class StateSource(Protocol):
    '''
    Where RobotModel reads the measured state: the ROS I/O's /lowstate subscriber.

    state: one consistent sample per read, every value from the same /lowstate message. A
        mapping with 'q' (rad) and 'dq' (rad/s), each an np.ndarray with one value per body
        joint in /lowcmd slot order, and optionally 'imu_state', an object whose `quaternion`
        is the IMU orientation (w, x, y, z). update_kinematics and state_reduced each take
        everything they use from one read; nothing modifies what a read returns.
    staleness_ms: the age of the newest sample, ms; inf before the first sample
    shutdown(): stops the source; RobotModel.shutdown() calls it
    '''

    @property
    def state(self) -> Mapping[str, Any]: ...

    @property
    def staleness_ms(self) -> float: ...

    def shutdown(self) -> None: ...


class FixedState:
    '''A StateSource that always reports the same state: q, dq (default 0), optionally an IMU'''

    def __init__(self, q, dq=None, imu_quaternion=None):
        self._q = np.array(q, dtype=float)
        self._dq = np.zeros_like(self._q) if dq is None else np.array(dq, dtype=float)
        self._imu_state = None
        if imu_quaternion is not None:
            self._imu_state = SimpleNamespace(quaternion=np.array(imu_quaternion, dtype=float))
        self.staleness_ms = 0.0

    @property
    def state(self) -> Mapping[str, Any]:
        state = {'q': self._q.copy(), 'dq': self._dq.copy()}
        if self._imu_state is not None:
            state['imu_state'] = self._imu_state
        return state

    def shutdown(self) -> None:
        pass


class RobotModel:
    def __init__(self, profile: RobotProfile, imu_frame: str | None = None):
        '''
        profile: the robot's URDF, body joints, IK joints and self-collision files
        imu_frame: the URDF frame the /lowstate IMU is mounted on, from which full_q recovers
        the pelvis orientation; default the profile's
        '''
        filename = profile.urdf_path
        if os.path.splitext(filename)[1] != '.urdf':
            raise ValueError(f'robot {profile.robot}: the robot model must be a .urdf file, '
                             f'not {filename}')
        self.profile = profile
        self.body_joints = tuple(profile.body_joints)
        self.num_motor = len(self.body_joints)
        # body-joint indices of each arm's IK joints
        self.left_arm_index = [self.body_joints.index(joint) for joint in profile.enabled_joints
                               if joint.startswith('left_')]
        self.right_arm_index = [self.body_joints.index(joint)
                                for joint in profile.enabled_joints if joint.startswith('right_')]

        # full model (free-flyer) for dynamics
        self.model, _, _ = self._load_urdf_freeflyer(filename)
        self.data = self.model.createData()
        imu_frame = profile.imu_frame if imu_frame is None else imu_frame
        if not self.model.existFrame(imu_frame):
            raise ValueError(f'robot {profile.robot}: IMU frame {imu_frame} is not in '
                             f'{filename}')
        self.imu_frame = imu_frame

        # model without free-flyer for IK solver (motor only)
        self.model_body, _, _ = self._load_urdf_body(filename)
        self.data_body = self.model_body.createData()

        # the measured state; without a source the model is at rest at q = 0
        self.state_subscriber: StateSource | None = None

        self.dynamics = RobotDynamics(self)

        # initialize with zero joint positions
        pin.forwardKinematics(self.model, self.data, self.zero_q)
        pin.updateFramePlacements(self.model, self.data)
        pin.forwardKinematics(self.model_body, self.data_body, self.zero_q_body)
        pin.updateFramePlacements(self.model_body, self.data_body)

        self._init_reduced_model(profile.enabled_joints)
        self.has_collision = profile.collision is not None
        if self.has_collision:
            self._init_collision_model(*profile.collision)

    def _init_reduced_model(self, enabled_joints):
        '''
        create reduced model from model_body (no free-flyer) for IK, the frozen joints locked
        at 0; reduced_mask picks the IK joints out of a motor-only array (length num_motor)
        '''
        frozen_joints = [joint for joint in self.body_joints if joint not in enabled_joints]
        # get joint ids in model_body (no free-flyer, so no offset)
        frozen_ids_body = [self.model_body.getJointId(joint_name) for joint_name in frozen_joints]
        frozen_motor_ids = [self.model_body.joints[joint_id].idx_q for joint_id in frozen_ids_body]
        # store frozen ids for collision model and rebuilds
        self.frozen_ids_body = frozen_ids_body
        self.reduced_mask = np.ones(self.num_motor, dtype=bool)
        self.reduced_mask[frozen_motor_ids] = False
        self._build_reduced_model(self.zero_q_body)

    def _build_reduced_model(self, reference_q):
        self.model_body_reduced = pin.buildReducedModel(
            self.model_body, self.frozen_ids_body, reference_q
        )
        self.data_body_reduced = self.model_body_reduced.createData()

    def rebuild_reduced_model(self, reference_q=None):
        '''
        Re-lock the frozen joints at reference_q (default: the measured
        state). Freezing bakes the frozen joints (torso or waist, legs) into
        fixed transforms, so a frozen joint that physically sits away from the
        frozen reference makes reduced-model FK disagree with the robot (and TF)
        by a constant end-effector offset no IK iteration can remove.
        Callers must rebind any object holding references to the reduced
        model (see IKSolver.rebuild_reduced).
        '''
        if reference_q is None:
            reference_q = self.state['q']
        reference_q = np.asarray(reference_q, dtype=float).copy()
        self._build_reduced_model(reference_q)
        if self.has_collision:
            self._build_reduced_collision_model(reference_q)

    def _init_collision_model(self, urdf_path, srdf_path):
        '''
        The reduced model's self-collision model, from another urdf (sphere collision) and
        the srdf of the pairs it leaves out
        '''
        # the unreduced sphere model is kept to re-lock the reduced one
        self._collision_model_full, self._collision_geom_full, _ = self._load_urdf_body(urdf_path)
        self._collision_srdf_path = srdf_path
        self._build_reduced_collision_model(self.zero_q_body)

    def _build_reduced_collision_model(self, reference_q):
        model_collision_reduced, collision_model_reduced = pin.buildReducedModel(
            self._collision_model_full, self._collision_geom_full,
            self.frozen_ids_body, reference_q
        )
        (self.collision_model_body_reduced,
         self.collision_data_body_reduced) = RobotModel._process_srdf(
            model_collision_reduced, collision_model_reduced, self._collision_srdf_path
        )

    def _keep_body_joints(self, urdf_path, model, collision_model, visual_model):
        '''
        Freeze every joint that is not a body joint (the grippers', the hands') at Pinocchio's
        neutral configuration (angle 0, also for a continuous joint)
        '''
        robot = self.profile.robot
        missing = [name for name in self.body_joints if not model.existJointName(name)]
        if missing:
            raise ValueError(f'robot {robot}: {urdf_path} is missing body joints {missing}')
        frozen_ids = [
            model.getJointId(name) for name in model.names[1:]
            if name not in self.body_joints and name != 'root_joint'
        ]
        reduced, [collision_model, visual_model] = pin.buildReducedModel(
            model, [collision_model, visual_model], frozen_ids, pin.neutral(model)
        )
        # the controller feeds /lowstate q straight into these models
        order = tuple(name for name in reduced.names[1:] if name != 'root_joint')
        if order != self.body_joints:
            raise ValueError(f'robot {robot}: the joints of {urdf_path} are in the order '
                             f'{order}, not the /lowcmd slot order {self.body_joints}')
        return reduced, collision_model, visual_model

    def _load_urdf_freeflyer(self, urdf_path):
        '''
        Load urdf with free-flyer root joint for dynamics
        Returns full model (free-flyer) with nq=7+num_motor, nv=6+num_motor
        '''
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            filename=urdf_path,
            package_dirs=os.path.dirname(urdf_path),
            root_joint=pin.JointModelFreeFlyer(),
        )
        return self._keep_body_joints(urdf_path, model, collision_model, visual_model)

    def _load_urdf_body(self, urdf_path):
        '''
        Load urdf without free-flyer for IK solver
        Returns model with nq=nv=num_motor (motor only)
        '''
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            filename=urdf_path,
            package_dirs=os.path.dirname(urdf_path),
        )
        return self._keep_body_joints(urdf_path, model, collision_model, visual_model)

    @staticmethod
    def _process_srdf(model, collision_model, srdf_path):
        collision_model.addAllCollisionPairs()
        pin.removeCollisionPairs(model, collision_model, srdf_path)
        collision_data = pin.GeometryData(collision_model)
        collision_data.enable_contact = True
        return collision_model, collision_data

    @property
    def state(self) -> Mapping[str, Any]:
        '''One sample of the measured state (see StateSource); without a source, at rest'''
        if self.state_subscriber is not None:
            return self.state_subscriber.state
        return {key: np.zeros(self.num_motor) for key in STATE_KEYS}

    @property
    def state_reduced(self):
        '''q and dq of the IK joints, from one sample of the measured state'''
        return self._reduce(self.state)

    def _reduce(self, state):
        reduced = {}
        for key in STATE_KEYS:
            values = np.asarray(state[key])
            if values.shape != (self.num_motor,):
                raise ValueError(f'measured {key} has shape {values.shape}: it must hold one '
                                 f'value per body joint ({self.num_motor})')
            reduced[key] = np.copy(values[self.reduced_mask])
        return reduced

    @staticmethod
    def _imu_quaternion(state):
        '''The IMU orientation (w, x, y, z) of a state sample, or None without one'''
        imu_state = state.get('imu_state')
        return None if imu_state is None else imu_state.quaternion

    def full_q(self, q: np.ndarray, imu_quat=None) -> np.ndarray:
        '''
        convert motor q (num_motor) to full pinocchio q (7+num_motor) with pelvis orientation
        if imu_quat provided (unitree wxyz), use it; otherwise the measured state's IMU, when
        a state source is set and its sample has one; otherwise identity
        for use with full model with free-flyer base
        '''
        if imu_quat is None and self.state_subscriber is not None:
            imu_quat = self._imu_quaternion(self.state)
        return self._full_q(q, imu_quat)

    def _full_q(self, q, imu_quat):
        full_q = np.zeros(self.model.nq)
        if imu_quat is not None:
            imu_quat = np.asarray(imu_quat, dtype=float).reshape(4)
            # Validate IMU quaternion before normalization to avoid propagating NaN/Inf
            if not np.isfinite(imu_quat).all():
                # fallback to identity quaternion (wxyz) if IMU data is invalid
                imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            else:
                imu_norm = np.linalg.norm(imu_quat)
                if (imu_norm < 1e-8) or (not np.isfinite(imu_norm)):
                    imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                else:
                    imu_quat = imu_quat / imu_norm

            imu_frame_quat = Quaternion(imu_quat[0], imu_quat[1], imu_quat[2], imu_quat[3])
            # get pelvis-to-imu transform from model_body (through the torso or waist joints
            # when the IMU sits above them)
            pin.forwardKinematics(self.model_body, self.data_body, q)
            pin.updateFramePlacements(self.model_body, self.data_body)
            pelvis_to_imu = self.data_body.oMf[self.model_body.getFrameId(self.imu_frame)]
            pelvis_to_imu_quat = Quaternion(matrix=pelvis_to_imu.rotation)
            # pelvis_quat = imu_quat * inv(pelvis_to_imu_quat)
            pelvis_quat = (imu_frame_quat * pelvis_to_imu_quat.inverse).normalised
            # convert to pinocchio xyzw
            full_q[FREEFLYER_QUAT] = [pelvis_quat.x, pelvis_quat.y, pelvis_quat.z, pelvis_quat.w]
        else:
            full_q[FREEFLYER_QUAT] = [0, 0, 0, 1]  # identity quaternion xyzw
        full_q[FREEFLYER_NQ:] = q
        return full_q

    def full_v(self, v: np.ndarray) -> np.ndarray:
        '''
        Convert motor v (num_motor) to full pinocchio v (6+num_motor) with zero base velocity
        For use with full model (free-flyer)
        '''
        full_v = np.zeros(self.model.nv)
        full_v[FREEFLYER_NV:] = v
        return full_v

    @property
    def zero_q(self):
        '''Zero configuration for full model (free-flyer), length 7+num_motor'''
        zero = np.zeros(self.model.nq)
        zero[FREEFLYER_QUAT] = [0, 0, 0, 1]  # identity quaternion xyzw
        return zero

    @property
    def zero_q_body(self):
        '''Zero configuration for model_body (no free-flyer), length num_motor'''
        return np.zeros(self.model_body.nq)

    def shutdown(self):
        if self.state_subscriber is not None:
            self.state_subscriber.shutdown()
            self.state_subscriber = None
        print('RobotModel shutdown', flush=True)

    @property
    def state_staleness_ms(self):
        '''How stale the subscribed low state is [ms]; 0 without a subscriber'''
        if self.state_subscriber is None:
            return 0.0
        return self.state_subscriber.staleness_ms

    def update_kinematics(self, imu_quat=None):
        '''update forward kinematics for all models, from one sample of the measured state'''
        state = self.state
        if imu_quat is None:
            imu_quat = self._imu_quaternion(state)
        # update full model (free-flyer) with current joint positions
        q_full = self._full_q(state['q'], imu_quat)
        dq_full = self.full_v(state['dq'])
        pin.forwardKinematics(self.model, self.data, q_full, dq_full)
        pin.updateFramePlacements(self.model, self.data)
        # update model_body (no free-flyer)
        pin.forwardKinematics(self.model_body, self.data_body, state['q'], state['dq'])
        pin.updateFramePlacements(self.model_body, self.data_body)
        # update reduced model
        reduced = self._reduce(state)
        pin.forwardKinematics(self.model_body_reduced, self.data_body_reduced,
                              reduced['q'], reduced['dq'])
        pin.updateFramePlacements(self.model_body_reduced, self.data_body_reduced)

    def _get_frame_transformation(self, frame_name, q: np.ndarray=None):
        frame_id = self.model.getFrameId(frame_name)
        if q is not None:
            q_full = self.full_q(q)
            data_temp = self.model.createData()
            pin.forwardKinematics(self.model, data_temp, q_full)
            pin.updateFramePlacements(self.model, data_temp)
            transformation = data_temp.oMf[frame_id]
        else:
            self.update_kinematics()
            transformation = self.data.oMf[frame_id]
        return transformation

    def get_frame_transformation(self, frame_name: str, q: np.ndarray=None):
        return self._get_frame_transformation(frame_name, q).np

    def get_frame_position(self, frame_name: str, q: np.ndarray=None):
        return self._get_frame_transformation(frame_name, q).translation

    def get_frame_rotation(self, frame_name: str, q: np.ndarray = None):
        return self._get_frame_transformation(frame_name, q).rotation

    def get_frame_transformation_body(self, frame_name: str, q: np.ndarray):
        '''
        The frame's 4x4 placement in the pelvis frame (model_body, no IMU orientation) at q,
        one value per body joint. Computed on data of its own
        '''
        data = self.model_body.createData()
        pin.forwardKinematics(self.model_body, data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(self.model_body, data)
        return data.oMf[self.model_body.getFrameId(frame_name)].np

    def _get_frame_transformation_reduced(self, frame_name, q_reduced: np.ndarray=None):
        '''Get frame transformation for the reduced model (model_body_reduced)'''
        frame_id = self.model_body_reduced.getFrameId(frame_name)
        if q_reduced is not None:
            data_body_reduced_temp = self.model_body_reduced.createData()
            pin.forwardKinematics(self.model_body_reduced, data_body_reduced_temp, q_reduced)
            pin.updateFramePlacements(self.model_body_reduced, data_body_reduced_temp)
            transformation = data_body_reduced_temp.oMf[frame_id]
        else:
            self.update_kinematics()
            transformation = self.data_body_reduced.oMf[frame_id]
        return transformation

    def get_frame_transformation_reduced(self, frame_name: str, q_reduced: np.ndarray=None):
        return self._get_frame_transformation_reduced(frame_name, q_reduced).np

    def get_frame_position_reduced(self, frame_name: str, q_reduced: np.ndarray=None):
        return self._get_frame_transformation_reduced(frame_name, q_reduced).translation

    def get_frame_rotation_reduced(self, frame_name: str, q_reduced: np.ndarray=None):
        return self._get_frame_transformation_reduced(frame_name, q_reduced).rotation

    def check_within_limits_reduced(self, q_reduced):
        '''Check if the given joint position violates joint limits for the reduced model'''
        return (
            np.all(q_reduced <= self.model_body_reduced.upperPositionLimit) and
            np.all(q_reduced >= self.model_body_reduced.lowerPositionLimit)
        )

    def _reduced_idx_to_name(self):
        '''Map each reduced-model configuration index to its joint name.'''
        model = self.model_body_reduced
        idx_to_name = {}
        for jid in range(1, model.njoints):
            joint = model.joints[jid]
            for k in range(joint.nq):
                idx_to_name[joint.idx_q + k] = model.names[jid]
        return idx_to_name

    def reduced_limit_violations(self, q_reduced, tol=0.0):
        '''
        Report reduced-model joints whose position is outside limits by more
        than tol. Returns a list of dicts with the joint name, config index,
        value, the violated bound, and the overshoot magnitude (>= 0).
        '''
        lower = self.model_body_reduced.lowerPositionLimit
        upper = self.model_body_reduced.upperPositionLimit
        idx_to_name = self._reduced_idx_to_name()
        violations = []
        for i in range(len(q_reduced)):
            if q_reduced[i] < lower[i] - tol:
                bound, overshoot = float(lower[i]), float(lower[i] - q_reduced[i])
            elif q_reduced[i] > upper[i] + tol:
                bound, overshoot = float(upper[i]), float(q_reduced[i] - upper[i])
            else:
                continue
            violations.append({
                'name': idx_to_name.get(i, f'q[{i}]'),
                'index': i,
                'value': float(q_reduced[i]),
                'lower': float(lower[i]),
                'upper': float(upper[i]),
                'bound': bound,
                'overshoot': overshoot,
            })
        return violations

    def closest_collision_pairs(self, q_reduced, count=3):
        '''
        The count enabled self-collision pairs closest at q_reduced, closest first, as
        (sphere, sphere, distance in m; negative: they overlap); empty without a collision
        model
        '''
        if not self.has_collision:
            return []
        geometry = self.collision_model_body_reduced
        data = self.collision_data_body_reduced
        pin.computeDistances(self.model_body_reduced, self.data_body_reduced, geometry, data,
                             np.asarray(q_reduced, dtype=np.float64))
        objects = geometry.geometryObjects
        pairs = sorted(
            ((objects[pair.first].name, objects[pair.second].name, float(result.min_distance))
             for pair, result in zip(geometry.collisionPairs, data.distanceResults)),
            key=lambda pair: pair[2])
        return pairs[:count]

    def check_collision_free_reduced(self, q_reduced):
        '''
        Check collision for the reduced model (model_body_reduced)
        Without a collision model this always returns True
        '''
        if not self.has_collision:
            return True
        return not pin.computeCollisions(
            self.model_body_reduced, self.data_body_reduced,
            self.collision_model_body_reduced, self.collision_data_body_reduced, q_reduced,
            stop_at_first_collision=True
        )
