'''
RobotModel: one robot's Pinocchio models, built from its profile (golem_upper_ik.profile).

- model, data: the free-flyer model, for dynamics and the pelvis orientation
- model_body, data_body: the body joints only, without the free-flyer, for the IK
- model_body_reduced, data_body_reduced: model_body with every joint but the IK joints
  frozen, for the IK and the planner
- collision_model_body(_reduced), collision_data_body(_reduced): the self-collision model,
  when the profile names a sphere URDF and SRDF; without one, every configuration counts as
  collision-free

Every URDF joint that is not a body joint (the grippers', the hands') is frozen out of every
model, so the joints left are the profile's body joints in /lowcmd slot order and /lowstate
values feed the models directly. Joint arrays (q, dq, tau) hold one value per body joint in
slot order; the _reduced ones one per IK joint, in the profile's enabled_joints order.
'''
import os

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

        # model without free-flyer for IK solver (motor only)
        self.model_body, self.collision_model_body, _ = self._load_urdf_body(filename)
        self.data_body = self.model_body.createData()
        imu_frame = profile.imu_frame if imu_frame is None else imu_frame
        if not self.model_body.existFrame(imu_frame):
            raise ValueError(f'robot {profile.robot}: IMU frame {imu_frame} is not in '
                             f'{filename}')
        self.imu_frame = imu_frame

        # The measured state source, set by the ROS I/O: any object with a `state` dict
        # (the keys of `state` below, plus 'imu_state' with a w, x, y, z `quaternion`), a
        # `staleness_ms` and a shutdown(). Without one, `state` reads the local fields below.
        self.state_subscriber = None

        # field variables tracking joint states (motor only, length num_motor)
        self._q = np.zeros(self.num_motor)
        self._dq = np.zeros(self.num_motor)
        self._ddq = np.zeros(self.num_motor)
        self._tau = np.zeros(self.num_motor)

        self.dynamics = RobotDynamics(self)

        # initialize with zero joint positions
        pin.forwardKinematics(self.model, self.data, self.zero_q)
        pin.updateFramePlacements(self.model, self.data)
        pin.forwardKinematics(self.model_body, self.data_body, self.zero_q_body)
        pin.updateFramePlacements(self.model_body, self.data_body)
        self.total_mass = pin.computeTotalMass(self.model)

        # flags for additional initialization
        self.init_reduced = False
        self.init_collision = False
        # placeholder mask for reduced model (motor only, length num_motor)
        self.reduced_mask = np.ones(self.num_motor, dtype=bool)

        self.init_reduced_model(profile.enabled_joints)
        if profile.collision is not None:
            self.init_collision_model(*profile.collision)

    @property
    def has_collision(self) -> bool:
        '''Whether the model has a self-collision model (the profile names one)'''
        return self.init_collision

    def init_reduced_model(self, enabled_joints):
        '''
        create reduced model from model_body (no free-flyer) for IK
        reduced_mask applies to motor-only arrays (length num_motor)
        '''
        self.init_reduced = True
        frozen_joints = [joint for joint in self.body_joints if joint not in enabled_joints]
        # get joint ids in model_body (no free-flyer, so no offset)
        frozen_ids_body = [self.model_body.getJointId(joint_name) for joint_name in frozen_joints]
        frozen_motor_ids = [self.model_body.joints[joint_id].idx_q for joint_id in frozen_ids_body]
        # store frozen ids for collision model and rebuilds
        self.frozen_ids_body = frozen_ids_body
        # set the reduced mask (motor only)
        self.reduced_mask = np.ones(self.num_motor, dtype=bool)
        self.reduced_mask[frozen_motor_ids] = False
        # create reduced model from model_body
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
        assert(self.init_reduced), 'Reduced model is not initialized.'
        if reference_q is None:
            reference_q = self.state['q']
        reference_q = np.asarray(reference_q, dtype=float).copy()
        self._build_reduced_model(reference_q)
        if self.init_collision:
            self._build_reduced_collision_model(reference_q)

    def init_collision_model(self, urdf_path, srdf_path):
        '''
        Initialize collision model from another urdf (sphere collision)
        Uses model_body (no free-flyer) for collision checking
        '''
        self.init_collision = True
        # load collision model without free-flyer
        model_collision, collision_model, _ = self._load_urdf_body(urdf_path)
        self.collision_model_body, self.collision_data_body = self._process_srdf(
            model_collision, collision_model, srdf_path
        )
        # keep the unreduced collision model and srdf path for reduced rebuilds
        self._collision_model_full = model_collision
        self._collision_geom_full = collision_model
        self._collision_srdf_path = srdf_path

        if self.init_reduced:
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
    def state(self):
        # if state_subscriber exists, use its state dict
        if self.state_subscriber is not None:
            return self.state_subscriber.state
        # otherwise, fallback to local fields
        return {
            'time_stamp': 0.0,
            'arrival_monotonic': 0.0,
            'tick': 0,
            'unwrapped_tick': 0,
            'tick_valid': False,
            'sequence': 0,
            'received': False,
            'q': np.copy(self._q),
            'dq': np.copy(self._dq),
            'ddq': np.copy(self._ddq),
            'tau': np.copy(self._tau),
        }

    @property
    def state_reduced(self):
        '''
        Apply reduced_mask to motor-only state arrays
        Reduced_mask is length num_motor
        '''
        mask_keys = {'mode', 'q', 'dq', 'ddq', 'tau', 'vol', 'motor_state', 'temperature',
                     'sensor'}
        state_reduced = {}
        for k, v in self.state.items():
            if k in mask_keys:
                assert v.shape[0] == self.reduced_mask.size, 'mask length mismatch'
                state_reduced[k] = np.copy(v[self.reduced_mask])
            else:
                state_reduced[k] = v
        return state_reduced

    def full_q(self, q: np.ndarray, imu_quat=None) -> np.ndarray:
        '''
        convert motor q (num_motor) to full pinocchio q (7+num_motor) with pelvis orientation
        if imu_quat provided (unitree wxyz), use it; otherwise try state subscriber
        for use with full model with free-flyer base
        '''
        full_q = np.zeros(self.model.nq)
        # get imu quaternion
        if imu_quat is None and self.state_subscriber is not None and 'imu_state' in self.state:
            imu_quat = self.state['imu_state'].quaternion  # unitree wxyz

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

    @property
    def zero_q_body_reduced(self):
        '''Zero configuration for model_body_reduced'''
        return np.zeros(self.model_body_reduced.nq)

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
        '''update forward kinematics for all models'''
        # update full model (free-flyer) with current joint positions
        q_full = self.full_q(self.state['q'], imu_quat=imu_quat)
        dq_full = self.full_v(self.state['dq'])
        pin.forwardKinematics(self.model, self.data, q_full, dq_full)
        pin.updateFramePlacements(self.model, self.data)
        # update model_body (no free-flyer)
        pin.forwardKinematics(self.model_body, self.data_body, self.state['q'], self.state['dq'])
        pin.updateFramePlacements(self.model_body, self.data_body)
        # update reduced model if initialized
        if self.init_reduced:
            pin.forwardKinematics(self.model_body_reduced, self.data_body_reduced,
                                  self.state_reduced['q'], self.state_reduced['dq'])
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

    def _get_frame_transformation_reduced(self, frame_name, q_reduced: np.ndarray=None):
        '''Get frame transformation for the reduced model (model_body_reduced)'''
        assert(self.init_reduced), 'Reduced model is not initialized.'
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
        assert(self.init_reduced), 'Reduced model is not initialized.'
        return self._get_frame_transformation_reduced(frame_name, q_reduced).np

    def get_frame_position_reduced(self, frame_name: str, q_reduced: np.ndarray=None):
        assert(self.init_reduced), 'Reduced model is not initialized.'
        return self._get_frame_transformation_reduced(frame_name, q_reduced).translation

    def get_frame_rotation_reduced(self, frame_name: str, q_reduced: np.ndarray=None):
        assert(self.init_reduced), 'Reduced model is not initialized.'
        return self._get_frame_transformation_reduced(frame_name, q_reduced).rotation

    def check_valid(self, q):
        '''
        Check if the given joint position is valid
        q is motor-only (num_motor)
        '''
        return self.check_within_limits(q) and self.check_collision_free(q)

    def check_within_limits(self, q):
        '''
        Check if the given motor joint position violates joint limits
        q is motor-only (num_motor)
        '''
        q_full = self.full_q(q)
        return (
            np.all(q_full <= self.model.upperPositionLimit) and
            np.all(q_full >= self.model.lowerPositionLimit)
        )

    def check_within_limits_reduced(self, q_reduced):
        '''Check if the given joint position violates joint limits for the reduced model'''
        assert(self.init_reduced), 'Reduced model is not initialized.'
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
        assert(self.init_reduced), 'Reduced model is not initialized.'
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

    def check_collision_free(self, q):
        '''
        check if the given joint position violates collision constraints
        q is motor-only (num_motor), uses collision_model_body (no free-flyer)
        Without a collision model this always returns True
        '''
        if not self.init_collision:
            return True
        return not pin.computeCollisions(
            self.model_body, self.data_body,
            self.collision_model_body, self.collision_data_body, q,
            stop_at_first_collision=True
        )

    def check_collision_free_reduced(self, q_reduced):
        '''
        Check collision for the reduced model (model_body_reduced)
        Without a collision model this always returns True
        '''
        assert(self.init_reduced), 'Reduced model is not initialized.'
        if not self.init_collision:
            return True
        return not pin.computeCollisions(
            self.model_body_reduced, self.data_body_reduced,
            self.collision_model_body_reduced, self.collision_data_body_reduced, q_reduced,
            stop_at_first_collision=True
        )
