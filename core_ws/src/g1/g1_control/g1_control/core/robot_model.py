import os
import numpy as np
import pinocchio as pin
from pyquaternion import Quaternion

from g1_control.core.channel_interface import StateSubscriber
from g1_control.core.robot_dynamics import RobotDynamics
from g1_control.core.robot_visualizer import RobotVisualizer
from g1_control.utility.joint_definition import (
    BODY_JOINTS, NUM_MOTOR,
    FREEFLYER_NQ, FREEFLYER_NV,
    FREEFLYER_QUAT
)

class RobotModel:
    def __init__(self, filename: str, imu_frame: str='imu_in_pelvis'):
        '''
        filename: G1 URDF (with or without the Dex3-1 hands; hand joints are frozen)
        imu_frame: URDF frame the rt/lowstate IMU is mounted on, used to recover
        the pelvis orientation (imu_in_pelvis or imu_in_torso)
        '''
        assert(os.path.splitext(filename)[1] == '.urdf'), \
            'Please provide a urdf file for the robot model.'
        # full model (free-flyer) for visualization and dynamics
        self.model, _, self.visual_model = self._load_urdf_freeflyer(filename)
        self.data = self.model.createData()

        # model without free-flyer for IK solver (motor only)
        self.model_body, self.collision_model_body, _ = self._load_urdf_body(filename)
        self.data_body = self.model_body.createData()
        assert self.model_body.existFrame(imu_frame), f'IMU frame {imu_frame} is not in the model'
        self.imu_frame = imu_frame

        # placeholder for visualizer and state subscriber
        self.visualizer = None
        self.state_subscriber = None

        # field variables tracking joint states (motor only, length NUM_MOTOR)
        self._q = np.zeros(NUM_MOTOR)
        self._dq = np.zeros(NUM_MOTOR)
        self._ddq = np.zeros(NUM_MOTOR)
        self._tau = np.zeros(NUM_MOTOR)

        self.dynamics = RobotDynamics(self)

        # initialize with zero joint positions
        pin.forwardKinematics(self.model, self.data, self.zero_q)
        pin.updateFramePlacements(self.model, self.data)
        pin.forwardKinematics(self.model_body, self.data_body, self.zero_q_body)
        pin.updateFramePlacements(self.model_body, self.data_body)

        # flags for additional initialization
        self.init_reduced = False
        self.init_collision = False
        # placeholder mask for reduced model (motor only, length NUM_MOTOR)
        self.reduced_mask = np.ones(NUM_MOTOR, dtype=bool)

    def init_reduced_model(self, enabled_joints):
        '''
        create reduced model from model_body (no free-flyer) for IK
        reduced_mask applies to motor-only arrays (length NUM_MOTOR)
        '''
        self.init_reduced = True
        frozen_joints = set(BODY_JOINTS) - set(enabled_joints)
        # get joint ids in model_body (no free-flyer, so no offset)
        frozen_ids_body = [self.model_body.getJointId(joint_name) for joint_name in frozen_joints]
        frozen_motor_ids = [self.model_body.joints[joint_id].idx_q for joint_id in frozen_ids_body]
        # create reduced model from model_body
        self.model_body_reduced = pin.buildReducedModel(
            self.model_body, frozen_ids_body, self.zero_q_body
        )
        self.data_body_reduced = self.model_body_reduced.createData()
        # set the reduced mask (motor only)
        self.reduced_mask[frozen_motor_ids] = False
        # store frozen ids for collision model
        self.frozen_ids_body = frozen_ids_body

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

        if self.init_reduced:
            model_collision_reduced, collision_model_reduced = pin.buildReducedModel(
                model_collision, collision_model, self.frozen_ids_body, self.zero_q_body
            )
            self.collision_model_body_reduced, self.collision_data_body_reduced = RobotModel._process_srdf(
                model_collision_reduced, collision_model_reduced, srdf_path
            )

    @staticmethod
    def _keep_body_joints(model, collision_model, visual_model):
        '''Freeze every joint that is not a body motor (e.g. the Dex3-1 fingers) at 0'''
        frozen_ids = [
            model.getJointId(name) for name in model.names[1:]
            if name not in BODY_JOINTS and name != 'root_joint'
        ]
        missing = [name for name in BODY_JOINTS if not model.existJointName(name)]
        assert not missing, f'URDF is missing body joints: {missing}'
        reduced, [collision_model, visual_model] = pin.buildReducedModel(
            model, [collision_model, visual_model], frozen_ids, pin.neutral(model)
        )
        # the controller feeds rt/lowstate q straight into these models
        order = [name for name in reduced.names[1:] if name != 'root_joint']
        assert order == BODY_JOINTS, 'model joint order must match the rt/lowcmd slot order'
        return reduced, collision_model, visual_model

    @staticmethod
    def _load_urdf_freeflyer(urdf_path):
        '''
        Load urdf with free-flyer root joint for visualization and dynamics
        Returns full model (free-flyer) with nq=7+NUM_MOTOR, nv=6+NUM_MOTOR
        '''
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            filename=urdf_path,
            package_dirs=os.path.dirname(urdf_path),
            root_joint=pin.JointModelFreeFlyer(),
        )
        return RobotModel._keep_body_joints(model, collision_model, visual_model)

    @staticmethod
    def _load_urdf_body(urdf_path):
        '''
        Load urdf without free-flyer for IK solver
        Returns model with nq=nv=NUM_MOTOR (motor only)
        '''
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            filename=urdf_path,
            package_dirs=os.path.dirname(urdf_path),
        )
        return RobotModel._keep_body_joints(model, collision_model, visual_model)

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
            'q': np.copy(self._q),
            'dq': np.copy(self._dq),
            'ddq': np.copy(self._ddq),
            'tau': np.copy(self._tau),
        }

    @property
    def state_reduced(self):
        '''
        Apply reduced_mask to motor-only state arrays
        Reduced_mask is length NUM_MOTOR
        '''
        mask_keys = {'mode', 'q', 'dq', 'ddq', 'tau', 'vol', 'motor_state', 'temperature', 'sensor'}
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
        convert motor q (NUM_MOTOR) to full pinocchio q (7+NUM_MOTOR) with pelvis orientation
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
            # get pelvis-to-imu transform from model_body (identity for imu_in_pelvis,
            # through the waist joints for imu_in_torso)
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
        Convert motor v (NUM_MOTOR) to full pinocchio v (6+NUM_MOTOR) with zero base velocity
        For use with full model (free-flyer)
        '''
        full_v = np.zeros(self.model.nv)
        full_v[FREEFLYER_NV:] = v
        return full_v

    @property
    def zero_q(self):
        '''Zero configuration for full model (free-flyer), length 7+NUM_MOTOR'''
        zero = np.zeros(self.model.nq)
        zero[FREEFLYER_QUAT] = [0, 0, 0, 1]  # identity quaternion xyzw
        return zero

    @property
    def zero_q_body(self):
        '''Zero configuration for model_body (no free-flyer), length NUM_MOTOR'''
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

    def init_visualizer(self):
        '''Initialize meshcat visualizer'''
        if self.visualizer is None:
            self.visualizer = RobotVisualizer(self)
        self.visualizer.init_visualizer()

    def config_visualizer(self,
                          show_com=None,
                          show_zmp=None,
                          show_sensors=None,
                          wrench_frames=None):
        '''Configure meshcat visualizations'''
        assert self.visualizer is not None, 'Visualizer must be initialized first.'
        self.visualizer.config_visualizer(
            show_com=show_com,
            show_zmp=show_zmp,
            show_sensors=show_sensors,
            wrench_frames=wrench_frames,
        )

    def update_visualizer(self):
        '''Update robot visualization and all enabled visualizations'''
        if self.visualizer is not None:
            self.visualizer.update_visualizer()

    def init_subscriber(self, low_state_topic='rt/lowstate'):
        self.state_subscriber = StateSubscriber(low_state_topic=low_state_topic)
        print('StateSubscriber initialized.', flush=True)

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
        q is motor-only (NUM_MOTOR)
        '''
        return self.check_within_limits(q) and self.check_collision_free(q)

    def check_within_limits(self, q):
        '''
        Check if the given motor joint position violates joint limits
        q is motor-only (NUM_MOTOR)
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

    def check_collision_free(self, q):
        '''
        check if the given joint position violates collision constraints
        q is motor-only (NUM_MOTOR), uses collision_model_body (no free-flyer)
        Without a collision model (the G1 has none yet) this always returns True
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
        Without a collision model (the G1 has none yet) this always returns True
        '''
        assert(self.init_reduced), 'Reduced model is not initialized.'
        if not self.init_collision:
            return True
        return not pin.computeCollisions(
            self.model_body_reduced, self.data_body_reduced,
            self.collision_model_body_reduced, self.collision_data_body_reduced, q_reduced,
            stop_at_first_collision=True
        )
