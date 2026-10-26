import numpy as np
import pinocchio as pin

from g1_control.core.controller.upper_controller import UpperController
from g1_control.utility.joint_definition import LEFT_ARM_INDEX, RIGHT_ARM_INDEX

class ArmController(UpperController):
    def __init__(self,
                 urdf_path: str,
                 init: bool=True,
                 visualize: bool=False,
                 config: dict=None,
                 collision_urdf_path: str=None,
                 collision_srdf_path: str=None):
        # initialize base controller
        super().__init__(
            urdf_path=urdf_path,
            init=init,
            visualize=visualize,
            config=config,
            collision_urdf_path=collision_urdf_path,
            collision_srdf_path=collision_srdf_path,
        )

        # add frame tasks to IK solver
        self.left_task_name = 'left_ee'
        self.right_task_name = 'right_ee'
        self.ik_solver.add_frame_task(self.left_task_name, self.left_ee_name)
        self.ik_solver.add_frame_task(self.right_task_name, self.right_ee_name)

        # set initial targets for IK solver from current configuration
        self.ik_solver.update_configurations()
        self.ik_solver.set_from_configuration()

        # variable for easier access
        self._left_arm_action = np.zeros(7)
        self._right_arm_action = np.zeros(7)

        # config visualizer
        if self.visualize:
            self.robot_model.config_visualizer(
                wrench_frames=[self.left_ee_name, self.right_ee_name]
            )

    '''
    joint action for left and right arms
    '''
    @property
    def left_arm_action(self):
        return np.copy(self._left_arm_action)

    @property
    def right_arm_action(self):
        return np.copy(self._right_arm_action)

    '''
    left end effector target properties
    left_ee_target_transformation: transformation matrix of the left end effector target
    left_ee_target_position: position of the left end effector target
    left_ee_target_rotation: rotation matrix of the left end effector target
    left_ee_target_rpy: roll, pitch, yaw of the left end effector target
    left_ee_target_pose: pose of the left end effector target (x, y, z, roll, pitch, yaw)
    left_ee_error: error between the current left end effector and the target
    '''
    @property
    def left_ee_target_transformation(self):
        task = self.ik_solver.frame_tasks.get(self.left_task_name)
        return np.copy(task.transform_target_to_world.np)

    @left_ee_target_transformation.setter
    def left_ee_target_transformation(self, transformation):
        assert(transformation.shape == (4, 4)), 'Transformation should be a 4x4 matrix.'
        task = self.ik_solver.frame_tasks.get(self.left_task_name)
        task.transform_target_to_world = pin.SE3(transformation)

    @property
    def left_ee_target_position(self):
        task = self.ik_solver.frame_tasks.get(self.left_task_name)
        return np.copy(task.transform_target_to_world.translation)

    @left_ee_target_position.setter
    def left_ee_target_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        task = self.ik_solver.frame_tasks.get(self.left_task_name)
        task.transform_target_to_world.translation = np.array(position)

    @property
    def left_ee_target_rotation(self):
        task = self.ik_solver.frame_tasks.get(self.left_task_name)
        return np.copy(task.transform_target_to_world.rotation)

    @left_ee_target_rotation.setter
    def left_ee_target_rotation(self, rotation):
        assert(rotation.shape == (3, 3)), 'Rotation should be a 3x3 matrix.'
        task = self.ik_solver.frame_tasks.get(self.left_task_name)
        task.transform_target_to_world.rotation = np.array(rotation)

    @property
    def left_ee_target_rpy(self):
        return pin.rpy.matrixToRpy(self.left_ee_target_rotation)

    @left_ee_target_rpy.setter
    def left_ee_target_rpy(self, rpy):
        assert(len(rpy) == 3), 'Rpy should be a list of 3 elements (roll, pitch, yaw).'
        self.left_ee_target_rotation = pin.rpy.rpyToMatrix(np.array(rpy))

    @property
    def left_ee_target_pose(self):
        position = self.left_ee_target_position
        rpy = self.left_ee_target_rpy
        return np.concatenate([position, rpy])

    @left_ee_target_pose.setter
    def left_ee_target_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.left_ee_target_position = pose[:3]
        self.left_ee_target_rpy = pose[3:]

    @property
    def left_ee_error(self):
        return self.ik_solver.compute_error_reduced(
            self.ik_solver.frame_tasks[self.left_task_name],
            self.robot_model.state_reduced['q']
        )

    '''
    right end effector target properties
    right_ee_target_transformation: transformation matrix of the right end effector target
    right_ee_target_position: position of the right end effector target
    right_ee_target_rotation: rotation matrix of the right end effector target
    right_ee_target_rpy: roll, pitch, yaw of the right end effector target
    right_ee_target_pose: pose of the right end effector target (x, y, z, roll, pitch, yaw)
    right_ee_error: error between the current right end effector and the target
    '''
    @property
    def right_ee_target_transformation(self):
        task = self.ik_solver.frame_tasks.get(self.right_task_name)
        return np.copy(task.transform_target_to_world.np)

    @right_ee_target_transformation.setter
    def right_ee_target_transformation(self, transformation):
        assert(transformation.shape == (4, 4)), 'Transformation should be a 4x4 matrix.'
        task = self.ik_solver.frame_tasks.get(self.right_task_name)
        task.transform_target_to_world = pin.SE3(transformation)

    @property
    def right_ee_target_position(self):
        task = self.ik_solver.frame_tasks.get(self.right_task_name)
        return np.copy(task.transform_target_to_world.translation)

    @right_ee_target_position.setter
    def right_ee_target_position(self, position):
        assert(len(position) == 3), 'Position should be a list of 3 elements (x, y, z).'
        task = self.ik_solver.frame_tasks.get(self.right_task_name)
        task.transform_target_to_world.translation = np.array(position)

    @property
    def right_ee_target_rotation(self):
        task = self.ik_solver.frame_tasks.get(self.right_task_name)
        return np.copy(task.transform_target_to_world.rotation)

    @right_ee_target_rotation.setter
    def right_ee_target_rotation(self, rotation):
        assert(rotation.shape == (3, 3)), 'Rotation should be a 3x3 matrix.'
        task = self.ik_solver.frame_tasks.get(self.right_task_name)
        task.transform_target_to_world.rotation = np.array(rotation)

    @property
    def right_ee_target_rpy(self):
        return pin.rpy.matrixToRpy(self.right_ee_target_rotation)

    @right_ee_target_rpy.setter
    def right_ee_target_rpy(self, rpy):
        assert(len(rpy) == 3), 'Rpy should be a list of 3 elements (roll, pitch, yaw).'
        self.right_ee_target_rotation = pin.rpy.rpyToMatrix(np.array(rpy))

    @property
    def right_ee_target_pose(self):
        position = self.right_ee_target_position
        rpy = self.right_ee_target_rpy
        return np.concatenate([position, rpy])

    @right_ee_target_pose.setter
    def right_ee_target_pose(self, pose):
        assert(len(pose) == 6), 'Pose should be a list of 6 elements (x, y, z, roll, pitch, yaw).'
        self.right_ee_target_position = pose[:3]
        self.right_ee_target_rpy = pose[3:]

    @property
    def right_ee_error(self):
        return self.ik_solver.compute_error_reduced(
            self.ik_solver.frame_tasks[self.right_task_name],
            self.robot_model.state_reduced['q']
        )

    @property
    def joint_error(self):
        return self.ik_solver.compute_error(
            self.ik_solver.config_task,
            self.robot_model.state['q']
        )

    @property
    def joint_error_reduced(self):
        return self.ik_solver.compute_error_reduced(
            self.ik_solver.config_task,
            self.robot_model.state_reduced['q']
        )

    def update_robot_model(self):
        # call parent update method
        super().update_robot_model()

    def _apply_joint_position(self, q):
        # call parent method
        super()._apply_joint_position(q)

        # update joint action for end effectors
        self._left_arm_action = q[LEFT_ARM_INDEX]
        self._right_arm_action = q[RIGHT_ARM_INDEX]
