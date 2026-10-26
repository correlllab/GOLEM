import numpy as np
import pinocchio as pin

from g1_control.core.controller.upper_controller import UpperController

class FrameController(UpperController):
    def __init__(self,
                 urdf_path: str,
                 init: bool=True,
                 visualize: bool=False,
                 config: dict=None,
                 collision_urdf_path: str=None,
                 collision_srdf_path: str=None):
        super().__init__(
            urdf_path=urdf_path,
            init=init,
            visualize=visualize,
            config=config,
            collision_urdf_path=collision_urdf_path,
            collision_srdf_path=collision_srdf_path,
        )

    def add_frame_task(self, task_name: str, frame_name: str, target: np.ndarray = None):
        '''Add a frame task with optional pose'''
        self.ik_solver.add_frame_task(task_name, frame_name)
        if target is not None:
            if target.shape == (6,):
                self.set_frame_task_pose(task_name, target)
            elif target.shape == (4, 4):
                self.set_frame_task_transformation(task_name, target)
            else:
                raise ValueError('Target must be either a 6D pose or a 4x4 transformation matrix')

    def set_frame_task_pose(self, task_name: str, pose: np.ndarray):
        '''Set pose for a frame task'''
        assert len(pose) == 6, 'Pose must be [x, y, z, roll, pitch, yaw]'
        task = self.ik_solver.frame_tasks.get(task_name)
        if task is None:
            raise ValueError(f'Task {task_name} not found')
        position = np.array(pose[:3])
        rpy = np.array(pose[3:])
        rotation = pin.rpy.rpyToMatrix(rpy)
        transform = pin.SE3(rotation, position)
        task.transform_target_to_world = transform

    def set_frame_task_transformation(self, task_name: str, transformation: np.ndarray):
        '''Set transformation for a frame task'''
        assert transformation.shape == (4, 4), 'Transformation must be a 4x4 matrix'
        task = self.ik_solver.frame_tasks.get(task_name)
        if task is None:
            raise ValueError(f'Task {task_name} not found')
        transform = pin.SE3(transformation)
        task.transform_target_to_world = transform

    def get_frame_task_error(self, task_name: str):
        '''Get error for a frame task'''
        task = self.ik_solver.frame_tasks.get(task_name)
        if task is None:
            raise ValueError(f'Task {task_name} not found')
        return self.ik_solver.compute_error_reduced(
            task,
            self.robot_model.state_reduced['q']
        )

    def get_frame_task_errors(self):
        '''Get errors for all frame tasks'''
        errors = {}
        for task_name, task in self.ik_solver.frame_tasks.items():
            errors[task_name] = self.ik_solver.compute_error_reduced(
                task,
                self.robot_model.state_reduced['q']
            )
        return errors

    def remove_frame_task(self, task_name: str):
        '''Remove a frame task'''
        self.ik_solver.remove_frame_task(task_name)

    def clear_frame_tasks(self):
        '''Clear all frame tasks'''
        self.ik_solver.clear_frame_tasks()

    def get_frame_pose(self, frame_name: str):
        '''Get current pose of a frame'''
        position = self.robot_model.get_frame_position(frame_name)
        rotation = self.robot_model.get_frame_rotation(frame_name)
        rpy = pin.rpy.matrixToRpy(rotation)
        return np.concatenate([position, rpy])

    def get_frame_transformation(self, frame_name: str):
        '''Get current transformation matrix of a frame'''
        return self.robot_model.get_frame_transformation(frame_name)
