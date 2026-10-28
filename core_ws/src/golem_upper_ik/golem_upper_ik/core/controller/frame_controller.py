'''FrameController: an UpperController whose IK tasks pull named frames to target poses'''
import numpy as np
import pinocchio as pin

from golem_upper_ik.core.controller.upper_controller import UpperController

__all__ = ['FrameController']


class FrameController(UpperController):
    def add_frame_task(self, task_name: str, frame_name: str, target: np.ndarray):
        '''A task pulling frame_name to target, a 4x4 placement in the pelvis frame'''
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError(f'the target of {frame_name} must be a 4x4 transformation, not '
                             f'shape {target.shape}')
        self.ik_solver.add_frame_task(task_name, frame_name)
        self.ik_solver.frame_tasks[task_name].transform_target_to_world = pin.SE3(target)

    def frame_task_target(self, task_name: str):
        '''The 4x4 target of a task, or None without the task'''
        task = self.ik_solver.frame_tasks.get(task_name)
        return None if task is None else task.transform_target_to_world.np

    def get_frame_task_error(self, task_name: str):
        '''The task's error at the measured state (linear, then angular)'''
        task = self.ik_solver.frame_tasks.get(task_name)
        if task is None:
            raise ValueError(f'Task {task_name} not found')
        return self.ik_solver.compute_error_reduced(task, self.robot_model.state_reduced['q'])

    def clear_frame_tasks(self):
        self.ik_solver.clear_frame_tasks()

    def get_frame_transformation(self, frame_name: str):
        '''
        The frame's 4x4 placement at the measured state, in the world of the free-flyer model
        (the pelvis at the origin, oriented by the IMU). Computed on data of its own, so a
        call from another thread leaves the IK's data alone
        '''
        return self.robot_model.get_frame_transformation(frame_name,
                                                         self.robot_model.state['q'])
