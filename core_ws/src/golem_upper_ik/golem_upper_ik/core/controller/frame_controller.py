"""Drive named frames to target poses: FrameController, the controller frame_task_server runs."""
import numpy as np
import pinocchio as pin

from golem_upper_ik.core.controller.upper_controller import UpperController

__all__ = ['FrameController']


class FrameController(UpperController):
    """Add frame-pose tasks (pelvis-frame targets) to an UpperController."""

    def unknown_frames(self, frame_names):
        """Return the names in frame_names that are not frames of the robot's models."""
        model = self.robot_model
        return [name for name in frame_names
                if not (model.model.existFrame(name) and
                        model.model_body_reduced.existFrame(name))]

    def add_frame_task(self, task_name: str, frame_name: str, target: np.ndarray):
        """Add a task pulling frame_name to target, a 4x4 placement in the pelvis frame."""
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError(f'the target of {frame_name} must be a 4x4 transformation, not '
                             f'shape {target.shape}')
        self.ik_solver.add_frame_task(task_name, frame_name, target=pin.SE3(target))

    def frame_task_target(self, task_name: str):
        """Return a task's 4x4 target, or None without the task or before its target is set."""
        task = self.ik_solver.frame_tasks.get(task_name)
        target = None if task is None else task.transform_target_to_world
        return None if target is None else target.np

    def get_frame_task_error(self, task_name: str):
        """Return a task's error at the measured state (linear m, then angular rad)."""
        task = self.ik_solver.frame_tasks.get(task_name)
        if task is None:
            raise ValueError(f'Task {task_name} not found')
        return self.ik_solver.compute_error_reduced(task, self.robot_model.state_reduced['q'])

    def frame_pose_error(self, frame_name: str, target: np.ndarray):
        """Return a frame's distance from target (m) and the angle between their rotations (rad).

        target is a 4x4 pose in the pelvis frame. The frame's pose is the IK's own view: the
        reduced model at the measured IK joints, the frozen joints at the angles the last goal
        locked them at.
        """
        pose = self.robot_model.get_frame_transformation_reduced(
            frame_name, self.robot_model.state_reduced['q'])
        target = np.asarray(target, dtype=np.float64)
        distance = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
        cosine = (np.trace(target[:3, :3].T @ pose[:3, :3]) - 1.0) / 2.0
        return distance, float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def clear_frame_tasks(self):
        """Remove every frame task."""
        self.ik_solver.clear_frame_tasks()

    def get_frame_transformation(self, frame_name: str):
        """Return a frame's 4x4 placement in the pelvis frame at the measured state.

        Every body joint is measured and the IMU orientation plays no part: this is the frame
        of the IK's targets. Computed on data of its own, so a call from another thread
        leaves the IK's data alone.
        """
        return self.robot_model.get_frame_transformation_body(frame_name,
                                                              self.robot_model.state['q'])
