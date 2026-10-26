import time

import numpy as np
from pinocchio.visualize import MeshcatVisualizer

import meshcat_shapes
import meshcat.geometry as geo
import meshcat.transformations as tf


class RobotVisualizer:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.viz = None
        # toggle flags for optional overlays
        self.viz_config = {
            'show_com': False,
            'show_zmp': False,
            'show_sensors': False,
            'wrench_frames': [],
        }
        # cached path metadata for pelvis-relative redraws
        self._path_specs = []

    def init_visualizer(self):
        '''Initialize meshcat visualizer'''
        try:
            # build the meshcat backend
            self.viz = MeshcatVisualizer(
                self.robot_model.model,
                visual_model=self.robot_model.visual_model,
                copy_models=False,
                data=self.robot_model.data,
            )
            self.viz.initViewer()
            self.viz.loadViewerModel('unitree_g1')
            # reset runtime visualization state on init
            self.viz_config = {
                'show_com': False,
                'show_zmp': False,
                'show_sensors': False,
                'wrench_frames': [],
            }
            self._path_specs = []
        except ImportError as err:
            print('ImportError: MeshcatVisualizer requires the meshcat package.')
            print(err)
            exit(0)

    def config_visualizer(self,
                          show_com=None,
                          show_zmp=None,
                          show_sensors=None,
                          wrench_frames=None):
        '''Configure dynamic visualizations'''
        assert self.viz is not None, 'Visualizer must be initialized first.'

        # only overwrite settings that were explicitly provided
        config_updates = {
            'show_com': show_com,
            'show_zmp': show_zmp,
            'show_sensors': show_sensors,
            'wrench_frames': wrench_frames,
        }
        for key, value in config_updates.items():
            if value is not None:
                self.viz_config[key] = value

        # create sensor frame markers once when enabled
        if show_sensors:
            meshcat_shapes.frame(self.viz.viewer['livox_frame'], opacity=1.0)
            meshcat_shapes.frame(self.viz.viewer['camera_frame'], opacity=1.0)

    def _has_frame(self, model, frame_name):
        # guard meshcat and pinocchio calls against invalid frame names
        frame_id = model.getFrameId(frame_name)
        return frame_id < len(model.frames)

    def _visualize_sensors(self):
        '''Update sensor frame visualizations (lidar and head camera)'''
        viewer = self.viz.viewer
        # keep sensor markers attached to their current world transforms
        if self._has_frame(self.robot_model.model, 'mid360_link'):
            viewer['livox_frame'].set_transform(
                self.robot_model.get_frame_transformation('mid360_link')
            )
        if self._has_frame(self.robot_model.model, 'd435_link'):
            viewer['camera_frame'].set_transform(
                self.robot_model.get_frame_transformation('d435_link')
            )

    def _visualize_com(self):
        '''Update center of mass visualization'''
        # draw com as a fixed-size sphere
        com_pos = self.robot_model.dynamics.get_com()
        transform = np.eye(4)
        transform[:3, 3] = com_pos
        viewer = self.viz.viewer
        viewer['com'].set_object(geo.Sphere(0.02))
        viewer['com'].set_transform(transform)
        viewer['com'].set_property('color', (1.0, 0.5, 0.0, 0.8))

    def _visualize_zmp(self):
        '''Update zero moment point visualization'''
        # draw zmp as a fixed-size sphere
        zmp_pos = self.robot_model.dynamics.get_zmp()
        transform = np.eye(4)
        transform[:3, 3] = zmp_pos
        viewer = self.viz.viewer
        viewer['zmp'].set_object(geo.Sphere(0.02))
        viewer['zmp'].set_transform(transform)
        viewer['zmp'].set_property('color', (0.0, 1.0, 0.0, 0.8))

    def _visualize_wrench(self, link_name):
        if not self._has_frame(self.robot_model.model, link_name):
            return

        # convert wrench to arrows
        wrench = self.robot_model.dynamics.get_frame_wrench(link_name)

        force = wrench[0:3]
        force_magnitude = np.linalg.norm(force) + 1e-6
        force_direction = force / force_magnitude
        force_magnitude *= 0.1
        force_transform = self._get_arrow_transformation(
            force_direction, force_magnitude
        )
        viewer = self.viz.viewer
        viewer[f'{link_name}/force_arrow'].set_object(
            geo.Cylinder(height=force_magnitude, radius=0.01)
        )
        viewer[f'{link_name}/force_arrow'].set_transform(force_transform)
        viewer[f'{link_name}/force_arrow'].set_property('color', (1.0, 0.0, 0.0, 0.8))

        torque = wrench[3:6]
        torque_magnitude = np.linalg.norm(torque) + 1e-6
        torque_direction = torque / torque_magnitude
        torque_magnitude *= 0.1
        torque_transform = self._get_arrow_transformation(
            torque_direction, torque_magnitude
        )
        self.viz.viewer[f'{link_name}/torque_arrow'].set_object(
            geo.Cylinder(height=torque_magnitude, radius=0.01)
        )
        self.viz.viewer[f'{link_name}/torque_arrow'].set_transform(torque_transform)
        self.viz.viewer[f'{link_name}/torque_arrow'].set_property('color', (0.0, 0.0, 1.0, 0.8))

    def _get_arrow_transformation(self, direction, magnitude):
        # align y with the direction
        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(y_axis, direction)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_matrix = np.eye(3) if np.dot(y_axis, direction) > 0 else -np.eye(3)
        else:
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(y_axis, direction), -1.0, 1.0))
            rotation_matrix = tf.rotation_matrix(angle, rotation_axis)[:3, :3]

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = rotation_matrix @ np.array([0, 1, 0]) * magnitude / 2

        return transform

    def update_visualizer(self):
        '''Update robot visualization and all enabled dynamic visualizations'''
        if self.viz is not None:
            # update the robot mesh
            self.viz.display()

            # refresh overlays
            if self.viz_config.get('show_sensors', False):
                self._visualize_sensors()
            if self.viz_config.get('show_com', False):
                self._visualize_com()
            if self.viz_config.get('show_zmp', False):
                self._visualize_zmp()
            for frame_name in self.viz_config.get('wrench_frames', []):
                self._visualize_wrench(frame_name)

            self.update_path()

    def _as_reduced_path(self, path):
        '''Convert path input to a reduced waypoint array'''
        path = np.asarray(path, dtype=float)
        if path.ndim == 1:
            path = path.reshape(1, -1)
        if path.ndim != 2 or path.shape[1] != self.robot_model.model_body_reduced.nq:
            raise ValueError(
                f'Path must have shape (N, {self.robot_model.model_body_reduced.nq})'
            )
        return path

    def _reduced_path_frame_transforms(self, path, frame_name):
        '''Get full-model frame transforms for a reduced path'''
        # rebuild full poses for the frame
        q = np.copy(self.robot_model.state['q'])
        transforms = []
        for q_reduced in path:
            q[self.robot_model.reduced_mask] = q_reduced
            transforms.append(self.robot_model._get_frame_transformation(frame_name, q))
        return transforms

    def visualize_reduced_path(self, path,
                               frame_names=('left_hand_palm_link', 'right_hand_palm_link')):
        '''Draw reduced path traces and frame samples in Meshcat'''
        assert self.viz is not None, 'Visualizer must be initialized first.'
        assert self.robot_model.init_reduced, 'Reduced model is not initialized.'
        path = self._as_reduced_path(path)

        # cache path data in pelvis space
        frame_colors = {
            'left_hand_palm_link': 0xff00ff,
            'right_hand_palm_link': 0x00ffff,
        }
        frame_names = tuple(frame_names)
        self._path_specs = []
        pelvis_to_world = self.robot_model.get_frame_transformation('pelvis')
        world_to_pelvis = np.linalg.inv(pelvis_to_world)
        for frame_name in frame_names:
            # skip missing frames
            prefix = f'planned_path/{frame_name}'
            if not self._has_frame(self.robot_model.model, frame_name):
                continue

            # project the whole path into pelvis space once
            self.viz.viewer[prefix].delete()
            transforms = self._reduced_path_frame_transforms(path, frame_name)
            points_world = np.asarray(
                [transform.translation for transform in transforms],
                dtype=float,
            ).T
            points_local_h = world_to_pelvis @ np.vstack([
                points_world,
                np.ones((1, points_world.shape[1])),
            ])
            points_local = points_local_h[:3, :]

            # visualize intermediate frames
            sample_local_transforms = []
            sample_names = []
            sample_fractions = (0.25, 0.5, 0.75)
            indices = np.rint((len(path) - 1) * np.asarray(sample_fractions)).astype(int)
            indices = np.clip(indices, 0, len(path) - 1)
            for sample_idx, path_idx in enumerate(indices):
                q = np.copy(self.robot_model.state['q'])
                q[self.robot_model.reduced_mask] = path[path_idx]
                transform_world = self.robot_model._get_frame_transformation(frame_name, q)
                sample_local_transforms.append(world_to_pelvis @ transform_world)
                sample_names.append(f'{prefix}/frame_{sample_idx}')

            self._path_specs.append({
                'prefix': prefix,
                'frame_name': frame_name,
                'points_local': points_local,
                'sample_local_transforms': sample_local_transforms,
                'sample_names': sample_names,
                'color': frame_colors.get(frame_name, 0xffffff),
            })

            # draw the cached path line
            self.viz.viewer[f'{prefix}/line'].set_object(
                geo.Line(
                    geo.PointsGeometry(points_local),
                    geo.LineBasicMaterial(
                        color=frame_colors.get(frame_name, 0xffffff),
                        linewidth=4.0,
                    ),
                )
            )

            # add the sampled frames once
            for path_name in sample_names:
                meshcat_shapes.frame(self.viz.viewer[path_name], opacity=0.8)

        # draw once now
        self.update_path()

    def clear_path_visualization(self):
        '''Remove planned path lines and frame samples from Meshcat'''
        # clear cached path data
        self._path_specs = []
        if self.viz is not None:
            self.viz.viewer['planned_path'].delete()

    def update_path(self):
        '''Redraw stored path lines and frames using current base orientation'''
        specs = getattr(self, '_path_specs', None)
        if not specs:
            return

        # apply the current pelvis pose
        pelvis_to_world = self.robot_model.get_frame_transformation('pelvis')
        for spec in specs:
            prefix = spec['prefix']
            self.viz.viewer[f'{prefix}/line'].set_transform(pelvis_to_world)

            # move sampled frames with the pelvis
            for path_name, local_transform in zip(
                spec['sample_names'], spec['sample_local_transforms']
            ):
                self.viz.viewer[path_name].set_transform(
                    (pelvis_to_world @ local_transform).astype(float)
                )

    def play_reduced_path(self, path, delay=0.05):
        '''Animate a reduced joint-space path in Meshcat'''
        assert self.viz is not None, 'Visualizer must be initialized first.'
        assert self.robot_model.init_reduced, 'Reduced model is not initialized.'
        path = self._as_reduced_path(path)

        # step through the path by displaying the full robot configuration
        q = np.copy(self.robot_model.state['q'])
        for q_reduced in path:
            q[self.robot_model.reduced_mask] = q_reduced
            self.viz.display(self.robot_model.full_q(q))
            time.sleep(delay)
