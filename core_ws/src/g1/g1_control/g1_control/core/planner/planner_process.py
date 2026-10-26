import queue
import traceback
import multiprocessing as mp
from dataclasses import asdict

import numpy as np

from g1_control.core.robot_model import RobotModel
from g1_control.core.planner.reduced_joint_planner import (
    PlanResult,
    PlannerConfig,
    ReducedJointPlanner,
)
from g1_control.utility.joint_definition import ENABLED_JOINTS


class PlannerClient:
    '''Client for a planner subprocess'''

    def __init__(self, urdf_path, collision_urdf_path, collision_srdf_path,
                 planner_config):
        # the planner needs a self-collision model (a collision URDF + SRDF);
        # the G1 does not have one yet, so the controller leaves planning off
        self.urdf_path = urdf_path
        self.collision_urdf_path = collision_urdf_path
        self.collision_srdf_path = collision_srdf_path
        self.planner_config = planner_config

        self._ctx = mp.get_context('spawn')
        self._request_queue = None
        self._response_queue = None
        self._process = None

    def plan_configuration(self, current_q, start, goal,
                           active_mask=None):
        '''Plan to a reduced joint target in the planner process'''
        return self._request({
            'command': 'plan_configuration',
            'current_q': np.asarray(current_q, dtype=float),
            'start': np.asarray(start, dtype=float),
            'goal': np.asarray(goal, dtype=float),
            'active_mask': _optional_array(active_mask, dtype=bool),
        })

    def shutdown(self):
        '''Stop the planner process'''
        if self._process is None:
            return
        try:
            self._request_queue.put({'command': 'shutdown'})
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        finally:
            self._process = None
            self._request_queue = None
            self._response_queue = None

    def _start(self):
        if self._process is not None and self._process.is_alive():
            return

        # lazily start the worker so spawn cost is paid only when needed
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        init_args = {
            'urdf_path': self.urdf_path,
            'collision_urdf_path': self.collision_urdf_path,
            'collision_srdf_path': self.collision_srdf_path,
            'planner_config': asdict(self.planner_config),
        }
        self._process = self._ctx.Process(
            target=_planner_worker_main,
            args=(init_args, self._request_queue, self._response_queue),
            name='planner_process',
            daemon=True,
        )
        self._process.start()

        try:
            message = self._response_queue.get(timeout=30.0)
        except queue.Empty as exc:
            raise RuntimeError(
                'Planner process did not start in time'
            ) from exc
        if message.get('type') == 'startup_error':
            raise RuntimeError(message['error'])
        if message.get('type') != 'ready':
            raise RuntimeError(
                f'Unexpected planner startup message: {message}'
            )

    def _request(self, request):
        self._start()
        self._request_queue.put(request)
        while True:
            try:
                response = self._response_queue.get(timeout=1.0)
            except queue.Empty as exc:
                if self._process is not None and self._process.is_alive():
                    continue
                raise RuntimeError(
                    'Planner process exited unexpectedly'
                ) from exc
            if isinstance(response, PlanResult):
                return response
            if response.get('type') == 'error':
                raise RuntimeError(response['error'])
            raise RuntimeError(f'Unexpected planner response: {response}')


class _PlannerWorker:
    '''Own the native planner state inside the child process'''

    def __init__(self, init_args):
        # build native state in the child so OMPL and pinocchio stay isolated
        self.robot_model = RobotModel(init_args['urdf_path'])
        self.robot_model.init_reduced_model(ENABLED_JOINTS)
        self.robot_model.init_collision_model(
            init_args['collision_urdf_path'],
            init_args['collision_srdf_path'],
        )
        planner_config = PlannerConfig(**init_args['planner_config'])
        self.planner = ReducedJointPlanner(
            self.robot_model,
            config=planner_config,
        )

    def handle(self, request):
        command = request['command']
        if command == 'plan_configuration':
            return self._plan_configuration(request)
        raise ValueError(f'Unsupported planner command: {command}')

    def _plan_configuration(self, request):
        self._set_current_q(request['current_q'])
        start = np.asarray(request['start'], dtype=float)
        goal = np.asarray(request['goal'], dtype=float)
        active_mask = _optional_array(request.get('active_mask'), dtype=bool)
        return self.planner.plan(start, goal, active_mask=active_mask)

    def _set_current_q(self, q):
        # sync the child process model with the caller's current joint state
        self.robot_model._q = np.asarray(q, dtype=float).copy()
        self.robot_model._dq = np.zeros_like(self.robot_model._q)
        self.robot_model.update_kinematics()


def _planner_worker_main(init_args, request_queue, response_queue):
    try:
        worker = _PlannerWorker(init_args)
        # notify the parent only after the child model is fully initialized
        response_queue.put({'type': 'ready'})
    except Exception as exc:
        response_queue.put({
            'type': 'startup_error',
            'error': _format_error(exc),
        })
        return

    while True:
        request = request_queue.get()
        if request.get('command') == 'shutdown':
            return
        try:
            result = worker.handle(request)
            response_queue.put(result)
        except Exception as exc:
            response_queue.put({
                'type': 'error',
                'error': _format_error(exc),
            })


def _optional_array(value, dtype=float):
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _format_error(exc):
    return f'{exc}\n{traceback.format_exc()}'
