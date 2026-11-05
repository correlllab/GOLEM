"""Run a ReducedJointPlanner in a subprocess of its own: PlannerClient.

The subprocess keeps OMPL's planning time and native state out of the controller process.
It builds its own RobotModel from the robot's profile, once, on the first request. A robot
whose profile has no self-collision model cannot plan: its client reports unavailable and
refuses every request without starting the subprocess.

Every request carries a sequence number that its reply echoes, and the client takes only
the reply to the request it waits for. A subprocess that may still owe a reply nobody waits
for (its start timed out, or a request was abandoned) is stopped, and the next request
starts a fresh one with fresh queues. Requests from several threads take turns: one waits
until the one before it has its reply.
"""
import itertools
import queue
import threading
import time
import traceback
import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import asdict, replace

import numpy as np

from golem_upper_ik.core.robot_model import FixedState, RobotModel
from golem_upper_ik.core.planner.reduced_joint_planner import (
    PlannerConfig,
    ReducedJointPlanner,
)
from golem_upper_ik.profile import RobotProfile, load_profile, robot_config

# why a robot without a self-collision model cannot plan: {robot} is its display name and
# {module} its name
PLANNER_UNAVAILABLE = (
    'Planning is unavailable: the {robot} has no self-collision model (its profile, '
    'golem_upper_ik/config/{module}/robot.yaml, names no sphere URDF and SRDF); send the '
    'goal with plan=false'
)
STARTUP_TIMEOUT = 30.0  # s for the subprocess to build its model and report ready
POLL_PERIOD = 0.1  # s between checks on the subprocess while waiting for it
SHUTDOWN_GRACE = 2.0  # s a graceful shutdown waits for the subprocess before terminating it


class PlannerUnavailableError(RuntimeError):
    """Signal that the robot cannot plan: its profile has no self-collision model."""


class _Worker:
    """Hold one planner subprocess and its request and response queues."""

    def __init__(self, ctx, target, init_args):
        """Create (not start) the subprocess running target(init_args, requests, responses)."""
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()
        self.process = ctx.Process(
            target=target,
            args=(init_args, self.request_queue, self.response_queue),
            name='planner_process',
            daemon=True,
        )
        self.closed = False
        self._lock = threading.Lock()

    def close(self, graceful):
        """Stop the subprocess, asked to shut down first when graceful; once only.

        It is terminated when not graceful or when it does not shut down within
        SHUTDOWN_GRACE.
        """
        with self._lock:
            if self.closed:
                return
            self.closed = True
        if self.process.pid is None:  # never started
            return
        if graceful and self.process.is_alive():
            self.request_queue.put({'command': 'shutdown'})
            self.process.join(timeout=SHUTDOWN_GRACE)
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1.0)
        # a request the subprocess never read must not hold up interpreter exit
        self.request_queue.cancel_join_thread()


class PlannerClient:
    """Send plan requests to a planner subprocess, started on the first request."""

    def __init__(self, profile: RobotProfile, planner_config: PlannerConfig = None,
                 frame_names=None, startup_timeout: float = STARTUP_TIMEOUT):
        """Set up the client; no subprocess starts before the first request.

        profile: the robot; the subprocess builds its model from load_profile(profile.robot).
        planner_config: the planner's settings, default PlannerConfig(); its frame_names are
            set from frame_names, and a config naming other frames is refused (ValueError).
        frame_names: the (left, right) planner frames, default the profile's planner frames
            (the controller passes the controller config's planner.frame_names).
        startup_timeout: s for the subprocess to build its model and report ready.
        """
        self.robot = profile.robot
        frames = tuple(profile.planner_frames if frame_names is None else frame_names)
        config = planner_config if planner_config is not None else PlannerConfig()
        if config.frame_names is not None and tuple(config.frame_names) != frames:
            raise ValueError(f'planner_config.frame_names {tuple(config.frame_names)} differ '
                             f'from the planner frames {frames}: pass them as frame_names')
        self.planner_config = replace(config, frame_names=frames)
        self.startup_timeout = startup_timeout
        self.unavailable_reason = None
        if profile.collision is None:
            self.unavailable_reason = PLANNER_UNAVAILABLE.format(
                robot=robot_config(profile.robot)['display_name'], module=profile.robot)

        self._ctx = mp.get_context('spawn')
        self._worker_main = _planner_worker_main
        self._worker = None
        self._request_ids = itertools.count(1)
        # one request at a time: two waiting on one worker's replies would each have to take
        # the other's reply off the queue to find its own
        self._request_lock = threading.Lock()

    @property
    def available(self) -> bool:
        """Return whether the robot can plan; unavailable_reason says why not."""
        return self.unavailable_reason is None

    def plan_configuration(self, current_q, start, goal,
                           active_mask=None, moving_speed=None,
                           max_interpolation_steps=None):
        """Return the subprocess's PlanResult from start to goal (reduced q).

        current_q is the measured body q, which places the frozen joints for the workspace
        bounds. The worker's PlannerConfig is serialized once at spawn, so per-goal speed
        changes travel with the request; None plans with the config the worker started with.
        PlannerUnavailableError when the robot cannot plan; RuntimeError ('planner shut
        down') when shutdown() runs while this waits, and for the worker's errors.
        """
        if not self.available:
            raise PlannerUnavailableError(self.unavailable_reason)
        return self._request({
            'command': 'plan_configuration',
            'current_q': np.asarray(current_q, dtype=float),
            'start': np.asarray(start, dtype=float),
            'goal': np.asarray(goal, dtype=float),
            'active_mask': _optional_array(active_mask, dtype=bool),
            'moving_speed': moving_speed,
            'max_interpolation_steps': max_interpolation_steps,
        })

    def shutdown(self):
        """Stop the planner process; a request waiting on it raises 'planner shut down'."""
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.close(graceful=True)

    def _stop(self, worker):
        """Terminate worker, which may owe a reply nobody waits for."""
        if self._worker is worker:
            self._worker = None
        worker.close(graceful=False)

    def _start(self):
        """Return the running worker, starting one if none runs (spawn cost only when needed)."""
        worker = self._worker
        if worker is not None and not worker.closed and worker.process.is_alive():
            return worker
        if worker is not None:
            self._stop(worker)

        init_args = {
            # the profile does not pickle (read-only mappings): the worker loads it by name
            'robot': self.robot,
            'planner_config': asdict(self.planner_config),
        }
        worker = _Worker(self._ctx, self._worker_main, init_args)
        self._worker = worker
        try:
            worker.process.start()
            self._await_ready(worker)
        except BaseException:
            self._stop(worker)
            raise
        return worker

    def _await_ready(self, worker):
        """Wait for worker's ready message; RuntimeError on its start-up error or a timeout."""
        deadline = time.monotonic() + self.startup_timeout
        while True:
            message = self._next_message(worker)
            if message is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Planner process did not start in time')
                continue
            if message.get('type') == 'startup_error':
                raise RuntimeError(message['error'])
            if message.get('type') != 'ready':
                raise RuntimeError(f'Unexpected planner startup message: {message}')
            return

    def _request(self, request):
        """Send request and return its result; RuntimeError for the worker's error."""
        with self._request_lock:
            worker = self._start()
            request_id = next(self._request_ids)
            try:
                worker.request_queue.put(dict(request, id=request_id))
                response = self._await_response(worker, request_id)
            except BaseException:
                # the reply to this request, should it come, would wait for the next one
                self._stop(worker)
                raise
        if response.get('type') == 'result':
            return response['result']
        if response.get('type') == 'error':
            raise RuntimeError(response['error'])
        raise RuntimeError(f'Unexpected planner response: {response}')

    def _await_response(self, worker, request_id):
        """Return the reply to request_id, dropping replies to other requests."""
        while True:
            message = self._next_message(worker)
            # None: still planning; another id: a reply to a request nobody waits for
            if message is not None and message.get('id') == request_id:
                return message

    def _next_message(self, worker):
        """Return the worker's next message, or None after a poll period without one.

        RuntimeError when the client was shut down, or the worker exited and left no message.
        """
        if worker.closed:
            raise RuntimeError('planner shut down')
        try:
            return worker.response_queue.get(timeout=POLL_PERIOD)
        except queue.Empty:
            pass
        if worker.closed:
            raise RuntimeError('planner shut down')
        if not worker.process.is_alive():
            # a message written just before it exited is still in the pipe
            try:
                return worker.response_queue.get(timeout=POLL_PERIOD)
            except queue.Empty:
                raise RuntimeError(f'Planner process exited unexpectedly (exit code '
                                   f'{worker.process.exitcode})') from None
        return None


class _PlannerWorker:
    """Own the robot model and the planner inside the subprocess."""

    def __init__(self, init_args):
        """Build the robot model and the planner from init_args' robot and planner config."""
        self.robot_model = RobotModel(load_profile(init_args['robot']))
        planner_config = PlannerConfig(**init_args['planner_config'])
        self.planner = ReducedJointPlanner(
            self.robot_model,
            config=planner_config,
        )

    def handle(self, request):
        """Return the result of one request; ValueError for an unknown command."""
        command = request['command']
        if command == 'plan_configuration':
            return self._plan_configuration(request)
        raise ValueError(f'Unsupported planner command: {command}')

    def _plan_configuration(self, request):
        """Return the PlanResult of a plan_configuration request."""
        self._set_current_q(request['current_q'])
        start = np.asarray(request['start'], dtype=float)
        goal = np.asarray(request['goal'], dtype=float)
        active_mask = _optional_array(request.get('active_mask'), dtype=bool)
        with self._config_override(
            request.get('moving_speed'),
            request.get('max_interpolation_steps'),
        ):
            return self.planner.plan(start, goal, active_mask=active_mask)

    @contextmanager
    def _config_override(self, moving_speed, max_interpolation_steps):
        """Apply a request's speed settings to the planner config for the block, then restore."""
        if moving_speed is None and max_interpolation_steps is None:
            yield
            return

        config = self.planner.config
        previous = (config.moving_speed, config.max_interpolation_steps)
        # PlannerConfig is validated at construction only, so an override is checked here, or
        # a bad value divides by zero when interpolating
        if moving_speed is not None:
            moving_speed = float(moving_speed)
            if moving_speed <= 0.0:
                raise ValueError('Planner moving_speed must be positive')
            config.moving_speed = moving_speed
        if max_interpolation_steps is not None:
            max_interpolation_steps = int(max_interpolation_steps)
            if max_interpolation_steps < config.min_interpolation_steps:
                raise ValueError(
                    'Planner max_interpolation_steps must be at least '
                    'min_interpolation_steps'
                )
            config.max_interpolation_steps = max_interpolation_steps
        try:
            yield
        finally:
            config.moving_speed, config.max_interpolation_steps = previous

    def _set_current_q(self, q):
        """Hold the caller's measured body q, at rest, as the model's state.

        It places the frozen joints for the planner frames' workspace bounds.
        """
        self.robot_model.state_subscriber = FixedState(q)
        self.robot_model.update_kinematics()


def _planner_worker_main(init_args, request_queue, response_queue):
    """Run the subprocess: send ready (or startup_error), then one reply per request.

    Each reply carries its request's id; a shutdown request ends the loop.
    """
    try:
        worker = _PlannerWorker(init_args)
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
            response_queue.put({'id': request.get('id'), 'type': 'result',
                                'result': worker.handle(request)})
        except Exception as exc:
            response_queue.put({
                'id': request.get('id'),
                'type': 'error',
                'error': _format_error(exc),
            })


def _optional_array(value, dtype=float):
    """Return value as an array of dtype, or None."""
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _format_error(exc):
    """Return exc's message with the current traceback, for the parent to raise."""
    return f'{exc}\n{traceback.format_exc()}'
