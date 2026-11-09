"""Tell and wait robot time: the sim's /clock under use_sim_time, else the steady clock.

The controller's rates, freshness windows and timeouts are in s of robot time. With
use_sim_time false (the real robots) robot time is time.monotonic() and a wait is time.sleep,
so the controller keeps its steady timing there; nothing below changes it. With use_sim_time
true it is the node's ROS clock, which follows the sim's /clock:

- it advances by each forward step of /clock and stands still while /clock does; a step back
  (a restarted sim) does not move it back;
- a wait wakes on every /clock message, and at least every WAIT_SLICE_S of steady time to
  check its stop event, so a thread waiting on a stopped /clock still stops;
- close() ends every wait: a wait after it raises RuntimeError, so no loop waits on a
  /clock that has stopped once the controller has shut down.
"""
import threading
import time

from rclpy.clock import JumpThreshold
from rclpy.duration import Duration

__all__ = ['RobotClock', 'uses_sim_time']

# s of steady time: the longest a robot-time wait goes without checking its stop event
WAIT_SLICE_S = 0.01


def uses_sim_time(node) -> bool:
    """Return node's use_sim_time parameter."""
    return bool(node.get_parameter('use_sim_time').value)


class RobotClock:
    """Robot time for one node (module doc); thread-safe."""

    def __init__(self, node=None):
        """Use node's ROS clock if node runs with use_sim_time, else the steady clock."""
        self._ros_clock = None
        if node is not None and uses_sim_time(node):
            self._ros_clock = node.get_clock()
        self._lock = threading.Lock()
        self._last_ros_ns = None
        self._time_ns = 0
        self._closed = False
        self._wake = threading.Condition()
        self._jump_handle = None
        if self._ros_clock is not None:
            threshold = JumpThreshold(min_forward=Duration(nanoseconds=1),
                                      min_backward=Duration(nanoseconds=-1),
                                      on_clock_change=True)
            self._jump_handle = self._ros_clock.create_jump_callback(
                threshold, post_callback=self._on_jump)

    def now(self) -> float:
        """Return robot time, s."""
        if self._ros_clock is None:
            return time.monotonic()
        with self._lock:
            # read under the lock: a read that lands after a newer one would step the last
            # reading back and count the step between them twice
            ros_ns = self._ros_clock.now().nanoseconds
            if self._last_ros_ns is None:
                self._time_ns = ros_ns
            elif ros_ns > self._last_ros_ns:
                self._time_ns += ros_ns - self._last_ros_ns
            self._last_ros_ns = ros_ns
            return self._time_ns * 1e-9

    def sleep(self, seconds: float, stop: threading.Event = None) -> None:
        """Wait seconds of robot time; under use_sim_time, also until stop is set.

        With the steady clock this is time.sleep(seconds). RuntimeError once close() has run
        (use_sim_time only).
        """
        if self._ros_clock is None:
            time.sleep(seconds)
            return
        deadline = self.now() + seconds
        with self._wake:
            while True:
                if self._closed:
                    raise RuntimeError('the robot clock is closed')
                if (stop is not None and stop.is_set()) or self.now() >= deadline:
                    return
                self._wake.wait(WAIT_SLICE_S)

    def wait_for_clock(self, timeout: float) -> None:
        """Wait for the first /clock, up to timeout s of steady time; at once without sim time.

        RuntimeError at the timeout, or once close() has run.
        """
        if self._ros_clock is None:
            return
        deadline = time.monotonic() + timeout
        with self._wake:
            while self._ros_clock.now().nanoseconds == 0:
                if self._closed:
                    raise RuntimeError('the robot clock is closed')
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        f'No /clock within {timeout:.1f} s of steady time (use_sim_time is '
                        'true): is the sim up and publishing /clock on this domain?')
                self._wake.wait(min(remaining, WAIT_SLICE_S))

    def close(self) -> None:
        """End every wait under use_sim_time: waits raise RuntimeError from now on."""
        if self._ros_clock is None:
            return
        with self._wake:
            self._closed = True
            self._wake.notify_all()

    def _on_jump(self, _jump) -> None:
        """Wake the waits: /clock has moved (the jump callback, on the executor's thread)."""
        with self._wake:
            self._wake.notify_all()
