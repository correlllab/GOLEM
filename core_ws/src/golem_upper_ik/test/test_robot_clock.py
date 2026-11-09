"""Check RobotClock: the steady clock without use_sim_time, the sim's /clock with it.

Without use_sim_time robot time is time.monotonic() and a wait is time.sleep, as the real
robots run. With it, robot time follows /clock (SimClock, here at other rates than the wall
clock's), stands still while /clock does, never steps back, and close() ends a wait on a
stopped /clock.
"""
import threading
import time
from types import SimpleNamespace

import pytest

from ros_harness import SimClock, wait_for


def clock_of(ros, sim_time):
    """Return a RobotClock on a new node, with use_sim_time as sim_time."""
    from golem_upper_ik.core.robot_clock import RobotClock
    return RobotClock(ros.node('controller', sim_time=sim_time))


def test_without_sim_time_it_is_the_steady_clock(ros, monkeypatch):
    """Without use_sim_time, now() is time.monotonic() and sleep() is time.sleep()."""
    from golem_upper_ik.core import robot_clock
    clock = clock_of(ros, sim_time=False)
    assert clock._ros_clock is None
    calls = []
    monkeypatch.setattr(robot_clock.time, 'monotonic', lambda: 42.0)
    monkeypatch.setattr(robot_clock.time, 'sleep', calls.append)
    assert clock.now() == 42.0
    clock.sleep(0.25)
    assert calls == [0.25]
    clock.wait_for_clock(0.0)
    clock.close()
    clock.sleep(0.5)
    assert calls == [0.25, 0.5]


def test_no_node_is_the_steady_clock():
    """A RobotClock without a node is the steady clock."""
    from golem_upper_ik.core.robot_clock import RobotClock
    clock = RobotClock()
    assert clock._ros_clock is None
    assert abs(clock.now() - time.monotonic()) < 0.01


@pytest.mark.parametrize('rate', [0.5, 2.0])
def test_with_sim_time_it_follows_clock(ros, rate):
    """With use_sim_time, now() follows /clock and a sleep lasts seconds of sim time."""
    sim = SimClock(ros, rate=rate)
    clock = clock_of(ros, sim_time=True)
    assert clock._ros_clock is not None
    sim.start()
    clock.wait_for_clock(2.0)
    assert abs(clock.now() - sim.now) < 0.05
    start_wall, start = time.monotonic(), clock.now()
    clock.sleep(0.4)
    wall, robot = time.monotonic() - start_wall, clock.now() - start
    assert 0.4 <= robot < 0.45, robot
    # 0.4 s of sim time is 0.8 s of wall time at half speed, 0.2 s at twice
    assert 0.4 / rate - 0.05 < wall < 0.4 / rate + 0.1, wall


def test_a_stopped_clock_holds_robot_time_and_a_sleep(ros):
    """While /clock stops, robot time stands still and a sleep does not end."""
    sim = SimClock(ros, rate=1.0)
    clock = clock_of(ros, sim_time=True)
    sim.start()
    clock.wait_for_clock(2.0)
    sim.pause()
    time.sleep(0.1)
    frozen = clock.now()
    done = threading.Event()
    threading.Thread(target=lambda: (clock.sleep(0.05), done.set()), daemon=True).start()
    time.sleep(0.3)
    assert clock.now() == frozen
    assert not done.is_set(), 'a sleep ended with /clock stopped'
    sim.resume()
    assert done.wait(1.0)


def test_close_ends_a_sleep_on_a_stopped_clock(ros):
    """close() makes a sleep on a stopped /clock raise at once."""
    sim = SimClock(ros, rate=1.0)
    clock = clock_of(ros, sim_time=True)
    sim.start()
    clock.wait_for_clock(2.0)
    sim.pause()
    raised = []

    def sleep():
        """Sleep 1 s of robot time, noting what it raises."""
        try:
            clock.sleep(1.0)
        except RuntimeError as error:
            raised.append(error)
    thread = threading.Thread(target=sleep, daemon=True)
    thread.start()
    time.sleep(0.1)
    clock.close()
    thread.join(0.5)
    assert not thread.is_alive()
    assert raised and 'closed' in str(raised[0])
    with pytest.raises(RuntimeError, match='closed'):
        clock.sleep(0.0)


def test_a_stop_event_ends_a_sleep_on_a_stopped_clock(ros):
    """A sleep given a stop event ends once it is set, /clock stopped or not."""
    sim = SimClock(ros, rate=1.0)
    clock = clock_of(ros, sim_time=True)
    sim.start()
    clock.wait_for_clock(2.0)
    sim.pause()
    stop = threading.Event()
    thread = threading.Thread(target=clock.sleep, args=(10.0, stop), daemon=True)
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()
    stop.set()
    thread.join(0.5)
    assert not thread.is_alive()


def test_robot_time_never_steps_back(ros):
    """A /clock that steps back (a restarted sim) holds robot time; it advances from there."""
    sim = SimClock(ros, rate=1.0)
    clock = clock_of(ros, sim_time=True)
    sim.start()
    clock.wait_for_clock(2.0)
    before = clock.now()
    sim.step_back(50.0)
    time.sleep(0.2)
    after = clock.now()
    assert after >= before
    assert after - before < 0.5
    assert wait_for(lambda: clock.now() > after + 0.1, 2.0)


def test_wait_for_clock_is_bounded(ros):
    """Without /clock, wait_for_clock raises after its timeout of steady time."""
    clock = clock_of(ros, sim_time=True)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match=r'No /clock within 0.3 s'):
        clock.wait_for_clock(0.3)
    assert 0.3 <= time.monotonic() - start < 0.6


class _InterleavedRosClock:
    """Stand in for a ROS clock whose reader 'slow' gets the older time and returns last."""

    def __init__(self):
        """Read 1 s until the threads start."""
        self.fast_done = threading.Event()
        self.started = False

    def now(self):
        """Return 2 s to the slow reader, after the fast one has read 3 s (or 0.3 s passed)."""
        name = threading.current_thread().name
        if not self.started:
            seconds = 1.0
        elif name == 'slow':
            seconds = 2.0
            self.fast_done.wait(0.3)
        else:
            seconds = 3.0
            if name == 'fast':
                self.fast_done.set()
        return SimpleNamespace(nanoseconds=int(seconds * 1e9))


def test_concurrent_reads_never_count_a_step_twice(ros):
    """Two threads reading at once never make robot time run ahead of /clock."""
    from golem_upper_ik.core.robot_clock import RobotClock
    clock = RobotClock(ros.node('controller', sim_time=True))
    fake = _InterleavedRosClock()
    clock._ros_clock = fake
    assert clock.now() == 1.0
    fake.started = True
    slow = threading.Thread(target=clock.now, name='slow')
    slow.start()
    time.sleep(0.05)
    fast = threading.Thread(target=clock.now, name='fast')
    fast.start()
    slow.join(1.0)
    fast.join(1.0)
    assert clock.now() == 3.0
