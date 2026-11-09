"""Check GOLEM_SIM_RATE, the sims' target real-time factor: its parse, the shared pacer and the gripper clocks.

The parse and the pacer need neither ROS nor MuJoCo:
PYTHONPATH=golem_robocasa python3 -m unittest discover -s tests/robocasa -p test_sim_rate.py -v
The gripper tests need rclpy (they stand in for magpie_msgs), so they run in the robocasa
container: ROS_DOMAIN_ID=221 tests/robocasa/run_in_container.sh -p test_sim_rate.py
"""
import functools
import importlib.util
import math
import os
import sys
import threading
import types
import unittest

import sim_common
from sim_timing import RealtimePacer

HAVE_RCLPY = importlib.util.find_spec("rclpy") is not None


class FakeClock:
    """Stand in for the wall clock and sleep: sleeping advances the clock and is recorded."""

    def __init__(self):
        """Start at 0 s with no sleeps."""
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        """Return the current time, s."""
        return self.now

    def sleep(self, delay):
        """Record a sleep and advance the clock by it, s."""
        self.sleeps.append(delay)
        self.now += delay


class SimRateParseTests(unittest.TestCase):
    """Check sim_common.sim_rate_from_env."""

    def test_unset_or_empty_is_real_time(self):
        """Unset, empty or blank gives 1: the sims pace to real time by default."""
        for env in ({}, {"GOLEM_SIM_RATE": ""}, {"GOLEM_SIM_RATE": "  "}):
            with self.subTest(env=env):
                self.assertEqual(sim_common.sim_rate_from_env(env), 1.0)

    def test_a_positive_number_is_the_factor(self):
        """A finite number above 0 is the factor, spaces ignored."""
        for raw, rate in (("2", 2.0), ("0.5", 0.5), (" 1.5 ", 1.5), ("1", 1.0)):
            with self.subTest(raw=raw):
                self.assertEqual(sim_common.sim_rate_from_env({"GOLEM_SIM_RATE": raw}), rate)

    def test_max_in_any_case_is_unpaced(self):
        """max, in any case, gives math.inf: the loop never sleeps."""
        for raw in ("max", "MAX", " Max "):
            with self.subTest(raw=raw):
                self.assertEqual(sim_common.sim_rate_from_env({"GOLEM_SIM_RATE": raw}), math.inf)

    def test_anything_else_stops_the_sim_naming_the_knob_and_value(self):
        """0, a negative number, nan, inf and text stop the sim with the knob and the value in the message."""
        for raw in ("0", "-1", "nan", "inf", "abc", "1e400"):
            with self.subTest(raw=raw):
                with self.assertRaises(SystemExit) as cm:
                    sim_common.sim_rate_from_env({"GOLEM_SIM_RATE": raw})
                self.assertIn("GOLEM_SIM_RATE", str(cm.exception.code))
                self.assertIn(repr(raw), str(cm.exception.code))

    def test_description(self):
        """The start-up line names the factor and the knob, or says the sim is unpaced."""
        self.assertEqual(sim_common.describe_sim_rate(1.0), "sim rate: 1x real time (GOLEM_SIM_RATE)")
        self.assertEqual(sim_common.describe_sim_rate(0.5), "sim rate: 0.5x real time (GOLEM_SIM_RATE)")
        self.assertEqual(sim_common.describe_sim_rate(math.inf),
                         "sim rate: unpaced, as fast as the loop runs (GOLEM_SIM_RATE=max)")


class RatePacerTests(unittest.TestCase):
    """Check RealtimePacer at a target real-time factor against a fake clock."""

    def test_default_rate_is_real_time(self):
        """Without a rate, ten 10 ms steps take 100 ms of wall time, as before the knob."""
        clock = FakeClock()
        pacer = RealtimePacer(0.01, clock=clock, sleep=clock.sleep)
        for _ in range(10):
            pacer.wait()
        self.assertAlmostEqual(clock.now, 0.1)

    def test_rate_two_waits_half_a_step(self):
        """At rate 2 a step of timestep dt waits dt/2 of wall time."""
        clock = FakeClock()
        pacer = RealtimePacer(0.002, 2.0, clock=clock, sleep=clock.sleep)
        for _ in range(100):
            pacer.wait()
        self.assertAlmostEqual(clock.now, 0.1)
        self.assertEqual(len(clock.sleeps), 100)
        for delay in clock.sleeps:
            self.assertAlmostEqual(delay, 0.001)

    def test_half_rate_waits_two_steps(self):
        """At rate 0.5 a step of timestep dt waits 2 dt of wall time."""
        clock = FakeClock()
        pacer = RealtimePacer(0.002, 0.5, clock=clock, sleep=clock.sleep)
        for _ in range(10):
            pacer.wait()
        self.assertAlmostEqual(clock.now, 0.04)

    def test_max_never_sleeps(self):
        """At rate math.inf (GOLEM_SIM_RATE=max) wait() never sleeps, however fast the steps are."""
        clock = FakeClock()
        pacer = RealtimePacer(0.001, math.inf, clock=clock, sleep=clock.sleep)
        for _ in range(1000):
            self.assertEqual(pacer.wait(), 0.0)
        self.assertEqual(clock.sleeps, [])

    def test_wait_returns_the_wall_time_slept(self):
        """wait() returns the wall seconds it slept, 0 when it was behind."""
        clock = FakeClock()
        pacer = RealtimePacer(0.01, 2.0, clock=clock, sleep=clock.sleep)
        clock.now = 0.001
        self.assertAlmostEqual(pacer.wait(), 0.004)
        clock.now += 0.02
        self.assertEqual(pacer.wait(), 0.0)

    def test_resync_is_in_wall_seconds_at_rate_two(self):
        """After a long pause at rate 2 the pacer repays at most max_lag of wall time, then sleeps again."""
        clock = FakeClock()
        pacer = RealtimePacer(0.01, 2.0, max_lag=0.1, clock=clock, sleep=clock.sleep)
        clock.now = 60.0
        self.assertEqual(pacer.wait(), 0.0)   # the resync
        unslept = 0
        while pacer.wait() == 0.0:
            unslept += 1
            self.assertLess(unslept, 100)
        # 0.1 s of wall lag at 5 ms of wall per step: 20 steps (0.2 s of sim time) run
        # without sleeping, the last one on the deadline to within rounding.
        self.assertIn(unslept, (19, 20))

    def test_invalid_rate_is_rejected(self):
        """A zero, negative or nan rate raises ValueError; math.inf is the unpaced rate."""
        for value in (0, -1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RealtimePacer(0.01, value)
        RealtimePacer(0.01, math.inf)


@functools.lru_cache(maxsize=None)
def _magpie_hand_bridge():
    """Load golem_robocasa/magpie_hand_bridge.py under its own name, with stand-ins for magpie_msgs.

    The container does not source magpie_msgs, and other tests of this suite put a stand-in
    module under the name magpie_hand_bridge, so the file is loaded by path.
    """
    if "magpie_msgs" not in sys.modules and importlib.util.find_spec("magpie_msgs") is None:
        for module, names in (("magpie_msgs", ()), ("magpie_msgs.action", ("DeliGrasp",)),
                              ("magpie_msgs.msg", ("GripperState",)),
                              ("magpie_msgs.srv", ("SetGripperForce", "SetGripperPosition"))):
            sys.modules[module] = types.ModuleType(module)
            for name in names:
                setattr(sys.modules[module], name, type(name, (), {}))
    path = os.path.join(os.path.dirname(os.path.abspath(sim_common.__file__)), "magpie_hand_bridge.py")
    spec = importlib.util.spec_from_file_location("magpie_hand_bridge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeData:
    """Stand in for MjData: only the sim time (s)."""

    def __init__(self):
        """Start at sim t 0."""
        self.time = 0.0


@unittest.skipUnless(HAVE_RCLPY, "needs rclpy (the robocasa container)")
class GripperSimClockTests(unittest.TestCase):
    """Check that the H1-2 sim's grippers keep their rates per sim second, whatever GOLEM_SIM_RATE is."""

    def bridge(self):
        """Return a MagpieHandBridge without its node: a fake sim clock, a counted state publish."""
        cls = _magpie_hand_bridge().MagpieHandBridge
        bridge = cls.__new__(cls)
        bridge.data = FakeData()
        bridge.sim_lock = threading.Lock()
        bridge._publish_period = 0.1
        bridge._next_state_t = -math.inf
        bridge.published = []
        bridge._publish_state = lambda: bridge.published.append(bridge.data.time)
        return bridge

    def test_state_publishes_every_period_of_sim_time(self):
        """tick() publishes /<side>/gripper/state at publish_rate_hz of sim time, from the first tick."""
        bridge = self.bridge()
        for step in range(1001):   # 1 s of 1 ms steps
            bridge.data.time = step * 0.001
            bridge.tick(bridge.data.time)
        self.assertEqual(len(bridge.published), 11)
        self.assertAlmostEqual(bridge.published[1] - bridge.published[0], 0.1)

    def test_settle_waits_for_sim_time(self):
        """The DeliGrasp settle returns once its seconds of sim time have passed, not of wall time."""
        bridge = self.bridge()
        polls = []

        def sleep(delay):
            """Advance the sim 62.5 ms per poll, as a sim far faster than real time would."""
            polls.append(delay)
            bridge.data.time += 0.0625

        bridge._sleep_sim(0.25, sleep=sleep)
        self.assertEqual(bridge.data.time, 0.25)
        self.assertEqual(len(polls), 4)


if __name__ == "__main__":
    unittest.main()
