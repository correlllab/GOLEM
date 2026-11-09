"""Check the H1-2 sim loop's real-time pacer and its --timing report (sim_timing.py), without ROS or MuJoCo."""
import io
import unittest
from contextlib import redirect_stdout

from sim_timing import RealtimePacer, LoopTimings


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


class PacingTests(unittest.TestCase):
    """Check RealtimePacer and LoopTimings against a fake clock."""

    def test_fast_steps_track_the_original_deadline(self):
        """Steps faster than real time are slowed to one 10 ms step per 10 ms."""
        clock = FakeClock()
        pacer = RealtimePacer(0.01, clock=clock, sleep=clock.sleep)
        for _ in range(10):
            clock.now += 0.002
            pacer.wait()
        self.assertAlmostEqual(clock.now, 0.1)

    def test_sensor_overrun_is_recovered_without_skipping_steps(self):
        """After an overrun the pacer runs the late steps without sleeping, then sleeps again."""
        clock = FakeClock()
        pacer = RealtimePacer(0.01, clock=clock, sleep=clock.sleep)
        clock.now = 0.025
        pacer.wait()
        pacer.wait()
        self.assertEqual(clock.sleeps, [])
        pacer.wait()
        self.assertAlmostEqual(clock.now, 0.03)
        self.assertAlmostEqual(clock.sleeps[0], 0.005)

    def test_long_pause_does_not_accumulate_unbounded_catchup(self):
        """After a long pause the pacer repays at most max_lag and then sleeps again."""
        clock = FakeClock()
        pacer = RealtimePacer(0.01, max_lag=0.1, clock=clock, sleep=clock.sleep)
        clock.now = 60.0
        pacer.wait()
        for _ in range(12):
            pacer.wait()
        self.assertTrue(clock.sleeps)
        self.assertGreater(clock.now, 60.0)
        self.assertTrue(all(delay > 0 for delay in clock.sleeps))

    def test_invalid_timestep_is_rejected(self):
        """A zero, negative or non-finite timestep raises ValueError."""
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RealtimePacer(value)

    def test_timing_is_silent_by_default(self):
        """Timings print nothing unless enabled."""
        timing = LoopTimings()
        with redirect_stdout(io.StringIO()) as output:
            timing.stop('physics', timing.start())
            timing.maybe_report(1.0)
        self.assertEqual(output.getvalue(), '')

    def test_timing_baseline_excludes_startup_and_existing_sim_time(self):
        """The real-time factor counts from the reset, not from start-up or sim time before it."""
        clock = FakeClock()
        timing = LoopTimings(enabled=True, interval=1, clock=clock)
        clock.now = 10.0
        timing.reset(5.0)
        clock.now = 11.0
        with redirect_stdout(io.StringIO()) as output:
            timing.maybe_report(5.5)
        self.assertIn('RTF=0.500', output.getvalue())

    def test_timing_reports_sim_wall_ratio_and_stage_cost(self):
        """The report gives the real-time factor and each stage's cost in ms."""
        clock = FakeClock()
        timing = LoopTimings(enabled=True, interval=1, clock=clock)
        start = timing.start()
        clock.now = 0.02
        timing.stop('physics', start)
        clock.now = 1.0
        with redirect_stdout(io.StringIO()) as output:
            timing.maybe_report(0.5)
        self.assertIn('RTF=0.500', output.getvalue())
        self.assertIn('physics', output.getvalue())
        self.assertIn('20.00', output.getvalue())

    def test_timing_report_appends_the_sensor_stats(self):
        """stats() is called only when a report is due, and its text ends the line."""
        clock = FakeClock()
        timing = LoopTimings(enabled=True, interval=1, clock=clock)
        calls = []

        def stats():
            """Return the sensors' counts and record the call."""
            calls.append(clock.now)
            return '9 lidar frames'

        with redirect_stdout(io.StringIO()) as output:
            clock.now = 0.5
            timing.maybe_report(0.5, stats)
            clock.now = 1.0
            timing.maybe_report(1.0, stats)
        self.assertEqual(calls, [1.0])
        self.assertTrue(output.getvalue().rstrip().endswith('; 9 lidar frames'), output.getvalue())


if __name__ == '__main__':
    unittest.main()
