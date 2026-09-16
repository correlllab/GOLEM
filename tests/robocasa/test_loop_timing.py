import io
import unittest
from contextlib import redirect_stdout

from sim_timing import RealtimePacer, LoopTimings


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


class PacingTests(unittest.TestCase):
    def test_fast_steps_track_the_original_deadline(self):
        clock = FakeClock()
        pacer = RealtimePacer(0.01, clock=clock, sleep=clock.sleep)
        for _ in range(10):
            clock.now += 0.002
            pacer.wait()
        self.assertAlmostEqual(clock.now, 0.1)

    def test_sensor_overrun_is_recovered_without_skipping_steps(self):
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
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RealtimePacer(value)

    def test_timing_is_silent_by_default(self):
        timing = LoopTimings()
        with redirect_stdout(io.StringIO()) as output:
            timing.stop('physics', timing.start())
            timing.maybe_report(1.0)
        self.assertEqual(output.getvalue(), '')

    def test_timing_baseline_excludes_startup_and_existing_sim_time(self):
        clock = FakeClock()
        timing = LoopTimings(enabled=True, interval=1, clock=clock)
        clock.now = 10.0
        timing.reset(5.0)
        clock.now = 11.0
        with redirect_stdout(io.StringIO()) as output:
            timing.maybe_report(5.5)
        self.assertIn('RTF=0.500', output.getvalue())

    def test_timing_reports_sim_wall_ratio_and_stage_cost(self):
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


if __name__ == '__main__':
    unittest.main()
