"""Pace the RoboCasa sim loop to a target real-time factor."""
import math
import time


class RealtimePacer:
    """Hold a sim loop to a wall-clock deadline per physics step, at rate sim seconds per wall second.

    rate is GOLEM_SIM_RATE (sim_common.sim_rate_from_env): 1 is real time, 2 twice real time,
    math.inf never sleeps. It never skips a physics step and never repays more than max_lag
    of wall lag, at any rate, so a long pause does not turn into minutes of catch-up.
    """

    def __init__(self, timestep, rate=1.0, max_lag=0.1, *, clock=time.perf_counter, sleep=time.sleep):
        """Pace steps of timestep sim seconds at rate, repaying at most max_lag (wall s) of lag."""
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError('timestep must be finite and positive')
        if math.isnan(rate) or rate <= 0:
            raise ValueError('rate must be positive (math.inf: unpaced)')
        if not math.isfinite(max_lag) or max_lag < 0:
            raise ValueError('max_lag must be finite and nonnegative')
        self.timestep = timestep
        self.rate = rate
        self.max_lag = max_lag
        self.clock = clock
        self.sleep = sleep
        self.deadline = clock()

    def wait(self):
        """Sleep until the next step's deadline, or move the deadline up when far behind.

        Returns the wall seconds slept (0 when behind, and always at rate math.inf).
        """
        if math.isinf(self.rate):
            return 0.0
        self.deadline += self.timestep / self.rate
        now = self.clock()
        delay = self.deadline - now
        if delay > 0:
            self.sleep(delay)
            return self.clock() - now
        if -delay > self.max_lag:
            # Only the wall deadline moves; sim time is never skipped.
            self.deadline = now - self.max_lag
        return 0.0
