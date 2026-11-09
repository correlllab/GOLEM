"""Pace both RoboCasa sims' loops to a target real-time factor; report the H1-2's per-stage timings (--timing)."""
import math
import threading
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


class LoopTimings:
    """Collect per-stage wall times and print mean/max ms and the real-time factor.

    Disabled (the default) it does nothing. The sensor worker threads record
    stages too, so updates and reports take a lock.
    """

    def __init__(self, enabled=False, interval=5.0, *, clock=time.perf_counter):
        """Report every interval wall seconds when enabled."""
        self.enabled = enabled
        self.interval = interval
        self.clock = clock
        self.wall_start = clock() if enabled else 0.0
        self.sim_start = 0.0
        self.stages = {}
        self._lock = threading.Lock()

    def reset(self, sim_time):
        """Start a new report interval at sim_time (s) and drop the stages so far."""
        self.wall_start = self.clock() if self.enabled else 0.0
        self.sim_start = sim_time
        with self._lock:
            self.stages.clear()

    def start(self):
        """Return the start mark of a stage, to pass to stop()."""
        return self.clock() if self.enabled else 0.0

    def stop(self, name, start):
        """Record the wall time since start under the stage name."""
        if self.enabled:
            elapsed = self.clock() - start
            with self._lock:
                total, count, longest = self.stages.get(name, (0.0, 0, 0.0))
                self.stages[name] = (total + elapsed, count + 1, max(longest, elapsed))

    def maybe_report(self, sim_time, stats=None):
        """Print the report and start a new interval once interval wall seconds have passed.

        stats, when given, is called at each report and what it returns ("" for nothing) is
        appended to the line (the sensors' counts).
        """
        if not self.enabled:
            return
        now = self.clock()
        wall = now - self.wall_start
        if wall < self.interval:
            return
        with self._lock:
            stages, self.stages = self.stages, {}
        fields = ' '.join(
            f'{name}={1000 * total / count:.2f}/{1000 * longest:.2f}ms(n={count})'
            for name, (total, count, longest) in stages.items()
        )
        extra = stats() if stats is not None else ''
        print(f'[sim-timing] RTF={(sim_time - self.sim_start) / wall:.3f} '
              f'(stage mean/max) {fields}' + (f'; {extra}' if extra else ''), flush=True)
        self.wall_start = now
        self.sim_start = sim_time
