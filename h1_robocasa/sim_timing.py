"""Wall-clock pacing and optional stage timings for the single-step simulator."""
import math
import time


class RealtimePacer:
    """Keep a wall deadline without skipping physics or repaying long pauses."""

    def __init__(self, timestep, max_lag=0.1, *, clock=time.perf_counter, sleep=time.sleep):
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError('timestep must be finite and positive')
        if not math.isfinite(max_lag) or max_lag < 0:
            raise ValueError('max_lag must be finite and nonnegative')
        self.timestep = timestep
        self.max_lag = max_lag
        self.clock = clock
        self.sleep = sleep
        self.deadline = clock()

    def wait(self):
        self.deadline += self.timestep
        now = self.clock()
        delay = self.deadline - now
        if delay > 0:
            self.sleep(delay)
        elif -delay > self.max_lag:
            # Renderer initialization/debugger pauses must not create minutes
            # of catch-up. Only the wall deadline changes, never simulation time.
            self.deadline = now - self.max_lag


class LoopTimings:
    """Report mean/max stage milliseconds and RTF every few wall seconds."""

    def __init__(self, enabled=False, interval=5.0, *, clock=time.perf_counter):
        self.enabled = enabled
        self.interval = interval
        self.clock = clock
        self.wall_start = clock() if enabled else 0.0
        self.sim_start = 0.0
        self.stages = {}

    def reset(self, sim_time):
        self.wall_start = self.clock() if self.enabled else 0.0
        self.sim_start = sim_time
        self.stages.clear()

    def start(self):
        return self.clock() if self.enabled else 0.0

    def stop(self, name, start):
        if self.enabled:
            elapsed = self.clock() - start
            total, count, longest = self.stages.get(name, (0.0, 0, 0.0))
            self.stages[name] = (total + elapsed, count + 1, max(longest, elapsed))

    def maybe_report(self, sim_time):
        if not self.enabled:
            return
        now = self.clock()
        wall = now - self.wall_start
        if wall < self.interval:
            return
        fields = ' '.join(
            f'{name}={1000 * total / count:.2f}/{1000 * longest:.2f}ms(n={count})'
            for name, (total, count, longest) in self.stages.items()
        )
        print(f'[sim-timing] RTF={(sim_time - self.sim_start) / wall:.3f} '
              f'(stage mean/max) {fields}', flush=True)
        self.wall_start = now
        self.sim_start = sim_time
        self.stages.clear()
