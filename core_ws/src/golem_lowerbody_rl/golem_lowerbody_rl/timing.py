"""
Read /lowstate's tick: its uint32 differences, the policy's step grid and the tick lag.

The tick counts milliseconds of the robot's clock (the sim's clock in the sims) and wraps
at 2**32.
"""

__all__ = ['TICK_MODULO', 'TICK_MS', 'TICK_SECONDS', 'TickGrid', 'TickLag', 'tick_delta']

TICK_SECONDS = 0.001  # s per tick
TICK_MODULO = 2 ** 32
TICK_MS = TICK_SECONDS * 1000.0  # ms per tick
# how fast (ms per s) TickLag's baseline may rise to follow drift between the robot's
# clock and the node's, as golem_upper_ik's TickLagTracker
LAG_DRIFT_MS_PER_S = 0.5


def tick_delta(prev: int, now: int) -> int:
    """Return now - prev in ticks across the uint32 wrap; negative if the tick went back."""
    delta = (int(now) - int(prev)) % TICK_MODULO
    return delta - TICK_MODULO if delta >= TICK_MODULO // 2 else delta


class TickGrid:
    """
    Say on which ticks a step is due, on a grid one period of ticks apart.

    The first tick passed sets the grid. A step is then due on the first tick at or past
    its grid tick, and the next one at the first grid tick at least half a period after
    that tick, so a step that comes late is not followed by a bunched one: steps average
    one period of ticks apart and are never closer than half a period. A tick that did not
    advance past the next grid tick is not due.
    """

    def __init__(self, period_s: float = 0.02) -> None:
        """Step every period_s (s of tick); ValueError unless a whole number of ticks."""
        ticks = period_s / TICK_SECONDS
        if not (ticks >= 1.0 and abs(ticks - round(ticks)) < 1e-6):
            raise ValueError(f'the grid period {period_s} s must be a whole number of '
                             f'{TICK_SECONDS} s ticks')
        self.period_ticks = int(round(ticks))
        self._due_tick: int | None = None

    def due(self, tick: int) -> bool:
        """Return True if a step is due on tick, and then set the next grid tick."""
        period = self.period_ticks
        if self._due_tick is None:
            self._due_tick, late = int(tick), 0
        else:
            late = tick_delta(self._due_tick, tick)
            if late < 0:
                return False
        # the smallest k >= 1 with k * period >= late + period / 2
        periods = (2 * late + 3 * period - 1) // (2 * period)
        self._due_tick = (self._due_tick + periods * period) % TICK_MODULO
        return True


class TickLag:
    """
    Measure how late (ms) samples stamped with the robot's millisecond tick arrive.

    The baseline is the earliest arrival seen; it may rise by LAG_DRIFT_MS_PER_S to follow
    clock drift, so only a delay that builds faster than drift registers. A tick that wraps
    the uint32 counter does not jump the lag.
    """

    def __init__(self) -> None:
        """Start with no baseline: the first sample's lag is 0."""
        self._last_tick: int | None = None
        self._wraps = 0
        self._base_ms: float | None = None
        self._base_time = 0.0

    def update(self, tick: int, now: float) -> float:
        """Return the lag (ms) of a sample with tick that arrived at now (s, the node's clock)."""
        tick = int(tick)
        if self._last_tick is not None and tick < self._last_tick - TICK_MODULO // 2:
            self._wraps += 1
        self._last_tick = tick
        offset_ms = now * 1000.0 - (tick + self._wraps * TICK_MODULO) * TICK_MS
        if self._base_ms is None:
            self._base_ms = offset_ms
        else:
            drift_ms = LAG_DRIFT_MS_PER_S * (now - self._base_time)
            self._base_ms = min(offset_ms, self._base_ms + drift_ms)
        self._base_time = now
        return offset_ms - self._base_ms
