"""Test the tick helpers: the uint32 tick difference, the 20 ms step grid and the tick lag."""

import pytest

from golem_lowerbody_rl.timing import TICK_SECONDS, tick_delta, TickGrid, TickLag

WRAP = 2 ** 32


def _due_ticks(grid, ticks):
    """Return the ticks of ticks on which grid is due."""
    return [tick for tick in ticks if grid.due(tick)]


def test_grid_steps_every_20_ms():
    """At 500 Hz the grid is due every 20th ms, from the first sample on."""
    assert TICK_SECONDS == 0.001
    due = _due_ticks(TickGrid(), range(1000, 1200, 2))
    assert due == list(range(1000, 1200, 20))


def test_grid_steps_on_the_first_sample_past_each_grid_tick():
    """With samples 7 ms apart, a step is due on the first sample at or past its grid tick."""
    due = _due_ticks(TickGrid(period_s=0.02), range(0, 200, 7))
    assert due[0] == 0
    gaps = [b - a for a, b in zip(due, due[1:])]
    # never closer than half a period, 20 ms on average
    assert min(gaps) >= 10
    assert (due[-1] - due[0]) / len(gaps) == pytest.approx(20, abs=2)


def test_grid_skips_missed_cells_without_bunching():
    """After a gap of several periods, the next step comes a full grid cell later."""
    grid = TickGrid()
    assert grid.due(0)
    assert grid.due(95)  # 4 periods missed: one step, not five
    assert not grid.due(99)
    assert not grid.due(119)
    assert grid.due(120)


def test_grid_does_not_step_on_a_repeated_tick():
    """A tick that did not advance is not due again."""
    grid = TickGrid()
    assert grid.due(40)
    assert not grid.due(40)
    assert grid.due(60)


def test_grid_crosses_the_wrap():
    """The grid keeps its 20 ms spacing across the uint32 wrap."""
    start = WRAP - 50
    ticks = [(start + 2 * i) % WRAP for i in range(60)]
    due = _due_ticks(TickGrid(), ticks)
    assert [tick_delta(a, b) for a, b in zip(due, due[1:])] == [20] * (len(due) - 1)
    assert len(due) == 6


def test_grid_period_must_be_whole_ticks():
    """A period that is not a whole number of ticks is refused."""
    with pytest.raises(ValueError, match='period'):
        TickGrid(period_s=0.0205)
    with pytest.raises(ValueError, match='period'):
        TickGrid(period_s=0.0)


def test_tick_wraps():
    """The difference is taken across the uint32 wrap and is negative when the tick goes back."""
    assert tick_delta(5, 25) == 20
    assert tick_delta(WRAP - 5, 3) == 8
    assert tick_delta(3, WRAP - 5) == -8
    assert tick_delta(25, 5) == -20
    assert tick_delta(7, 7) == 0


def test_tick_lag():
    """The lag is the delay beyond the earliest arrival, unchanged by the uint32 wrap."""
    lag = TickLag()
    assert lag.update(1000, 50.0) == 0.0
    # on time: 10 ms of tick in 10 ms of clock
    assert lag.update(1010, 50.010) == pytest.approx(0.0, abs=0.01)
    # 5 ms late, less the 0.5 ms/s drift allowance
    assert lag.update(1020, 50.025) == pytest.approx(5.0, abs=0.01)
    # an earlier arrival lowers the baseline at once
    assert lag.update(1030, 50.028) == pytest.approx(0.0, abs=1e-6)


def test_tick_lag_crosses_the_wrap():
    """A tick that wraps does not jump the lag."""
    lag = TickLag()
    lag.update(WRAP - 10, 1.0)
    assert lag.update(5, 1.015) == pytest.approx(0.0, abs=0.01)
    assert lag.update(15, 1.027) == pytest.approx(2.0, abs=0.01)


def test_tick_lag_follows_slow_drift():
    """A delay that builds as slowly as clock drift is absorbed into the baseline."""
    lag = TickLag()
    lag.update(0, 0.0)
    now, tick = 0.0, 0
    for _ in range(100):
        # 0.4 ms of drift per s, under the 0.5 ms/s allowance
        now += 1.0
        tick += 1000
        value = lag.update(tick, now + 0.0004 * now)
    assert value == pytest.approx(0.0, abs=1e-6)
