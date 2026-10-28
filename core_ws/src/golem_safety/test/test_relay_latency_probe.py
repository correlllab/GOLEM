'''
tools/relay_latency_probe.py's matching and statistics, on synthetic input and output
streams with known delays. No ROS: the probe imports rclpy only when it records.
'''

import importlib.util
from pathlib import Path

import pytest

PROBE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'relay_latency_probe.py'
PERIOD = 0.002  # s, 500 Hz


def load_probe():
    spec = importlib.util.spec_from_file_location('relay_latency_probe', PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_probe()


def pose(index: int) -> tuple:
    '''An upper-slot q that no other index gives'''
    return tuple(0.001 * index + 0.0001 * slot for slot in range(15))


def test_latencies_are_the_known_delays():
    # 1001 inputs, each with its own q; the relay sends each once, after a known delay:
    # 980 after 0.4 ms, 11 after 1.2 ms, 10 after 3.0 ms (past the next input, whose q differs)
    delays = [0.0004] * 980 + [0.0012] * 11 + [0.0030] * 10
    inputs = [(i * PERIOD, pose(i)) for i in range(len(delays))]
    outputs = sorted((t + delay, q) for (t, q), delay in zip(inputs, delays))

    matching = probe.match_outputs(inputs, outputs)

    assert matching.unmatched == 0
    assert len(matching.latencies) == len(delays)
    stats = probe.latency_stats(matching.latencies)
    assert stats['matched'] == 1001
    assert stats['median_ms'] == pytest.approx(0.4)
    # linear interpolation, as numpy.percentile: rank 0.99 * 1000 = 990, the last 1.2 ms
    assert stats['p99_ms'] == pytest.approx(1.2)
    assert stats['max_ms'] == pytest.approx(3.0)


def test_the_zero_stream_and_a_clipped_sample_match_no_input():
    zero = tuple(0.0 for _ in range(15))
    inputs = [(1.0 + i * PERIOD, pose(i + 1)) for i in range(100)]
    # 50 zero-gain commands before the controller's first, then one output per input 0.5 ms
    # later; output 40 has one slot clipped
    outputs = [(0.9 + i * PERIOD, zero) for i in range(50)]
    for i, (t, q) in enumerate(inputs):
        if i == 40:
            q = q[:3] + (q[3] - 0.25,) + q[4:]
        outputs.append((t + 0.0005, q))

    matching = probe.match_outputs(inputs, outputs)

    assert matching.no_equal_input == 51
    assert matching.repeated == 0
    assert matching.unmatched == 51
    assert len(matching.latencies) == 99
    assert probe.latency_stats(matching.latencies)['median_ms'] == pytest.approx(0.5)


def test_a_held_command_is_unmatched():
    # The controller sends one q at 500 Hz for 100 samples, a second q for 50, then stops; the
    # relay keeps sending its newest command every 2 ms, 0.5 ms after each input's arrival
    inputs = [(i * PERIOD, pose(1)) for i in range(100)]
    inputs += [((100 + i) * PERIOD, pose(2)) for i in range(50)]
    outputs = [(t + 0.0005, q) for t, q in inputs]
    # held: 30 more outputs of the last q after the controller stopped
    outputs += [((150 + i) * PERIOD + 0.0005, pose(2)) for i in range(30)]

    matching = probe.match_outputs(inputs, outputs)

    assert len(matching.latencies) == 150
    assert matching.repeated == 30
    assert matching.no_equal_input == 0
    assert matching.unmatched == 30
    stats = probe.latency_stats(matching.latencies)
    assert stats['median_ms'] == pytest.approx(0.5)
    assert stats['p99_ms'] == pytest.approx(0.5)


def test_a_repeated_q_matches_its_newest_input():
    # Inputs every 2 ms repeat one q; a 7 ms gap in the controller's stream leaves the relay
    # sending the last one again 3 times (held), and the next input is matched as it arrives
    times = [i * PERIOD for i in range(10)] + [0.025 + i * PERIOD for i in range(10)]
    inputs = [(t, pose(3)) for t in times]
    outputs = [(0.0007 + i * PERIOD, pose(3)) for i in range(23)]

    matching = probe.match_outputs(inputs, outputs)

    # outputs at 0.7 .. 18.7 ms match inputs 0-9 (0.7 ms); 20.7, 22.7 and 24.7 ms repeat input 9;
    # 26.7 ms matches the input at 25 ms (1.7 ms), and so on every 2 ms
    assert matching.repeated == 3
    assert matching.no_equal_input == 0
    assert len(matching.latencies) == 20
    assert sorted(matching.latencies)[:10] == pytest.approx([0.0007] * 10)
    assert sorted(matching.latencies)[10:] == pytest.approx([0.0017] * 10)


def test_output_rate_from_arrival_times():
    # 1001 arrivals: 999 intervals of 2 ms and one gap of 10 ms
    times = [0.0]
    for i in range(1000):
        times.append(times[-1] + (0.010 if i == 500 else PERIOD))

    stats = probe.rate_stats(times)

    assert stats['samples'] == 1001
    assert stats['mean_hz'] == pytest.approx(1000 / 2.008)
    assert stats['p99_interval_ms'] == pytest.approx(2.0)
    assert stats['max_gap_ms'] == pytest.approx(10.0)


def test_statistics_need_samples():
    with pytest.raises(ValueError):
        probe.rate_stats([1.0])
    with pytest.raises(ValueError):
        probe.latency_stats([])


@pytest.mark.parametrize('value', [None, '', '0', ' 0', 'one'])
def test_domain_zero_and_an_unusable_domain_are_refused(value):
    refusal = probe.domain_refusal(value)
    assert refusal is not None
    assert 'ROS_DOMAIN_ID' in refusal


def test_a_sim_domain_is_accepted():
    assert probe.domain_refusal('193') is None
