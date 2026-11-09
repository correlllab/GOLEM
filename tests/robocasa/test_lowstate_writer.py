"""Check unitree_common.LowStateWriter, the sims' numpy path for rt/lowstate.

The writer must put on the wire exactly the bytes cyclonedds' DataWriter.write puts there for a
LowState_ filled the way the sims filled it through unitree_sdk2py: the IDL serializer's CDR,
padded to 4 bytes, with the SDK's CRC. No DDS participant opens: the publishers are in-process
stand-ins and cyclonedds' ddspy_write is replaced for the length of a test. It runs in the
standard runner (tests/robocasa/run_in_container.sh -p test_lowstate_writer.py).
"""
import contextlib
import io
import time
import types
import unittest
from unittest import mock

import numpy as np

from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
from unitree_sdk2py.utils.crc import CRC

import unitree_common
from unitree_common import LowStateWriter, rpy_from_quat

# (mode_machine, motor slots filled) of the H1-2 and the G1.
ROBOTS = ((6, 27), (5, 29))
SAMPLES = 200
TIMED_SAMPLES = 2000
REF = 4242   # the stand-in DataWriter's entity handle


def sdk_crc():
    """Return a CRC object running unitree_sdk2py's own code, without fast_crc's numpy packing."""
    crc = object.__new__(CRC)   # CRC() is a per-process singleton that fast_crc() patches
    crc.__init__()
    return crc


SDK_CRC = sdk_crc()


class FakePublisher:
    """Stand in for ChannelPublisher, keeping the samples its Write is given."""

    def __init__(self):
        """Start with nothing written."""
        self.samples = []

    def Write(self, sample):
        """Keep the sample."""
        self.samples.append(sample)


def dds_publisher():
    """Return a stand-in ChannelPublisher whose channel holds a cyclonedds-like DataWriter.

    It has the private attribute chain of unitree_sdk2py's ChannelPublisher, ending in an object
    with DataWriter's _ref and _use_version_2.
    """
    writer = types.SimpleNamespace(_ref=REF, _use_version_2=None)
    channel = types.SimpleNamespace(_Channel__writer=types.SimpleNamespace(_Writer__writer=writer))
    pub = FakePublisher()
    pub._ChannelPublisher__channel = channel
    return pub


def random_states(rng, motors, count):
    """Yield count finite states (tick, q, dq, tau, quat_wxyz, gyro, acc), the quaternion unit."""
    for _ in range(count):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        yield (int(rng.integers(0, 2**32)), rng.normal(size=motors) * 2.0,
               rng.normal(size=motors) * 10.0, rng.normal(size=motors) * 50.0, quat,
               rng.normal(size=3), rng.normal(size=3) * 9.81)


def sdk_message(mode_machine, motors, tick, q, dq, tau, quat, gyro, acc, crc=SDK_CRC):
    """Return the LowState_ the sims' after_step filled through unitree_sdk2py, with its CRC."""
    msg = LowState_default()
    msg.mode_pr = unitree_common.MODE_PR
    msg.mode_machine = mode_machine
    msg.tick = tick
    for i, motor in enumerate(msg.motor_state[:motors]):
        motor.q = float(q[i])
        motor.dq = float(dq[i])
        motor.tau_est = float(tau[i])
    imu = msg.imu_state
    quat = [float(v) for v in quat]
    imu.quaternion = quat
    imu.gyroscope = [float(v) for v in gyro]
    imu.accelerometer = [float(v) for v in acc]
    imu.rpy = rpy_from_quat(*quat)
    msg.crc = crc.Crc(msg)
    return msg


def sdk_bytes(msg, use_version_2):
    """Return what cyclonedds' DataWriter.write hands ddspy_write for msg."""
    ser = msg.serialize(use_version_2=use_version_2)
    return ser.ljust((len(ser) + 3) & ~3, b'\0')


def quiet(fn, *args):
    """Call fn(*args) with its printed lines captured; return its result and the lines."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args)
    return result, out.getvalue().splitlines()


class LowStateWriterTests(unittest.TestCase):
    """Compare LowStateWriter with the SDK's serializer and CRC."""

    def test_lowstate_bytes_match_the_sdk(self):
        """The writer's bytes equal the SDK's serialized, padded, CRC-signed LowState_ for random states."""
        for mode_machine, motors in ROBOTS:
            writer, lines = quiet(LowStateWriter, dds_publisher(), mode_machine, motors)
            self.assertEqual(lines, [])
            self.assertTrue(writer.fast)
            rng = np.random.default_rng(0)
            for state in random_states(rng, motors, SAMPLES):
                expected = sdk_bytes(sdk_message(mode_machine, motors, *state), writer._use_version_2)
                self.assertEqual(writer.bytes_for(*state), expected)

    def test_write_hands_ddspy_write_the_bytes(self):
        """write() hands cyclonedds' ddspy_write the writer's entity and the SDK's bytes, and never the publisher."""
        pub = dds_publisher()
        writer = LowStateWriter(pub, 6, 27)
        sent = []
        with mock.patch.object(unitree_common, 'ddspy_write',
                               lambda ref, data: sent.append((ref, bytes(data))) or 0):
            states = list(random_states(np.random.default_rng(1), 27, 3))
            for state in states:
                writer.write(*state)
        self.assertEqual(sent, [(REF, sdk_bytes(sdk_message(6, 27, *s), None)) for s in states])
        self.assertEqual(pub.samples, [])

    def test_failed_write_is_warned(self):
        """A write that cyclonedds refuses is reported in a rate-limited warning, not raised."""
        writer = LowStateWriter(dds_publisher(), 5, 29)
        state = next(random_states(np.random.default_rng(2), 29, 1))
        with mock.patch.object(unitree_common, 'ddspy_write', lambda ref, data: -3):
            _, lines = quiet(lambda: [writer.write(*state) for _ in range(3)])
        self.assertEqual(len(lines), 1, lines)
        self.assertIn('rt/lowstate', lines[0])

    def test_publisher_without_a_datawriter_takes_the_sdk_path(self):
        """A publisher with no cyclonedds DataWriter gets SDK LowState_ samples through its Write, with a warning."""
        pub = FakePublisher()
        writer, lines = quiet(LowStateWriter, pub, 6, 27)
        self.assertFalse(writer.fast)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("SDK's serializer", lines[0])
        state = next(random_states(np.random.default_rng(3), 27, 1))
        writer.write(*state)
        self.assertEqual(len(pub.samples), 1)
        self.assertEqual(sdk_bytes(pub.samples[0], None), sdk_bytes(sdk_message(6, 27, *state), None))

    def test_an_unforeseen_error_in_the_check_takes_the_sdk_path(self):
        """Any exception while building the numpy path leaves it off, with one warning naming it."""
        def broken(self):
            """Fail as an SDK change the probes do not know would."""
            raise TypeError("unexpected SDK layout")

        pub = dds_publisher()
        with mock.patch.object(LowStateWriter, '_build', broken):
            writer, lines = quiet(LowStateWriter, pub, 6, 27)
        self.assertFalse(writer.fast)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("unexpected SDK layout", lines[0])
        self.assertIn("SDK's serializer", lines[0])
        state = next(random_states(np.random.default_rng(5), 27, 1))
        self.assertEqual(writer.bytes_for(*state), sdk_bytes(sdk_message(6, 27, *state), None))

    def test_layout_mismatch_takes_the_sdk_path(self):
        """Bytes that differ from the SDK's at start-up leave the fast path off, with one warning."""
        real_fill = LowStateWriter._fill

        def corrupt(self, *args):
            """Fill as the writer does, then flip one byte of the motor data."""
            buf = real_fill(self, *args)
            buf[100] ^= 0xFF
            return buf

        pub = dds_publisher()
        with mock.patch.object(LowStateWriter, '_fill', corrupt):
            writer, lines = quiet(LowStateWriter, pub, 5, 29)
            self.assertFalse(writer.fast)
            self.assertEqual(len(lines), 1, lines)
            self.assertIn("SDK's serializer", lines[0])
            state = next(random_states(np.random.default_rng(4), 29, 1))
            with mock.patch.object(unitree_common, 'ddspy_write', side_effect=AssertionError):
                writer.write(*state)
                self.assertEqual(writer.bytes_for(*state), sdk_bytes(sdk_message(5, 29, *state), None))
        self.assertEqual(len(pub.samples), 1)
        self.assertEqual(sdk_bytes(pub.samples[0], None), sdk_bytes(sdk_message(5, 29, *state), None))

    def test_missing_imu_leaves_it_zero(self):
        """IMU arguments of None leave the IMU fields at the IDL's zeros, as an H1-2 model without IMU sensors gets."""
        writer = LowStateWriter(dds_publisher(), 6, 27)
        tick, q, dq, tau, *_ = next(random_states(np.random.default_rng(5), 27, 1))
        msg = sdk_message(6, 27, tick, q, dq, tau, np.zeros(4), np.zeros(3), np.zeros(3))
        self.assertEqual(writer.bytes_for(tick, q, dq, tau, None, None, None), sdk_bytes(msg, None))

    def test_lowstate_writer_is_faster(self):
        """Print the mean time per sample of the writer's bytes and of the SDK's fill, CRC and serialize."""
        crc = unitree_common.fast_crc()
        for mode_machine, motors in ROBOTS:
            writer = LowStateWriter(dds_publisher(), mode_machine, motors)
            states = list(random_states(np.random.default_rng(6), motors, TIMED_SAMPLES))
            start = time.perf_counter()
            for state in states:
                writer.bytes_for(*state)
            fast_us = (time.perf_counter() - start) / TIMED_SAMPLES * 1e6
            start = time.perf_counter()
            for state in states:
                sdk_bytes(sdk_message(mode_machine, motors, *state, crc=crc), None)
            sdk_us = (time.perf_counter() - start) / TIMED_SAMPLES * 1e6
            print(f'\nrt/lowstate, {motors} motors: SDK path {sdk_us:.1f} us per sample, '
                  f'fast path {fast_us:.1f} us per sample')


if __name__ == '__main__':
    unittest.main()
