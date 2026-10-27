"""The npz chunks the C++ ChunkLogger writes, read back with numpy.

test_chunk_writer (built by this package's CMake, not installed) logs deterministic
samples; the formulas below are the ones in test/test_chunk_writer.cpp. The keys, dtypes
and shapes are the ones g1_safety_layer's chunk_logger.py wrote, and combine_chunks.py
(installed under lib/golem_safety) must still read them.
"""
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from ament_index_python.packages import get_package_prefix
import numpy as np
import pytest

KEYS = (
    'time_stamp', 'mode', 'q', 'dq', 'ddq', 'tau_est', 'temperature', 'vol', 'sensor',
    'motorstate', 'reserve', 'imu_quaternion', 'imu_gyroscope', 'imu_accelerometer',
    'imu_rpy', 'imu_temperature', 'q_cmd', 'dq_cmd', 'tau_cmd', 'kp_cmd', 'kd_cmd',
)
DTYPES = {
    'time_stamp': np.float64, 'mode': np.uint8, 'q': np.float32, 'dq': np.float32,
    'ddq': np.float32, 'tau_est': np.float32, 'temperature': np.int16, 'vol': np.float32,
    'sensor': np.uint32, 'motorstate': np.uint32, 'reserve': np.uint32,
    'imu_quaternion': np.float32, 'imu_gyroscope': np.float32,
    'imu_accelerometer': np.float32, 'imu_rpy': np.float32, 'imu_temperature': np.int16,
    'q_cmd': np.float32, 'dq_cmd': np.float32, 'tau_cmd': np.float32, 'kp_cmd': np.float32,
    'kd_cmd': np.float32,
}


def writer_executable():
    path = os.environ.get('GOLEM_SAFETY_TEST_CHUNK_WRITER')
    assert path and Path(path).is_file(), f'GOLEM_SAFETY_TEST_CHUNK_WRITER is not a file: {path}'
    return path


def combine_script():
    prefix = Path(get_package_prefix('golem_safety'))
    script = prefix / 'lib' / 'golem_safety' / 'combine_chunks.py'
    assert script.is_file(), f'{script} is not installed'
    return script


def write_chunks(directory, per_chunk, samples, motors):
    result = subprocess.run(
        [writer_executable(), str(directory), 'chunk_', str(per_chunk), str(samples),
         str(motors)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def expected(first, count, motors):
    """The samples first .. first+count-1 as the writer makes them, one array per key."""
    k = np.arange(first, first + count, dtype=np.int64)[:, None]
    i = np.arange(motors, dtype=np.int64)[None, :]
    k1 = np.arange(first, first + count, dtype=np.float64)

    def pair(a, b):
        return np.stack([a, b], axis=-1)

    def imu(*columns):
        return np.stack([np.broadcast_to(c, k1.shape) for c in columns], axis=-1)

    return {
        'time_stamp': 1700000000.0 + 0.5 * k1,
        'mode': (k + i) % 251,
        'q': 0.25 * (k * 1000 + i),
        'dq': -0.5 * (k + i),
        'ddq': 0.125 * (7 * k + i),
        'tau_est': 0.0625 * (3 * k - i),
        'temperature': pair(
            np.broadcast_to(k - i, (count, motors)), np.broadcast_to(i - 2 * k, (count, motors))),
        'vol': np.broadcast_to(24.0 + 0.5 * (i % 8), (count, motors)),
        'sensor': pair(
            np.broadcast_to(1000 * k + i, (count, motors)),
            np.broadcast_to(0xFFFFFFFF - k, (count, motors))),
        'motorstate': 3 * k + i,
        'reserve': np.stack([
            np.broadcast_to(k, (count, motors)), np.broadcast_to(i, (count, motors)),
            np.full((count, motors), 7), np.full((count, motors), 0xDEADBEEF)], axis=-1),
        'imu_quaternion': imu(1.0, 0.5 * k1, -0.25, 2.0),
        'imu_gyroscope': imu(0.125 * k1, 0.25, -0.5),
        'imu_accelerometer': imu(9.5, -0.5 * k1, 0.75),
        'imu_rpy': imu(0.0625 * k1, -0.0625, 0.5),
        'imu_temperature': -k1.astype(np.int64),
        'q_cmd': 0.5 * (k + i),
        'dq_cmd': 0.25 * (k - i),
        'tau_cmd': 0.5 * (i + 2 * k),
        'kp_cmd': np.broadcast_to(100.0 + 0.5 * k, (count, motors)),
        'kd_cmd': 2.0 + 0.25 * i + 0.125 * k,
    }


def assert_chunk(data, first, count, motors):
    assert list(data.keys()) == list(KEYS)
    want = expected(first, count, motors)
    for key in KEYS:
        got = data[key]
        assert got.dtype == np.dtype(DTYPES[key]).newbyteorder('<'), key
        assert got.shape == want[key].shape, key
        np.testing.assert_array_equal(got, want[key].astype(DTYPES[key]), err_msg=key)


@pytest.mark.parametrize('motors', [27, 29, 35])
def test_chunks_load_in_numpy(tmp_path, motors):
    write_chunks(tmp_path, per_chunk=100, samples=250, motors=motors)
    files = sorted(tmp_path.glob('chunk_*.npz'))
    assert [f.name for f in files] == [f'chunk_{n:06d}.npz' for n in range(3)]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f.name for f in files]

    for n, (first, count) in enumerate([(0, 100), (100, 100), (200, 50)]):
        with np.load(files[n], allow_pickle=False) as data:
            assert_chunk(data, first, count, motors)


def test_chunks_are_valid_stored_zips(tmp_path):
    write_chunks(tmp_path, per_chunk=40, samples=40, motors=29)
    with zipfile.ZipFile(tmp_path / 'chunk_000000.npz') as archive:
        assert archive.testzip() is None
        assert [info.filename for info in archive.infolist()] == [f'{k}.npy' for k in KEYS]
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
        for info in archive.infolist():
            assert info.compress_size == info.file_size


def test_combine_chunks_concatenates(tmp_path):
    write_chunks(tmp_path, per_chunk=100, samples=250, motors=29)
    combined = tmp_path / 'out' / 'combined.npz'
    result = subprocess.run(
        [sys.executable, str(combine_script()), '--load', str(tmp_path), '--prefix', 'chunk_',
         '--save', str(combined)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'combined chunks into' in result.stdout

    with np.load(combined, allow_pickle=False) as data:
        assert_chunk(data, 0, 250, 29)


def test_combine_chunks_defaults_to_the_golem_safety_directory():
    # argparse does not print defaults in --help, and running without arguments would
    # combine whatever a real run left in the default directory.
    source = combine_script().read_text()
    assert "default='/tmp/golem_safety'" in source
    assert "default='/tmp/golem_safety/combined.npz'" in source
    assert 'g1_safety_layer' not in source
