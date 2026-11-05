"""
Replay the golden traces in test/golden/ through today's policy classes.

The recorder (test/golden/record_goldens.py, its docstring lists every array)
drives the classes; this test drives fresh instances through the same inputs
and checks they compute the recorded numbers, which proves the recording is
deterministic. HOMIE is loaded from the g1_homie source tree beside this
package and needs onnxruntime.
"""

import importlib.util
import os

import numpy as np
import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
ATOL = 1e-6


def _recorder():
    """Import test/golden/record_goldens.py by path (it is not installed)."""
    spec = importlib.util.spec_from_file_location(
        'record_goldens', os.path.join(GOLDEN_DIR, 'record_goldens.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('name', ['walk', 'almi', 'fame', 'homie'])
def test_current_classes_reproduce_their_goldens(name: str) -> None:
    """Today's class computes the golden's network input, output and targets from its inputs."""
    if name == 'homie':
        pytest.importorskip('onnxruntime', reason='HOMIE runs on onnxruntime')
    else:
        pytest.importorskip('torch', reason='the H1-2 policies run on torch')
    recorder = _recorder()
    golden = np.load(os.path.join(GOLDEN_DIR, f'{name}.npz'))
    assert set(golden.files) == set(recorder.INPUT_KEYS) | set(recorder.OUTPUT_KEYS[name])

    replay = recorder.run_policy(name, {key: golden[key] for key in recorder.INPUT_KEYS})

    assert set(replay) == set(recorder.OUTPUT_KEYS[name])
    for key in recorder.OUTPUT_KEYS[name]:
        assert replay[key].shape == golden[key].shape, key
        if golden[key].dtype == bool:
            np.testing.assert_array_equal(replay[key], golden[key], err_msg=key)
        else:
            np.testing.assert_allclose(replay[key], golden[key], rtol=0, atol=ATOL, err_msg=key)
