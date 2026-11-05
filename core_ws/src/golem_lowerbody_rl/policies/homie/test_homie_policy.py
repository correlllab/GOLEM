"""Tests for homie_policy.py; no simulator needed.

    python test_homie_policy.py
    python -m pytest test_homie_policy.py
"""
import math
import os
import tempfile

import numpy as np
import yaml

from homie_policy import HERE, HomiePolicy

# the legs' default angles in g1.yaml
DEFAULT_ANGLES = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2
STAND = dict(cmd=[0.0, 0.0, 0.0], height=0.74, gyro=[0.0, 0.0, 0.0], quat_wxyz=[1.0, 0.0, 0.0, 0.0],
             q=DEFAULT_ANGLES + [0.0] * 15, dq=[0.0] * 27)


class FakeSession:
    """Stands in for the ONNX session: records each input and returns the given actions"""

    def __init__(self, actions):
        self.actions = list(actions)
        self.inputs = []

    def run(self, output_names, feed):
        (x,) = feed.values()
        self.inputs.append(x.copy())
        return [np.array([self.actions.pop(0)], dtype=np.float32)]


def raises(error, f, *args, **kwargs):
    """The error f(*args, **kwargs) raises, or None"""
    try:
        f(*args, **kwargs)
    except error as e:
        return e
    return None


def test_observation_layout():
    policy = HomiePolicy()
    i = np.arange(1, 28)
    q = np.concatenate([DEFAULT_ANGLES, np.zeros(15)]) + 0.01 * i
    # pitched 30 deg nose down: gravity is (sin 30, 0, -cos 30) in the pelvis frame
    quat = [math.cos(math.radians(15)), 0.0, math.sin(math.radians(15)), 0.0]
    obs = policy.observation([0.5, -0.25, 0.4], 0.6, [0.2, -0.4, 1.0], quat, q, 1.0 * i)
    expected = np.concatenate([
        [1.0, -0.5, 0.2],               # cmd * (2, 2, 0.5)
        [0.6],                          # height, unscaled
        [0.1, -0.2, 0.5],               # angular velocity * 0.5
        [0.5, 0.0, -math.sqrt(3) / 2],  # gravity
        0.01 * i,                       # (q - default) * 1.0; waist and arms default to 0
        0.05 * i,                       # dq * 0.05
        np.zeros(12),                   # last action: none yet
    ])
    assert obs.dtype == np.float32 and obs.shape == (76,)
    np.testing.assert_allclose(obs, expected, rtol=0, atol=1e-6)


def test_history_order():
    policy = HomiePolicy()
    policy.session = FakeSession([np.zeros(12)] * 9)
    for k in range(1, 9):
        policy.step(**dict(STAND, height=0.1 * k))
    inputs = policy.session.inputs
    assert inputs[0].shape == (1, 456) and inputs[0].dtype == np.float32
    heights = [x.reshape(6, 76)[:, 3] for x in inputs]
    # oldest first, zeros before the first step
    assert not inputs[0][0, :5 * 76].any()
    np.testing.assert_allclose(heights[0], [0, 0, 0, 0, 0, 0.1], atol=1e-6)
    np.testing.assert_allclose(heights[2], [0, 0, 0, 0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(heights[7], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], atol=1e-6)

    # reset() starts over from zeros
    policy.reset()
    policy.step(**dict(STAND, height=0.5))
    assert not inputs[8][0, :5 * 76].any()
    np.testing.assert_allclose(inputs[8].reshape(6, 76)[:, 3], [0, 0, 0, 0, 0, 0.5], atol=1e-6)


def test_action_to_target():
    policy = HomiePolicy()
    action = [150.0, -150.0, 1.0, -1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    clipped = [100.0, -100.0, 1.0, -1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]  # clip_actions 100
    policy.session = FakeSession([action, action, action])
    targets = policy.step(**STAND)
    # the targets are for the leg joints: default_angles + 0.25 * the clipped action
    assert policy.leg_joints == policy.policy_joints[:12] and targets.shape == (12,)
    expected = np.array(DEFAULT_ANGLES) + 0.25 * np.array(clipped)
    np.testing.assert_allclose(targets, expected, rtol=0, atol=1e-6)

    # the clipped action is the next step's last action, until reset()
    policy.step(**STAND)
    np.testing.assert_allclose(policy.session.inputs[1][0, -12:], clipped, atol=1e-6)
    policy.reset()
    policy.step(**STAND)
    assert not policy.session.inputs[2][0, -12:].any()


def test_policy_runs():
    policy = HomiePolicy()
    targets = policy.step(**STAND)
    assert targets.shape == (12,) and targets.dtype == np.float32 and np.isfinite(targets).all()


def test_bad_state_rejected():
    # the real network: a NaN let into the history would reach every later
    # output through the last action
    policy, reference = HomiePolicy(), HomiePolicy()
    walk = dict(STAND, cmd=[0.5, 0.0, 0.0])
    for _ in range(3):
        policy.step(**walk)
        reference.step(**walk)
    history, action = np.concatenate(policy.history), policy.action.copy()
    cases = [
        (dict(walk, dq=[math.nan] + [0.0] * 26), 'observation is not finite at [37]'),
        (dict(walk, height=math.inf), 'observation is not finite at [3]'),
        (dict(walk, gyro=[0.0] * 4), 'observation has 77 values, expected 76'),
    ]
    for state, message in cases:
        error = raises(ValueError, policy.step, **state)
        assert str(error) == message, error
        # the history and the last action are unchanged
        np.testing.assert_array_equal(np.concatenate(policy.history), history)
        np.testing.assert_array_equal(policy.action, action)

    # later steps, past the 6 in the history, are the same as if the bad states never came
    for _ in range(8):
        np.testing.assert_array_equal(policy.step(**walk), reference.step(**walk))


def test_input_size_check():
    with open(os.path.join(HERE, 'g1.yaml')) as f:
        config = yaml.safe_load(f)
    config['policy_path'] = os.path.join(HERE, 'deploy.onnx')
    # 5 steps of 76, or 6 steps of 74 with a joint missing
    changes = [({'obs_history_len': 5}, 380), ({'policy_joints': config['policy_joints'][:-1]}, 444)]
    for change, given in changes:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'g1.yaml')
            with open(path, 'w') as f:
                yaml.safe_dump(dict(config, **change), f)
            error = raises(ValueError, HomiePolicy, path)
            assert str(error) == f'policy takes 456 inputs, config gives {given}', error


def test_session_one_thread():
    # no thread pool, so no thread pinned to a CPU or spinning (policy_session),
    # and the same actions as onnxruntime's default session
    import onnxruntime as ort
    tasks = '/proc/self/task'
    before = set(os.listdir(tasks)) if os.path.isdir(tasks) else None
    policy = HomiePolicy()
    if before is not None:
        assert set(os.listdir(tasks)) <= before, 'the session started threads'
    options = policy.session.get_session_options()
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (1, 1)
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    for key in ('session.intra_op.allow_spinning', 'session.inter_op.allow_spinning'):
        assert options.get_session_config_entry(key) == '0', key
    default = ort.InferenceSession(policy.config['policy_path'], providers=['CPUExecutionProvider'])
    size = policy.session.get_inputs()[0].shape[-1]
    for x in np.random.default_rng(0).normal(0.0, 1.0, (300, 1, size)).astype(np.float32):
        np.testing.assert_array_equal(policy.session.run(None, {policy.input_name: x})[0],
                                      default.run(None, {policy.input_name: x})[0])


def test_clip_command():
    policy = HomiePolicy()
    cases = [
        ([0.5, -0.2, 0.3], 0.5, [0.5, -0.2, 0.3], 0.5),       # in range: unchanged
        ([2.0, 1.0, 1.5], 0.9, [1.2, 0.5, 0.8], 0.74),        # above
        ([-2.0, -1.0, -1.5], 0.1, [-0.8, -0.5, -0.8], 0.24),  # below
    ]
    for cmd, height, clipped, clipped_height in cases:
        cmd, height = policy.clip_command(cmd, height)
        np.testing.assert_allclose(cmd, clipped, rtol=0, atol=1e-6)
        assert height == clipped_height
    assert raises(ValueError, policy.clip_command, [math.nan, 0.0, 0.0], 0.74)
    assert raises(ValueError, policy.clip_command, [0.0, 0.0, 0.0], math.inf)


def test_gains():
    policy = HomiePolicy()
    kp, kd = policy.gains(['left_hip_roll_joint', 'right_knee_joint', 'left_ankle_pitch_joint',
                           'waist_pitch_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint',
                           'right_wrist_roll_joint', 'left_hand_thumb_0_joint'])
    np.testing.assert_array_equal(kp, [100, 150, 40, 300, 200, 100, 20, 10])
    np.testing.assert_array_equal(kd, [2, 4, 2, 5, 4, 1, 0.5, 2])
    assert raises(KeyError, policy.gains, ['head_joint'])


if __name__ == '__main__':
    tests = [test for name, test in list(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print('ok', test.__name__)
    print(f'{len(tests)} passed')
