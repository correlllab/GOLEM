"""
Record golden traces of the four leg policies as today's classes compute them.

Run from the repo root in a disposable golem_ros container (torch and
onnxruntime installed); it writes walk.npz, almi.npz, fame.npz and homie.npz
beside this file:

    python3 core_ws/src/golem_lowerbody_rl/test/golden/record_goldens.py

Each policy runs one scripted 120-step sequence (one step per 20 ms), drawn
from ``numpy.random.default_rng(7)`` afresh for each policy, and is driven as
its node drives it: ``WalkPolicy``, ``AlmiPolicy`` and ``FamePolicy`` from
``golem_lowerbody_rl.policy`` as ``lowerbody_controller_node`` does (reset at
engage and at the band release, then ``compute``), ``HomiePolicy`` from
``g1_homie/policy/homie_policy.py`` as ``g1_homie``'s ``LocomotionController``
does (``load_policy``, ``CommandFilter.update`` with ``g1_homie/config/sim.yaml``'s
``commands`` at 50 Hz, ``reset`` at engage and at the band release, then
``step``). The network (and FAME's encoder) is wrapped to record what it
gets and returns. Torch runs on one thread, so the numbers do not depend on
the host's core count.

Arrays in every ``<policy>.npz`` (N = 120 steps; S = 27 body slots on the
H1-2, 29 on the G1):

- ``t`` (N,) float64, s: the node clock handed to the policy (WALK's and
  ALMI's gait phase); 5.0 at step 0, 0.02 apart.
- ``q``, ``dq`` (N, S) float32, rad and rad/s: every body slot of
  ``rt/lowstate``, slot order; the policy's default pose (unobserved slots 0)
  plus uniform noise of +-0.05 rad, and uniform +-0.5 rad/s.
- ``quat_wxyz`` (N, 4) float32: the IMU orientation handed to the policy, w
  first, a unit quaternion tilted up to 10 deg about a random axis (the sim's
  IMU trim is zero, so it is also the ``rt/lowstate`` quaternion).
- ``gyro`` (N, 3) float32, rad/s: uniform +-0.3.
- ``cmd`` (N, 3), m/s, m/s, rad/s: the velocity command as the node hands it
  on. H1-2: float32, ``/cmd_vel`` clipped to +-1 per axis
  (``LegSupervisor.cmd_vel``). G1: float64, the ``/cmd_vel`` request that goes
  into ``CommandFilter``. The script: 20 steps of zero, a ramp to
  (0.6, 0.0, 0.3) over steps 20-39, a hold to step 69, a ramp down to zero
  over steps 70-79, zero to step 99 (ALMI's effective command falls below
  0.05), a ramp up again over steps 100-119.
- ``height`` (N,) float64, m: the height command as the node hands it on.
  H1-2: the squat command, ``default_height_cmd`` 1.0 throughout (only FAME
  observes it). G1: the ``/locomotion/height`` request into ``CommandFilter``,
  0.74 for steps 0-39 and 0.60 from step 40.
- ``reset`` (N,) bool: the node reset the policy before this step: step 0
  (engage) and step 60 (band release). ALMI's own onset resets are in
  ``onset``, not here.
- ``net_input`` (N, I) float32: what the network got (I = 47 WALK, 65 ALMI,
  252 FAME, 456 HOMIE).
- ``net_output`` (N, 12) float32: what the network returned, before any
  scale or clip.
- ``targets`` (N, 12) float32, rad: the leg targets the class returned, in
  leg slot order.
- ``kp`` (12,), ``kd`` (12,): the leg gains (float32 on the H1-2, float64 on
  the G1).

Per policy:

- ALMI ``onset`` (N,) bool: ``AlmiPolicy.compute`` reset itself at this step
  (the effective command rose through 0.1 while standing); the script makes
  two, on the first and the second ramp up.
- ALMI ``cmd_policy`` (N, 3) float32: the effective command the network
  observed before ``cmd_scale``: ``cmd`` times ``cmd_gain``, clipped to +-1,
  smoothed (``cmd_smooth_alpha``), and at an onset or a reset the clipped
  command itself.
- FAME ``enc_input`` (N, 21) float32: the encoder's e_t; ``z`` (N, 8) float32:
  its latent.
- HOMIE ``cmd_policy`` (N, 3) float64 and ``height_policy`` (N,) float64: what
  ``CommandFilter.update`` returned and ``HomiePolicy.step`` got.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.normpath(os.path.join(HERE, '..', '..'))
G1_HOMIE_DIR = os.path.normpath(os.path.join(PACKAGE_DIR, '..', 'g1_homie'))
HOMIE_POLICY_DIR = os.path.join(G1_HOMIE_DIR, 'policy')

POLICIES = ('walk', 'almi', 'fame', 'homie')
SEED = 7
N_STEPS = 120
DT = 0.02  # s, one 50 Hz policy step
T0 = 5.0  # s of node clock at step 0
ENGAGE_STEP = 0
BAND_RELEASE_STEP = 60
CMD_PEAK = np.array([0.6, 0.0, 0.3])  # vx (m/s), vy (m/s), yaw (rad/s)
CMD_VEL_LIMIT = 1.0  # the H1-2 node's per-axis /cmd_vel clip
H12_HEIGHT = 1.0  # m, the H1-2 node's default_height_cmd
G1_HEIGHTS = (0.74, 0.60)  # m, /locomotion/height before and from step 40
G1_HEIGHT_STEP = 40
MAX_TILT_DEG = 10.0
Q_NOISE = 0.05  # rad
DQ_NOISE = 0.5  # rad/s
GYRO_NOISE = 0.3  # rad/s

H12_SLOTS = 27
G1_SLOTS = 29
# rt/lowstate slots of HOMIE's 27 policy joints (g1_homie.locomotion.POLICY_SLOTS)
G1_POLICY_SLOTS = np.r_[0:13, 15:29]
# rt/lowstate slots of ALMI's 21 observed joints (AlmiPolicy.OBS_JOINT_IDX)
ALMI_SLOTS = np.r_[0:17, 20:24]

INPUT_KEYS = ('t', 'q', 'dq', 'quat_wxyz', 'gyro', 'cmd', 'height', 'reset')
OUTPUT_KEYS = {
    'walk': ('net_input', 'net_output', 'targets', 'kp', 'kd'),
    'almi': ('net_input', 'net_output', 'targets', 'kp', 'kd', 'onset', 'cmd_policy'),
    'fame': ('net_input', 'net_output', 'targets', 'kp', 'kd', 'enc_input', 'z'),
    'homie': ('net_input', 'net_output', 'targets', 'kp', 'kd', 'cmd_policy',
              'height_policy'),
}


def _yaml(path: str) -> dict:
    """Return the YAML file at path."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def h12_config_path(name: str) -> str:
    """Return the path of the H1-2 policy's YAML in this package's policies/."""
    return os.path.join(PACKAGE_DIR, 'policies', name, f'{name}.yaml')


def default_pose(name: str) -> np.ndarray:
    """Return the policy's default pose over every body slot, 0 where it has none (rad)."""
    if name == 'homie':
        pose = np.zeros(G1_SLOTS)
        pose[G1_POLICY_SLOTS[:12]] = _yaml(os.path.join(HOMIE_POLICY_DIR, 'g1.yaml'))[
            'default_angles']
        return pose
    cfg = _yaml(h12_config_path(name))
    pose = np.zeros(H12_SLOTS)
    if name == 'almi':
        pose[ALMI_SLOTS] = cfg['default_angles']
    else:
        pose[:12] = cfg['default_angles']
    if name == 'fame':
        pose[12:] = cfg['default_angles_arms']
    return pose


def command_profile() -> np.ndarray:
    """Return the scripted (vx, vy, yaw) request per step, float64 (module docstring)."""
    scale = np.zeros(N_STEPS)
    scale[20:40] = np.arange(1, 21) / 20.0
    scale[40:70] = 1.0
    scale[70:80] = np.arange(9, -1, -1) / 10.0
    scale[100:120] = np.arange(1, 21) / 20.0
    return scale[:, None] * CMD_PEAK


def make_inputs(name: str) -> dict[str, np.ndarray]:
    """Return the scripted inputs of one policy's trace, drawn from default_rng(SEED)."""
    rng = np.random.default_rng(SEED)
    pose = default_pose(name)
    n_slots = pose.size
    q = pose + rng.uniform(-Q_NOISE, Q_NOISE, (N_STEPS, n_slots))
    dq = rng.uniform(-DQ_NOISE, DQ_NOISE, (N_STEPS, n_slots))
    gyro = rng.uniform(-GYRO_NOISE, GYRO_NOISE, (N_STEPS, 3))
    axis = rng.normal(size=(N_STEPS, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.radians(rng.uniform(0.0, MAX_TILT_DEG, N_STEPS)) / 2.0
    quat = np.concatenate([np.cos(half)[:, None], np.sin(half)[:, None] * axis], axis=1)

    request = command_profile()
    if name == 'homie':
        cmd = request
        height = np.where(np.arange(N_STEPS) < G1_HEIGHT_STEP, *G1_HEIGHTS)
    else:
        cmd = np.clip(request, -CMD_VEL_LIMIT, CMD_VEL_LIMIT).astype(np.float32)
        height = np.full(N_STEPS, H12_HEIGHT)
    reset = np.zeros(N_STEPS, dtype=bool)
    reset[[ENGAGE_STEP, BAND_RELEASE_STEP]] = True
    # rt/lowstate carries float32 joint and IMU values
    return {
        't': T0 + DT * np.arange(N_STEPS),
        'q': q.astype(np.float32),
        'dq': dq.astype(np.float32),
        'quat_wxyz': quat.astype(np.float32),
        'gyro': gyro.astype(np.float32),
        'cmd': cmd,
        'height': height.astype(np.float64),
        'reset': reset,
    }


class TorchRecorder:
    """Stand in for a TorchScript network or encoder: call it and record each input and output."""

    def __init__(self, module):
        """Wrap module (called with a (1, n) tensor)."""
        self._module = module
        self.inputs: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []
        self.memory_resets = 0

    def __call__(self, x):
        """Run the wrapped module on x, recording a copy of x and of the result."""
        y = self._module(x)
        self.inputs.append(x.detach().numpy()[0].copy())
        self.outputs.append(y.detach().numpy()[0].copy())
        return y

    def reset_memory(self) -> None:
        """Clear the wrapped LSTM's recurrent state (ALMI's export) and count the call."""
        self.memory_resets += 1
        self._module.reset_memory()


class OnnxRecorder:
    """Stand in for HOMIE's onnxruntime session: run it and record each input and output."""

    def __init__(self, session):
        """Wrap session."""
        self._session = session
        self.inputs: list[np.ndarray] = []
        self.outputs: list[np.ndarray] = []

    def run(self, output_names, feeds):
        """Run the wrapped session, recording a copy of the one input and the first output."""
        result = self._session.run(output_names, feeds)
        (x,) = feeds.values()
        self.inputs.append(np.asarray(x)[0].copy())
        self.outputs.append(np.asarray(result[0])[0].copy())
        return result


def _run_h12(name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drive today's H1-2 class as lowerbody_controller_node does; return what it computed."""
    import torch

    from golem_lowerbody_rl.policy import AlmiPolicy, FamePolicy, RobotState, WalkPolicy

    torch.set_num_threads(1)
    cls = {'walk': WalkPolicy, 'almi': AlmiPolicy, 'fame': FamePolicy}[name]
    policy = cls(h12_config_path(name))
    net = TorchRecorder(policy._policy)
    policy._policy = net
    encoder = None
    if name == 'fame':
        if not policy.has_encoder:
            raise RuntimeError('FAME encoder weights are missing: z would be zero')
        encoder = TorchRecorder(policy._encoder)
        policy._encoder = encoder

    targets, onset, cmd_policy = [], [], []
    kp = kd = None
    for i in range(N_STEPS):
        state = RobotState(
            q=inputs['q'][i].copy(), dq=inputs['dq'][i].copy(),
            quat=inputs['quat_wxyz'][i].copy(), gyro=inputs['gyro'][i].copy(),
            cmd=np.asarray(inputs['cmd'][i], dtype=np.float32).copy(),
            height_cmd=float(inputs['height'][i]), t=float(inputs['t'][i]))
        if inputs['reset'][i]:
            policy.reset(state)
        memory_resets = net.memory_resets
        leg = policy.compute(state)
        if len(net.inputs) != i + 1:
            raise RuntimeError(f'{name} step {i} called its network {len(net.inputs) - i} times')
        onset.append(net.memory_resets > memory_resets)
        if name == 'almi':
            cmd_policy.append(policy._cmd_filt.copy())
        targets.append(np.asarray(leg.target_q).copy())
        if kp is None:
            kp, kd = np.asarray(leg.kp).copy(), np.asarray(leg.kd).copy()
        elif not (np.array_equal(kp, leg.kp) and np.array_equal(kd, leg.kd)):
            raise RuntimeError(f'{name} changed its gains at step {i}')

    out = {
        'net_input': np.stack(net.inputs),
        'net_output': np.stack(net.outputs),
        'targets': np.stack(targets),
        'kp': kp,
        'kd': kd,
    }
    if name == 'almi':
        out['onset'] = np.array(onset, dtype=bool)
        out['cmd_policy'] = np.stack(cmd_policy)
    if encoder is not None:
        out['enc_input'] = np.stack(encoder.inputs)
        out['z'] = np.stack(encoder.outputs)
    return out


def _load_module(module_name: str, path: str):
    """Import the Python file at path as module_name, without touching sys.path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_homie(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drive today's HomiePolicy as LocomotionController does; return what it computed."""
    locomotion = _load_module(
        'golden_g1_homie_locomotion', os.path.join(G1_HOMIE_DIR, 'g1_homie', 'locomotion.py'))
    commands = _yaml(os.path.join(G1_HOMIE_DIR, 'config', 'sim.yaml'))['commands']
    if locomotion.POLICY_DT != DT:
        raise RuntimeError(f'g1_homie steps every {locomotion.POLICY_DT} s, the trace every {DT}')
    if not np.array_equal(locomotion.POLICY_SLOTS, G1_POLICY_SLOTS):
        raise RuntimeError('g1_homie.locomotion.POLICY_SLOTS changed')
    policy = locomotion.load_policy(HOMIE_POLICY_DIR)
    command_filter = locomotion.CommandFilter(policy, commands)
    net = OnnxRecorder(policy.session)
    policy.session = net
    kp, kd = policy.gains(policy.leg_joints)

    targets, cmd_policy, height_policy = [], [], []
    for i in range(N_STEPS):
        if inputs['reset'][i]:
            policy.reset()
        cmd, height = command_filter.update(inputs['cmd'][i], float(inputs['height'][i]))
        # g1_homie.locomotion.read_state widens rt/lowstate's float32 values to float64
        q = inputs['q'][i].astype(np.float64)[G1_POLICY_SLOTS]
        dq = inputs['dq'][i].astype(np.float64)[G1_POLICY_SLOTS]
        leg = policy.step(cmd, height, inputs['gyro'][i].astype(np.float64),
                          inputs['quat_wxyz'][i].astype(np.float64), q, dq)
        if len(net.inputs) != i + 1:
            raise RuntimeError(f'homie step {i} called its network {len(net.inputs) - i} times')
        targets.append(np.asarray(leg).copy())
        cmd_policy.append(np.asarray(cmd, dtype=np.float64))
        height_policy.append(float(height))

    return {
        'net_input': np.stack(net.inputs),
        'net_output': np.stack(net.outputs),
        'targets': np.stack(targets),
        'kp': kp,
        'kd': kd,
        'cmd_policy': np.stack(cmd_policy),
        'height_policy': np.array(height_policy, dtype=np.float64),
    }


def run_policy(name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Drive a freshly loaded instance of today's class for name through inputs.

    Return the arrays named by OUTPUT_KEYS[name]. Raises RuntimeError when the
    class does not run as the trace assumes (one network call per step,
    constant gains, FAME's encoder present).
    """
    if name == 'homie':
        return _run_homie(inputs)
    return _run_h12(name, inputs)


def record(name: str) -> dict[str, np.ndarray]:
    """Return one policy's golden trace: its scripted inputs and what today's class computed."""
    inputs = make_inputs(name)
    trace = {**inputs, **run_policy(name, inputs)}
    if name == 'almi':
        onsets = np.flatnonzero(trace['onset']).tolist()
        if len(onsets) != 2 or not (20 <= onsets[0] < 40 and 100 <= onsets[1] < 120):
            raise RuntimeError(f'ALMI onsets at steps {onsets}, expected one in each ramp up')
    return trace


def main() -> None:
    """Record every policy's trace and write <policy>.npz beside this file."""
    sys.path.insert(0, PACKAGE_DIR)
    for name in POLICIES:
        trace = record(name)
        path = os.path.join(HERE, f'{name}.npz')
        np.savez_compressed(path, **trace)
        shapes = ', '.join(f'{k} {v.dtype}{list(v.shape)}' for k, v in trace.items())
        print(f'{path}: {shapes}')


if __name__ == '__main__':
    main()
