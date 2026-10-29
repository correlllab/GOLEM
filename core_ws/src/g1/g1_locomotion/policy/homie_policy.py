"""HOMIE's G1 walking policy: its observation, history and action.

    from homie_policy import HomiePolicy

    policy = HomiePolicy()                          # g1.yaml next to this file
    kp, kd = policy.gains(joint_names)              # PD gains for any joints
    cmd, height = policy.clip_command(cmd, height)  # to the trained ranges
    targets = policy.step(cmd, height, gyro, quat_wxyz, q, dq)  # at 50 Hz

q and dq are the positions and velocities of policy.policy_joints, in that
order; the targets are for policy.leg_joints. Quaternions are w, x, y, z. It
needs only numpy, onnxruntime and pyyaml, so simulators and ROS nodes can
share it.
"""
import collections
import os

import numpy as np
import onnxruntime as ort
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path):
    """Load the YAML config; paths in it are relative to the file.

    In the containers this directory is /home/code/g1_locomotion, beside
    /home/code/CL_Assets; in the repo it is core_ws/src/g1/g1_locomotion/policy,
    so a path that is not there is looked up from the repo's root as well.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base = os.path.dirname(os.path.abspath(config_path))
    for path_key in ['policy_path', 'xml_path']:
        path = os.path.normpath(os.path.join(base, config[path_key]))
        in_repo = os.path.normpath(os.path.join(base, '..', '..', '..', '..', config[path_key]))
        config[path_key] = path if os.path.exists(path) or not os.path.exists(in_repo) else in_repo

    array_keys = ['default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)

    return config


def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    q_conj = np.array([w, -x, -y, -z])

    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),

        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),

        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])


def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)


def policy_session(policy_path):
    """The policy's onnxruntime session: one thread, no spinning, no pinning

    onnxruntime's default session makes a thread pool as wide as the machine
    (23 threads on a 24-core laptop), pins each thread to one CPU, the same
    CPUs in every process, and lets them spin between runs. At 50 Hz two of
    them each kept a CPU 100% busy (CPUs 1 and 8 on the laptop) in every
    process that ran the policy. With two processes running it on one host,
    the second one's pool threads on those two CPUs got no CPU time for
    minutes, and it never finished loading. The policy is a
    small MLP: on the calling thread a step takes about 0.2 ms, with the same
    actions. With one intra-op thread there is no pool, so nothing is pinned
    or spins; the spinning entries also cover a pool if a count changes.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(policy_path, sess_options=options, providers=['CPUExecutionProvider'])


def joint_gains(name, gains):
    """Gain for a joint, matched by substring as in legged_gym"""
    for key, value in gains.items():
        if key in name:
            return value
    raise KeyError(f'no gain matches joint {name}')


class HomiePolicy:
    """The ONNX policy with its observation history and last action"""

    def __init__(self, config_path=os.path.join(HERE, 'g1.yaml')):
        self.config = config = load_config(config_path)
        self.policy_joints = config['policy_joints']
        self.leg_joints = self.policy_joints[:config['num_actions']]
        self.default_angles = config['default_angles']
        # default positions of the policy joints: the legs' default angles, 0 for the rest
        self.default_q = np.zeros(len(self.policy_joints), dtype=np.float32)
        self.default_q[:len(self.leg_joints)] = self.default_angles

        self.session = policy_session(config['policy_path'])
        policy_input = self.session.get_inputs()[0]
        self.input_name = policy_input.name
        # cmd, height, angular velocity, gravity, joint positions and velocities, last action
        self.num_obs = 3 + 1 + 3 + 3 + 2 * len(self.policy_joints) + len(self.leg_joints)
        num_inputs = self.num_obs * config['obs_history_len']
        if policy_input.shape[-1] != num_inputs:
            raise ValueError(f'policy takes {policy_input.shape[-1]} inputs, config gives {num_inputs}')
        self.reset()

    def gains(self, names):
        """kp and kd for each joint name, matched by substring as in training"""
        kp = np.array([joint_gains(n, self.config['stiffness']) for n in names], dtype=np.float64)
        kd = np.array([joint_gains(n, self.config['damping']) for n in names], dtype=np.float64)
        return kp, kd

    def clip_command(self, cmd, height):
        """The command and height target clipped to the trained ranges; ValueError if not finite"""
        cmd = np.asarray(cmd, dtype=np.float32)
        if not (np.all(np.isfinite(cmd)) and np.isfinite(height)):
            raise ValueError(f'command is not finite: {cmd}, height {height}')
        low, high = np.array(self.config['cmd_range'], dtype=np.float32).T
        low_height, high_height = self.config['height_range']
        return np.clip(cmd, low, high), min(max(float(height), low_height), high_height)

    def reset(self):
        """Zero the observation history and the last action"""
        history_len = self.config['obs_history_len']
        self.action = np.zeros(len(self.leg_joints), dtype=np.float32)
        self.history = collections.deque(
            [np.zeros(self.num_obs, dtype=np.float32)] * history_len, maxlen=history_len)

    def observation(self, cmd, height, gyro, quat_wxyz, q, dq):
        """One step of the policy's observation, as in HOMIE's training env"""
        # the state in float64, as MuJoCo gives it, whatever dtype the caller passes;
        # the command in float32, like cmd_scale
        config = self.config
        return np.concatenate([
            np.asarray(cmd, dtype=np.float32) * config['cmd_scale'],
            [height],
            np.asarray(gyro, dtype=np.float64) * config['ang_vel_scale'],
            get_gravity_orientation(np.asarray(quat_wxyz, dtype=np.float64)),
            (np.asarray(q, dtype=np.float64) - self.default_q) * config['dof_pos_scale'],
            np.asarray(dq, dtype=np.float64) * config['dof_vel_scale'],
            self.action,
        ]).astype(np.float32)

    def step(self, cmd, height, gyro, quat_wxyz, q, dq):
        """Run the policy on the current state; returns the leg position targets

        cmd is vx, vy (m/s) and the yaw rate (rad/s), height the pelvis height
        target (m), gyro the pelvis angular velocity in the pelvis frame (rad/s),
        quat_wxyz the pelvis orientation, and q, dq the positions and velocities
        of policy_joints. Call it at 50 Hz, the rate the policy was trained at.
        If the observation has the wrong size or a value that is not finite, it
        raises ValueError and leaves the history and the last action unchanged.
        """
        # check before changing any state: one NaN kept in the history would
        # reach every later output through the last action
        current = self.observation(cmd, height, gyro, quat_wxyz, q, dq)
        if current.shape != (self.num_obs,):
            raise ValueError(f'observation has {current.size} values, expected {self.num_obs}')
        not_finite = np.flatnonzero(~np.isfinite(current)).tolist()
        if not_finite:
            raise ValueError(f'observation is not finite at {not_finite}')
        self.history.append(current)
        obs = np.concatenate(self.history)[None, :]
        action = self.session.run(None, {self.input_name: obs})[0][0]
        self.action = np.clip(action, -self.config['clip_actions'], self.config['clip_actions'])
        return self.action * self.config['action_scale'] + self.default_angles
