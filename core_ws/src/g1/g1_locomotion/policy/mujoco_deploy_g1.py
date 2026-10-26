"""HOMIE's G1 walking policy in MuJoCo (sim2sim).

    python mujoco_deploy_g1.py                                   # viewer, standing
    python mujoco_deploy_g1.py --cmd 0.5 0 0                     # walk forward at 0.5 m/s
    python mujoco_deploy_g1.py --headless --duration 10 --cmd 0.5 0 0

Adapted from OpenHomie's MujocoDeploy/mujoco_deploy_g1.py. It runs the ONNX
policy on the leprechaun_g1 G1 model; joints are looked up by name.
"""
import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from homie_policy import HomiePolicy, get_gravity_orientation

HERE = os.path.dirname(os.path.abspath(__file__))


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class Robot:
    """Name-based indices into the MuJoCo model"""

    def __init__(self, m, policy):
        if m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or m.jnt_qposadr[0] != 0:
            raise ValueError('expected the floating base as the first joint')

        # every actuator is a motor on one hinge joint
        joint_ids = m.actuator_trnid[:, 0]
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]
        self.qadr = m.jnt_qposadr[joint_ids]
        self.vadr = m.jnt_dofadr[joint_ids]
        self.kp, self.kd = policy.gains(self.names)

        # policy joints as actuator indices
        index = {n: i for i, n in enumerate(self.names)}
        missing = [n for n in policy.policy_joints if n not in index]
        if missing:
            raise KeyError(f'policy joints not actuated in the model: {missing}')
        self.policy = np.array([index[n] for n in policy.policy_joints])
        self.legs = self.policy[:len(policy.leg_joints)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=os.path.join(HERE, 'g1.yaml'))
    parser.add_argument('--cmd', type=float, nargs=3, metavar=('VX', 'VY', 'YAW'),
                        help='velocity command (default: cmd_init)')
    parser.add_argument('--height', type=float, help='pelvis height command (default: height_cmd)')
    parser.add_argument('--duration', type=float, help='seconds (default: simulation_duration)')
    parser.add_argument('--headless', action='store_true', help='no viewer; run as fast as possible')
    args = parser.parse_args()

    # Load configuration and policy
    policy = HomiePolicy(args.config)
    config = policy.config
    cmd = np.array(args.cmd, dtype=np.float32) if args.cmd else config['cmd_init'].copy()
    height_cmd = config['height_cmd'] if args.height is None else args.height
    duration = config['simulation_duration'] if args.duration is None else args.duration

    # Load robot model
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    m.dof_damping[:] = config['joint_damping']
    m.dof_armature[:] = config['joint_armature']
    m.dof_frictionloss[:] = config['joint_frictionloss']
    robot = Robot(m, policy)

    # Initialize variables
    target_q = np.zeros(m.nu)
    target_q[robot.legs] = policy.default_angles

    def step():
        """One physics step; the policy runs every control_decimation steps"""
        d.ctrl[:] = pd_control(target_q, d.qpos[robot.qadr], robot.kp,
                               0.0, d.qvel[robot.vadr], robot.kd)
        mujoco.mj_step(m, d)

        if counter % config['control_decimation'] == 0:
            # imu_in_pelvis has no rotation relative to the pelvis, the floating base
            target_q[robot.legs] = policy.step(cmd, height_cmd, d.qvel[3:6], d.qpos[3:7],
                                               d.qpos[robot.qadr[robot.policy]],
                                               d.qvel[robot.vadr[robot.policy]])

    print(f'cmd vx {cmd[0]:.2f} vy {cmd[1]:.2f} yaw {cmd[2]:.2f}, height {height_cmd:.2f}, {duration:.0f} s')
    counter = 0
    if args.headless:
        report_every = int(round(1.0 / m.opt.timestep))
        while d.time < duration:
            counter += 1
            step()
            if counter % report_every == 0:
                x, y, z = d.qpos[0:3]
                print(f't {d.time:5.1f}  pelvis x {x:6.2f} y {y:6.2f} z {z:5.3f}')
                # fallen: the pelvis tilted more than 60 deg from upright
                if get_gravity_orientation(d.qpos[3:7])[2] > -0.5:
                    print('fell')
                    return 1
        return 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()
            counter += 1
            step()

            # Sync viewer
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
