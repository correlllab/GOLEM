"""Test-only passive physics recorder. No privileged measurements enter ROS.

Executed in place of the simulator entry point by run.sh, using the same
launcher, arguments, and environment. The real simulation loop still runs.
"""
import json
import os
from pathlib import Path
import runpy
import sys


class Recorder:
    def __init__(self, filename):
        self.stream = open(filename, 'w', buffering=1)
        self.last_time = -1.

    def write(self, stamp, position, quaternion, joints, support):
        # 50 Hz physical measurements, independent of rendering and ROS delivery.
        if stamp - self.last_time < .019:
            return
        self.last_time = stamp
        self.stream.write(json.dumps(dict(sim_time=float(stamp),
            position=list(map(float, position)), quaternion=list(map(float, quaternion)),
            joint_positions=list(map(float, joints)), support_active=bool(support))) + '\n')


def main():
    simulator = os.environ['SIMULATOR']
    recorder = Recorder(os.environ['GOLEM_TEST_TELEMETRY'])
    if simulator == 'isaac':
        entry = Path('/home/code/CL_isaaclab_sim/sim_main.py')
        sys.path.insert(0, str(entry.parent))
        import gymnasium as gym
        make = gym.make

        def observed_make(*args, **kwargs):
            wrapper = make(*args, **kwargs)
            env = wrapper.unwrapped
            step = env.step
            from src.python.control.motor_contract import BODY_JOINT_NAMES
            robot = env.scene['robot']
            indices = [robot.joint_names.index(name) for name in BODY_JOINT_NAMES]

            def observed_step(*a, **kw):
                result = step(*a, **kw)
                recorder.write(env._sim_step_counter * env.physics_dt,
                    robot.data.root_pos_w[0].tolist(), robot.data.root_quat_w[0].tolist(),
                    robot.data.joint_pos[0, indices].tolist(), robot.is_fixed_base)
                return result

            env.step = observed_step
            return wrapper

        gym.make = observed_make
    elif simulator == 'robocasa':
        entry = Path('/home/code/h1_robocasa/h12_mujoco.py')
        sys.path.insert(0, str(entry.parent))
        import mujoco
        step = mujoco.mj_step

        def observed_step(model, data, *args, **kwargs):
            result = step(model, data, *args, **kwargs)
            # Only the production loop owns the resolver and band. Ignore
            # robosuite's initialization/settling calls (which reset time).
            caller = sys._getframe(1)
            local = caller.f_locals
            if caller.f_code.co_filename == str(entry) and 'band' in local:
                resolver = local['resolver']
                body = resolver.body_id('pelvis')
                if body < 0:
                    raise RuntimeError('Test recorder cannot find pelvis body')
                band = local['band']
                recorder.write(data.time, data.xpos[body], data.xquat[body],
                    data.qpos[resolver.motor_qpos], band is not None and band.enabled)
            return result

        mujoco.mj_step = observed_step
    else:
        raise ValueError(f'Unknown simulator: {simulator}')
    sys.argv[0] = str(entry)
    try:
        runpy.run_path(str(entry), run_name='__main__')
    finally:
        recorder.stream.close()


if __name__ == '__main__':
    main()
