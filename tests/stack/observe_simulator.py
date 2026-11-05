"""Run the real sim and record its base pose, joints and support to a private JSONL file.

run.sh runs this in place of the sim's entry point, with the sim's own launcher, arguments
and environment. It wraps the physics step to write each record to $GOLEM_TEST_TELEMETRY,
which only the checker reads: nothing it records reaches ROS.
"""
import json
import os
from pathlib import Path
import runpy
import sys


class Recorder:
    """Write telemetry rows to a line-buffered JSONL file at about 50 Hz of sim time."""

    def __init__(self, filename):
        """Open `filename` for writing."""
        self.stream = open(filename, 'w', buffering=1)
        self.last_time = -1.

    def write(self, stamp, position, quaternion, joints, support):
        """Write one row: sim time (s), base position (m, world), quaternion (WXYZ), joints, support.

        `joints` are the 27 body joints in Unitree wire order; `support` is the band or a fixed
        base. Rows are 50 Hz of sim time, independent of rendering and ROS delivery.
        """
        if stamp - self.last_time < .019:
            return
        self.last_time = stamp
        self.stream.write(json.dumps(dict(sim_time=float(stamp),
            position=list(map(float, position)), quaternion=list(map(float, quaternion)),
            joint_positions=list(map(float, joints)), support_active=bool(support))) + '\n')


def main():
    """Wrap the selected sim's physics step with the recorder and run the sim's entry point."""
    simulator = os.environ['SIMULATOR']
    recorder = Recorder(os.environ['GOLEM_TEST_TELEMETRY'])
    if simulator == 'isaac':
        entry = Path('/home/code/CL_isaaclab_sim/sim_main.py')
        sys.path.insert(0, str(entry.parent))
        import gymnasium as gym
        make = gym.make

        def observed_make(*args, **kwargs):
            """Make the Isaac env and wrap its step to record the robot after each step."""
            wrapper = make(*args, **kwargs)
            env = wrapper.unwrapped
            step = env.step
            from src.python.control.motor_contract import BODY_JOINT_NAMES
            robot = env.scene['robot']
            indices = [robot.joint_names.index(name) for name in BODY_JOINT_NAMES]

            def observed_step(*a, **kw):
                """Step the env and record the robot's root and body joints."""
                result = step(*a, **kw)
                recorder.write(env._sim_step_counter * env.physics_dt,
                    robot.data.root_pos_w[0].tolist(), robot.data.root_quat_w[0].tolist(),
                    robot.data.joint_pos[0, indices].tolist(), robot.is_fixed_base)
                return result

            env.step = observed_step
            return wrapper

        gym.make = observed_make
    elif simulator == 'robocasa':
        entry = Path('/home/code/golem_robocasa/h12_mujoco.py')
        sys.path.insert(0, str(entry.parent))
        import mujoco
        step = mujoco.mj_step

        def observed_step(model, data, *args, **kwargs):
            """Step MuJoCo and, for steps of h12_mujoco.py's main loop, record the pelvis and joints."""
            result = step(model, data, *args, **kwargs)
            # Only h12_mujoco.py's main loop has `resolver` and `band` locals; robosuite's
            # initialisation and settling steps, which reset time, are not recorded.
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
