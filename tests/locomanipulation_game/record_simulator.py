"""Passive RoboCasa recorder for the loco-manipulation game runs. Nothing it measures enters ROS.

Executed in place of h12_mujoco.py by run.sh, with the canonical launcher, arguments and environment; the real
simulation loop runs unchanged (the tests/stack/observe_simulator.py pattern). It writes, into $GOLEM_RECORD_DIR:
  model.mjb         the compiled scene, saved at the first production step (render_recording.py replays it)
  qpos.f32          float32 rows [sim_time, qpos...] at 30 Hz of sim time, for the video
  telemetry.jsonl   50 Hz: sim time, pelvis position and wxyz quaternion, the 27 motor q, qd and applied
                    torque, band state
"""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

ENTRY = Path('/home/code/h1_robocasa/h12_mujoco.py')
FRAME_DT = 1.0 / 30.0
TELEMETRY_DT = 0.02


class Recorder:
    def __init__(self, directory: Path):
        self.dir = directory
        self.qpos = open(directory / 'qpos.f32', 'wb', buffering=0)
        self.telemetry = open(directory / 'telemetry.jsonl', 'w', buffering=1)
        self.next_frame = 0.0
        self.next_telemetry = 0.0
        self.saved_model = False

    def step(self, model, data, resolver, band) -> None:
        import mujoco

        if not self.saved_model:
            mujoco.mj_saveModel(model, str(self.dir / 'model.mjb'), None)
            (self.dir / 'model.json').write_text(json.dumps({'nq': int(model.nq), 'timestep': float(model.opt.timestep)}))
            self.saved_model = True
            self.next_frame = self.next_telemetry = data.time
        if data.time >= self.next_frame:
            self.qpos.write(np.concatenate([[data.time], data.qpos]).astype(np.float32).tobytes())
            self.next_frame += FRAME_DT
        if data.time >= self.next_telemetry:
            pelvis = resolver.body_id('pelvis')
            self.telemetry.write(json.dumps({
                'sim_time': round(float(data.time), 4),
                'pelvis_pos': [round(float(v), 5) for v in data.xpos[pelvis]],
                'pelvis_quat': [round(float(v), 5) for v in data.xquat[pelvis]],
                'q': [round(float(v), 5) for v in data.qpos[resolver.motor_qpos]],
                'qd': [round(float(v), 4) for v in data.qvel[resolver.motor_qvel]],
                'tau': [round(float(v), 3) for v in data.sensordata[resolver.motor_tau]],
                'band': bool(band is not None and band.enabled),
            }) + '\n')
            self.next_telemetry += TELEMETRY_DT


def main():
    recorder = Recorder(Path(os.environ['GOLEM_RECORD_DIR']))
    sys.path.insert(0, str(ENTRY.parent))
    import mujoco

    step = mujoco.mj_step

    def recorded_step(model, data, *args, **kwargs):
        result = step(model, data, *args, **kwargs)
        # only the production loop holds the resolver and band; robosuite's settling calls reset time
        caller = sys._getframe(1)
        local = caller.f_locals
        if caller.f_code.co_filename == str(ENTRY) and 'band' in local:
            recorder.step(model, data, local['resolver'], local['band'])
        return result

    mujoco.mj_step = recorded_step
    sys.argv[0] = str(ENTRY)
    try:
        runpy.run_path(str(ENTRY), run_name='__main__')
    finally:
        recorder.qpos.close()
        recorder.telemetry.close()


if __name__ == '__main__':
    main()
