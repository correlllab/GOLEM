"""Offline recording/evaluation tests. Never initialize ROS, DDS or controllers."""

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from video_recording import RetentionTracker, VideoRecorder, recording_fps


class RetentionTests(unittest.TestCase):
    def test_physical_clearance_contact_retention_and_lift_required(self):
        tracker = RetentionTracker()
        tracker.observe(0, -0.022, [0, 0, 0.18], 1.0, False, False)
        for i in range(1, 23):
            row = tracker.observe(i * 0.05, 0.011, [0, 0, 0.18], 1.04, False, True)
        self.assertTrue(row['physical_extraction_candidate'])
        for clearance, relative, welded, contact in (
                (0.009, [0, 0, 0.18], False, True),
                (0.02, [0, 0, 0.4], False, True),
                (0.02, [0, 0, 0.18], True, True),
                (0.02, [0, 0, 0.18], False, False)):
            row = tracker.observe(1.15, clearance, relative, 1.04, welded, contact)
            self.assertFalse(row['physical_extraction_candidate'])

    def test_relative_slip_and_missing_samples_do_not_prove_retention(self):
        tracker = RetentionTracker()
        tracker.observe(0, -0.02, [0, 0, 0.18], 1, False, False)
        for i in range(1, 23):
            row = tracker.observe(i * 0.05, 0.02, [i * 0.001, 0, 0.18], 1.04, False, True)
        self.assertFalse(row['physical_extraction_candidate'])
        row = tracker.observe(2, 0.02, [0, 0, 0.18], 1.04, False, True)
        self.assertEqual(row['retained_sim_seconds'], 0)

    def test_previous_weld_cannot_prove_physical_grasp(self):
        tracker = RetentionTracker()
        tracker.observe(0, -0.02, [0, 0, 0.18], 1, True, True)
        for i in range(1, 24):
            row = tracker.observe(i * 0.05, 0.02, [0, 0, 0.18], 1.04, False, True)
        self.assertFalse(row['physical_extraction_candidate'])
        self.assertTrue(row['peg_weld_seen'])

    def test_invalid_fps(self):
        for value in (0, -1, 61, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                recording_fps(value)


def toy_scene():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="960" offheight="720"/></visual>
      <worldbody><light pos="0 0 2"/>
        <body name="pegboard_board"><geom type="box" size=".16 .125 .025" group="1"/>
          <site name="pegboard_hole_cylinder_1_entry" pos="0 0 .025"/></body>
        <body name="peg_cylinder_peg" pos="0 0 .003"><freejoint/>
          <geom type="cylinder" size=".005 .03" pos="0 0 .03" group="1"/>
          <site name="peg_cylinder_tip"/></body>
        <body name="robot0_left_wrist_yaw_link" pos="0 .1 .2"/>
        <body name="robot0_right_wrist_yaw_link" pos="0 -.1 .2"/>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = SimpleNamespace(peg_prefix='peg_cylinder_', peg_id='cylinder_1',
                             source_hole='cylinder_1',
                             peg_variant={'body': 'peg', 'tip_site': 'tip'},
                             source_hole_spec={'entry_site': 'hole_cylinder_1_entry'})
    return model, data, SimpleNamespace(scene_config=config)


class SnapshotTests(unittest.TestCase):
    def test_busy_worker_never_blocks_capture(self):
        recorder = VideoRecorder.__new__(VideoRecorder)
        recorder.error = None
        recorder.fps = 20
        recorder.origin = recorder.next_t = None
        recorder.last_t = None
        recorder.free = queue.Queue()
        recorder.dropped = 0
        recorder.capture(SimpleNamespace(time=0.0), None)
        self.assertEqual(recorder.dropped, 1)
        self.assertAlmostEqual(recorder.next_t, 0.05)

    @unittest.skipUnless(os.environ.get('GOLEM_TEST_VIDEO_RENDER') == '1' and shutil.which('ffmpeg'),
                         'opt-in real GL/ffmpeg smoke test')
    def test_real_renderer_webm_jsonl_and_live_state_unmodified(self):
        model, data, env = toy_scene()
        before = {name: getattr(data, name).copy() for name in
                  ('qpos', 'qvel', 'ctrl', 'xfrc_applied')}
        model_before = {name: getattr(model, name).copy() for name in
                        ('geom_friction', 'geom_contype', 'geom_conaffinity')}
        with tempfile.TemporaryDirectory(dir='/tmp/kilo') as directory:
            path = Path(directory) / 'capture.webm'
            recorder = VideoRecorder(model, env, path)
            try:
                for t in (0, 0.05, 0.2):
                    data.time = t
                    recorder.capture(data, SimpleNamespace(enabled=True))
                    deadline = time.monotonic() + 5
                    while recorder.free.qsize() < 3 and not recorder.error:
                        if time.monotonic() > deadline:
                            self.fail('renderer did not consume snapshot')
                        time.sleep(0.01)
            finally:
                recorder.close()
            self.assertIsNone(recorder.error)
            info = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
            stream = info['streams'][0]
            self.assertEqual((stream['codec_name'], stream['width'], stream['height']),
                             ('vp9', 960, 720))
            rows = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
            frames = [r for r in rows if 'frame' in r]
            self.assertEqual(len(frames), 5)
            self.assertEqual([r['repeated_image'] for r in frames], [False, False, True, True, False])
            self.assertEqual([r['video_time'] for r in frames], [0, 0.05, 0.1, 0.15, 0.2])
            self.assertAlmostEqual(frames[0]['tip_entry_clearance_m'], -0.022)
            self.assertEqual(frames[0]['peg_tip_site'], 'peg_cylinder_tip')
            self.assertTrue(all(r['support_band_enabled'] for r in frames))
            self.assertEqual(rows[-1]['event'], 'recording_end')
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(data, name), value)
        for name, value in model_before.items():
            np.testing.assert_array_equal(getattr(model, name), value)


if __name__ == '__main__':
    unittest.main()
