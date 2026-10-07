"""Render a record_simulator.py recording to mp4 with a camera that follows the robot. Runs in the robocasa image.

    python3 render_recording.py <record dir> [--out video.mp4] [--start 0] [--end inf] [--width 960 --height 540]
        [--overlay <commander.jsonl>]

The camera follows the pelvis from above and behind, turning with the robot's heading (azimuth is an offset from
it; --fixed_azimuth holds it instead). The room's walls are not drawn, so the camera can sit outside them. With --overlay, the commander's event log
labels each frame with the active event and its command.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('record')
parser.add_argument('--out', default=None)
parser.add_argument('--start', type=float, default=0.0, help='sim seconds')
parser.add_argument('--end', type=float, default=float('inf'))
parser.add_argument('--width', type=int, default=960)
parser.add_argument('--height', type=int, default=540)
parser.add_argument('--distance', type=float, default=2.8)
parser.add_argument('--azimuth', type=float, default=25.0, help='deg from the robot heading (0: straight behind)')
parser.add_argument('--fixed_azimuth', action='store_true', help='azimuth in world degrees instead')
parser.add_argument('--elevation', type=float, default=-35.0)
parser.add_argument('--overlay', default=None)
args = parser.parse_args()

rec = Path(args.record)
model = mujoco.MjModel.from_binary_path(str(rec / 'model.mjb'))
model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
# the room's walls would block a follow camera: draw them in a hidden geom group (the floor stays)
for i in range(model.ngeom):
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]) or ''
    if body.startswith('wall'):
        model.geom_group[i] = 4
data = mujoco.MjData(model)
rows = np.fromfile(rec / 'qpos.f32', dtype=np.float32)
rows = rows[: len(rows) // (model.nq + 1) * (model.nq + 1)].reshape(-1, model.nq + 1)
rows = rows[(rows[:, 0] >= args.start) & (rows[:, 0] <= args.end)]
pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
if pelvis < 0:   # robosuite prefixes the robot's bodies (robot0_pelvis)
    pelvis = next(i for i in range(model.nbody) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i).endswith('pelvis'))
print('following body', mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, pelvis))

events = []
if args.overlay and Path(args.overlay).exists():
    for line in open(args.overlay):
        r = json.loads(line)
        if r.get('type') in ('event', 'estop', 'obstacle_stop', 'done'):
            events.append(r)

label_font = None
try:
    from PIL import Image, ImageDraw, ImageFont
    label_font = ImageFont.load_default()
except ImportError:
    Image = None


def label(t: float) -> str:
    current = None
    for r in events:
        if r['t'] <= t:
            current = r
    if current is None:
        return f't {t:6.1f} s'
    if current['type'] == 'event' and current['kind'] == 'walk':
        return f"t {t:6.1f} s  walk  cmd vx {current['cmd'][0]:+.2f} vy {current['cmd'][1]:+.2f} wz {current['cmd'][2]:+.2f}"
    if current['type'] == 'event':
        kind = 'crouch reach' if current.get('needs_crouch') else ('low reach' if current.get('low') else 'standing reach')
        return f't {t:6.1f} s  arm goal ({kind})'
    return f"t {t:6.1f} s  {current['type']}"


renderer = mujoco.Renderer(model, args.height, args.width)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
out = args.out or str(rec / 'video.mp4')
writer = imageio.get_writer(out, fps=30, codec='libx264', quality=7, macro_block_size=1)
lookat, heading = None, None
for row in rows:
    data.qpos[:] = row[1:]
    mujoco.mj_forward(model, data)
    target = data.xpos[pelvis].copy()
    target[2] = max(target[2] - 0.15, 0.5)
    lookat = target if lookat is None else 0.9 * lookat + 0.1 * target   # smooth the follow
    xmat = data.xmat[pelvis].reshape(3, 3)
    yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
    heading = yaw if heading is None else heading + 0.05 * np.angle(np.exp(1j * (yaw - heading)))
    camera.lookat[:] = lookat
    if not args.fixed_azimuth:
        camera.azimuth = np.degrees(heading) + args.azimuth
    renderer.update_scene(data, camera)
    frame = renderer.render()
    if Image is not None and events is not None:
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, args.width, 22], fill=(0, 0, 0))
        draw.text((8, 5), label(float(row[0])), fill=(255, 255, 255), font=label_font)
        frame = np.asarray(img)
    writer.append_data(frame)
writer.close()
print(f'wrote {out}: {len(rows)} frames, sim {rows[0, 0]:.1f}-{rows[-1, 0]:.1f} s')
