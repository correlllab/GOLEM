"""Render a record_simulator.py recording to mp4 with a camera that follows the robot. Runs in the robocasa image.

    python3 render_recording.py <record dir> [--out video.mp4] [--start 0] [--end inf] [--width 960 --height 540]
        [--overlay <commander.jsonl>]

The camera follows the pelvis from above and behind, turning with the robot's heading (azimuth is an offset from
it; --fixed_azimuth holds it instead). The room's walls are not drawn, so the camera can sit outside them. When a
cabinet or counter blocks the line of sight to the robot, the camera swings to the first clear view in VIEWS (other
sides, then steeper), and moves in if none is clear; it returns to the default view once that has been clear for 2 s.
With --overlay, the commander's event log labels each frame with the active event and its command.
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


# views tried when the default is blocked: (azimuth offset, elevation) in degrees, in order of preference
VIEWS = [(0.0, args.elevation), (60.0, args.elevation), (-60.0, args.elevation), (120.0, args.elevation),
         (-120.0, args.elevation), (180.0, args.elevation), (0.0, -65.0), (90.0, -65.0), (-90.0, -65.0), (180.0, -65.0)]
DRAWN = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)   # geom groups the renderer draws by default
robot_root = model.body_rootid[pelvis]
robot_geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == robot_root)
draw_groups = model.geom_group.copy()


def clearance(lookat: np.ndarray, azimuth: float, elevation: float) -> float:
    """Distance from the look-at point toward the camera to the first drawn geom that isn't the robot."""
    az, el = np.radians(azimuth), np.radians(elevation)
    toward_camera = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    hit = mujoco.mj_ray(model, data, lookat, toward_camera, DRAWN, 1, -1, np.zeros(1, np.int32))
    return np.inf if hit < 0 else hit


renderer = mujoco.Renderer(model, args.height, args.width)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
out = args.out or str(rec / 'video.mp4')
writer = imageio.get_writer(out, fps=30, codec='libx264', quality=7, macro_block_size=1)
lookat, heading = None, None
view, shown, distance, default_clear = 0, np.array(VIEWS[0]), args.distance, 0
for row in rows:
    data.qpos[:] = row[1:]
    mujoco.mj_forward(model, data)
    target = data.xpos[pelvis].copy()
    target[2] = max(target[2] - 0.15, 0.5)
    lookat = target if lookat is None else 0.9 * lookat + 0.1 * target   # smooth the follow
    xmat = data.xmat[pelvis].reshape(3, 3)
    yaw = np.arctan2(xmat[1, 0], xmat[0, 0])
    heading = yaw if heading is None else heading + 0.05 * np.angle(np.exp(1j * (yaw - heading)))
    base = (0.0 if args.fixed_azimuth else np.degrees(heading)) + args.azimuth
    model.geom_group[robot_geoms] = 5   # the rays skip the robot
    need = args.distance + 0.2
    default_clear = default_clear + 1 if clearance(lookat, base + VIEWS[0][0], VIEWS[0][1]) >= need else 0
    if clearance(lookat, base + VIEWS[view][0], VIEWS[view][1]) < need or (view and default_clear >= 60):
        room = [clearance(lookat, base + a, e) for a, e in VIEWS]
        view = next((i for i, c in enumerate(room) if c >= need), int(np.argmax(room)))
    step = np.array(VIEWS[view]) - shown
    step[0] = (step[0] + 180.0) % 360.0 - 180.0
    shown = shown + np.clip(step, -6.0, 6.0)   # swing at most 180 deg/s
    room = clearance(lookat, base + shown[0], shown[1])
    model.geom_group[:] = draw_groups
    distance = min(distance + 0.05, args.distance, max(room - 0.25, 0.8))
    camera.lookat[:] = lookat
    camera.azimuth, camera.elevation, camera.distance = base + shown[0], shown[1], distance
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
