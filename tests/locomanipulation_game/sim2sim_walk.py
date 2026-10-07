"""Walking sim-to-sim test: the deploy controller on RoboCasa's own MuJoCo physics, without ROS. Runs in the robocasa image.

The loco-manipulation game walks far slower in RoboCasa than in Isaac Lab. This test runs the same policy export
through locomotion_game_deploy's MarlController (the code marl_deploy_node runs) on the compiled scene a RoboCasa
run recorded (record_simulator.py's model.mjb), with only the robot's and the floor's collisions kept and the floor
widened, so the robot walks a fixed schedule on open floor. Each variant changes one physics or timing setting from
RoboCasa's; whichever brings the walking speed back says what RoboCasa does differently from training.

The loop follows h12_mujoco.py's production loop and the deploy stack:
  every physics step  the elastic band force while it is on; the PD torque kp (q* - q) - kd dq from the latched
                      targets and the current state (unitree_interface.write_ctrl); mj_step; h12_safety_layer's
                      e-stop checks (relax_safety_split, at the physics rate as its 500 Hz poll), counted, not latched
  every 20 ms         the state rt/lowstate carries (q, dq from qpos/qvel; IMU and tau_est from sensordata), one
                      MarlController.step, the targets clipped to the safety layer's clip range and latched
                      --delay_ms later
  start               1 s of pre-pose hold with the band on, engage (controller reset), band release 1 s later
                      (controller reset), as marl_deploy_node does
A fall (pelvis below 0.55 m or tilted past 1 rad) stands the robot up where it is and resumes at the next event.

Variants (join with '+' to combine; each is a change from RoboCasa's settings):
  robocasa        as recorded: 2 ms Euler steps, elliptic cone (impratio 20), contact solref 0.02, every joint damping
                  10, armature 0.1, frictionloss 0.2, explicit PD every step, no added delay
  frictionloss0   robot joints' frictionloss 0
  damping0        robot joints' passive damping 0 (the PD's kd stays)
  armature001     robot joints' armature 0.01 (the Isaac asset's)
  isaac_joints    the three above
  stiff_contact   foot and floor solref 0.004, solimp (0.95, 0.99, 0.001)
  soft_contact    foot and floor solref 0.04
  friction06      foot and floor sliding friction 0.6
  pyramidal       pyramidal friction cone, impratio 1
  condim4         foot and floor contacts with torsional friction (condim 4), at the model's coefficient 0.005 m
  torsion02       condim 4 with torsional friction 0.02 m
  dt5ms           5 ms physics steps (Isaac's), PD still recomputed every step
  implicit_pd     the 27 motors as MuJoCo position servos (kp, kd) under the implicitfast integrator: the PD is
                  integrated implicitly, as PhysX drives integrate it
  gravcomp0       no gravity compensation on the hands and grippers
  delay<N>ms      targets latched N ms after the state they were computed from

Writes <out>/<variant>/: telemetry.jsonl and qpos.f32 (record_simulator.py's format, plus feet and contact), model.mjb
(render_recording.py replays it), commander.jsonl (events in the game commander's format) and segments.json; then
<out>/report.md and <out>/metrics.json compare the variants. --replay runs a RoboCasa run's events instead of the
schedule; --score only scores recordings already in this format (Locomanipulation_game's eval_sim2sim_walk.py
writes Isaac's), with numpy alone.

    python3 sim2sim_walk.py --model <RoboCasa result>/record/model.mjb --out <dir> [--variants robocasa,damping0,...]
    python3 sim2sim_walk.py --score <recording dir>... --out <dir>
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

DEPLOY_PKG = Path(os.environ.get('LMG_DEPLOY_PKG', '/home/code/locomotion_game_deploy'))
SAFETY_PKG = Path(os.environ.get('LMG_SAFETY_PKG', '/home/code/h12_safety_layer'))
SIM_DIR = Path(os.environ.get('LMG_SIM_DIR', '/home/code/h1_robocasa'))

DEFAULT_SCHEDULE = ('stand:2,walk:8:0.5:0:0,stand:2,walk:6:-0.4:0:0,stand:2,walk:6:0:0.25:0,stand:2,walk:6:0:-0.25:0,'
                    'stand:2,walk:5:0:0:0.4,stand:2,walk:8:0.3:0:0.3,stand:2,walk:6:0.7:0:0,stand:2,'
                    'reach:standing,reach:crouch,reach:crouch,reach:crouch,stand:2,walk:6:0.5:0:0,stand:2')
FALL_HEIGHT = 0.55
FALL_TILT = 1.0
TELEMETRY_DT = 0.02
FRAME_DT = 1.0 / 30.0
REACH_TOL = 0.05        # m, both wrists, held REACH_HOLD s: the commander's reached
REACH_HOLD = 1.0
ARM_GOAL_S = 4.0
SETTLE_TO_ARM_S = 0.75

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--model', default=None, help="a RoboCasa run's record/model.mjb")
parser.add_argument('--policy', default=str(DEPLOY_PKG / 'policies/marl_golem/marl_golem.yaml'))
parser.add_argument('--goals', default='arm_goals_7_10.npz', help='arm-goal samples, next to --policy')
parser.add_argument('--urdf', default='/home/code/CL_Assets/ros_assets/h1_2_magpie.urdf')
parser.add_argument('--safety', default=str(SAFETY_PKG / 'config/relax_safety_split.yaml'))
parser.add_argument('--variants', default='robocasa')
parser.add_argument('--schedule', default=DEFAULT_SCHEDULE,
                    help='comma list: stand:S | walk:S:vx:vy:wz | reach:standing|crouch')
parser.add_argument('--seed', type=int, default=0, help='picks the arm goals')
parser.add_argument('--replay', default=None, help="a RoboCasa run's ros/commander.jsonl: replay its events instead of --schedule")
parser.add_argument('--score', nargs='+', default=None, metavar='DIR',
                    help='only score recordings already written in this format (any sim) into --out')
parser.add_argument('--out', required=True)
parser.add_argument('--jobs', type=int, default=8)


# -- the variants ------------------------------------------------------------------------------------------------
def _variant_settings(name):
    settings = {'delay_ms': 0.0, 'mods': []}
    for part in name.split('+'):
        if part == 'robocasa':
            continue
        if part.startswith('delay') and part.endswith('ms'):
            settings['delay_ms'] = float(part[5:-2])
        elif part == 'isaac_joints':
            settings['mods'] += ['frictionloss0', 'damping0', 'armature001']
        elif part in MODS:
            settings['mods'].append(part)
        else:
            raise ValueError(f'unknown variant {part!r}')
    return settings


def _set_contact(m, ctx, solref=None, solimp=None, friction=None, condim=None, torsion=None):
    for g in list(ctx['robot_geoms']) + [ctx['floor']]:
        if solref is not None:
            m.geom_solref[g] = solref
        if solimp is not None:
            m.geom_solimp[g, :3] = solimp
        if friction is not None:
            m.geom_friction[g, 0] = friction
        if condim is not None:
            m.geom_condim[g] = condim
        if torsion is not None:
            m.geom_friction[g, 1] = torsion


def _implicit_pd(m, ctx):
    import mujoco
    for i, a in enumerate(ctx['resolver'].motor_ctrl):
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_gainprm[a, 0] = ctx['kp'][i]
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_biasprm[a, 1] = -ctx['kp'][i]
        m.actuator_biasprm[a, 2] = -ctx['kd'][i]
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    ctx['implicit_pd'] = True


MODS = {
    'frictionloss0': lambda m, ctx: m.dof_frictionloss.__setitem__(ctx['motor_dofs'], 0.0),
    'damping0': lambda m, ctx: m.dof_damping.__setitem__(ctx['motor_dofs'], 0.0),
    'armature001': lambda m, ctx: m.dof_armature.__setitem__(ctx['motor_dofs'], 0.01),
    'stiff_contact': lambda m, ctx: _set_contact(m, ctx, solref=(0.004, 1.0), solimp=(0.95, 0.99, 0.001)),
    'soft_contact': lambda m, ctx: _set_contact(m, ctx, solref=(0.04, 1.0)),
    'friction06': lambda m, ctx: _set_contact(m, ctx, friction=0.6),
    'pyramidal': lambda m, ctx: (setattr(m.opt, 'cone', 0), setattr(m.opt, 'impratio', 1.0)),
    'condim4': lambda m, ctx: _set_contact(m, ctx, condim=4),
    'torsion02': lambda m, ctx: _set_contact(m, ctx, condim=4, torsion=0.02),
    'dt5ms': lambda m, ctx: setattr(m.opt, 'timestep', 0.005),
    'implicit_pd': _implicit_pd,
    'gravcomp0': lambda m, ctx: m.body_gravcomp.__setitem__(ctx['robot_bodies'], 0.0),
}


# -- the scene ----------------------------------------------------------------------------------------------------
def _name(m, objtype, i):
    import mujoco
    return mujoco.mj_id2name(m, objtype, i) or ''


def open_floor(m, d, resolver):
    """Keep only the robot's and the room floor's collisions; widen the floor; hide everything else in renders."""
    import mujoco
    pelvis = resolver.body_id('pelvis')
    root = m.body_rootid[pelvis]
    robot_bodies = np.flatnonzero(m.body_rootid == root)
    robot_geoms = np.flatnonzero(m.body_rootid[m.geom_bodyid] == root)
    mujoco.mj_forward(m, d)
    floors = [g for g in range(m.ngeom) if g not in set(robot_geoms) and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
              and _name(m, mujoco.mjtObj.mjOBJ_GEOM, g).startswith('floor_room') and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if not floors:
        raise RuntimeError('no floor_room collision box in the model')
    floor = floors[0]
    robot = set(robot_geoms.tolist())
    for g in range(m.ngeom):
        if g in robot or g == floor:
            continue
        m.geom_contype[g] = 0
        m.geom_conaffinity[g] = 0
        m.geom_rgba[g, 3] = 0.0
    m.geom_size[floor, :2] = 60.0
    m.geom_rbound[floor] = float(np.linalg.norm(m.geom_size[floor]))
    if hasattr(m, 'geom_aabb'):
        m.geom_aabb[floor, 3:] = m.geom_size[floor]
    m.geom_rgba[floor] = (0.55, 0.55, 0.52, 1.0)
    m.geom_group[floor] = 1
    mujoco.mj_forward(m, d)
    top = float(d.geom_xpos[floor][2] + m.geom_size[floor, 2])
    return {'robot_bodies': robot_bodies, 'robot_geoms': robot_geoms, 'floor': floor, 'floor_top': top,
            'floor_xy': d.geom_xpos[floor][:2].copy()}


def _free_joint(m, body):
    import mujoco
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == body and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return j
    raise RuntimeError('no free joint on the pelvis')


def _lowest_point(m, d, geoms):
    import mujoco
    low = math.inf
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            low = min(low, float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()))
        else:
            low = min(low, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
    return low


# -- the schedule ---------------------------------------------------------------------------------------------------
def build_events(args, policy_dir):
    """The event list: --replay's RoboCasa run, or --schedule with arm goals drawn from --goals by --seed."""
    if args.replay:
        recs = [json.loads(line) for line in open(args.replay)]
        evs = [r for r in recs if r['type'] == 'event']
        end = next((r['t'] for r in recs if r['type'] in ('estop', 'done')), None)
        events = []
        for i, r in enumerate(evs):
            # the realized length: obstacle stops end walks early
            stop = evs[i + 1]['t'] if i + 1 < len(evs) else (end if end is not None else r['t'] + r['duration'])
            length = max(0.0, stop - r['t'] - (r.get('settle', 0.0) if r['kind'] == 'arm_goal' else 0.0))
            if r['kind'] == 'walk':
                events.append({'kind': 'walk', 'cmd': [float(v) for v in r['cmd']], 'duration': round(length, 3)})
            else:
                events.append({'kind': 'arm_goal', 'goal': r['goal'], 'low': r['low'], 'needs_crouch': r['needs_crouch'],
                               'poses': r['poses'], 'duration': round(length, 3)})
        return events
    goals = np.load(Path(policy_dir) / args.goals)
    rng = np.random.default_rng(args.seed)
    standing_pool = np.flatnonzero(~goals['low'] & (goals['poses'][:, :, 0].min(axis=1) > 0.25))
    crouch_pool = np.flatnonzero(goals['low'] & goals['needs_crouch'])
    events = []
    for item in args.schedule.split(','):
        p = item.split(':')
        if p[0] == 'stand':
            events.append({'kind': 'walk', 'cmd': [0.0, 0.0, 0.0], 'duration': float(p[1])})
        elif p[0] == 'walk':
            events.append({'kind': 'walk', 'cmd': [float(v) for v in p[2:5]], 'duration': float(p[1])})
        elif p[0] == 'reach':
            idx = int(rng.choice(standing_pool if p[1] == 'standing' else crouch_pool))
            events.append({'kind': 'arm_goal', 'goal': idx, 'low': bool(goals['low'][idx]),
                           'needs_crouch': bool(goals['needs_crouch'][idx]), 'poses': goals['poses'][idx].astype(float).tolist(),
                           'duration': ARM_GOAL_S})
        else:
            raise ValueError(f'schedule item {item!r}')
    return events


# -- one variant --------------------------------------------------------------------------------------------------
def run_variant(args, variant, out):
    import mujoco
    import torch
    import yaml

    torch.set_num_threads(1)
    sys.path.insert(0, str(SIM_DIR))
    sys.path.insert(0, str(DEPLOY_PKG))
    sys.path.insert(0, str(SAFETY_PKG))
    from sim_names import NameResolver
    from locomotion_game_deploy.controller import ARM_START, MarlController, RobotState
    from locomotion_game_deploy.kinematics import ArmKinematics
    from locomotion_game_deploy.networks import TorchScriptNetwork
    from h12_safety_layer.core.config import load_config
    from h12_safety_layer.core.joint_limits import JOINT_NAMES

    out.mkdir(parents=True, exist_ok=True)
    settings = _variant_settings(variant)
    m = mujoco.MjModel.from_binary_path(args.model)
    d = mujoco.MjData(m)
    resolver = NameResolver(m)

    # the controller, as marl_deploy_node builds it
    cfg_path = Path(args.policy)
    cfg = yaml.safe_load(open(cfg_path))
    export = cfg_path.parent / 'export.yaml'
    if export.exists():
        cfg.update(yaml.safe_load(open(export)) or {})
    legs_odom = cfg.get('legs_odometry', {}) if cfg.get('odometry') == 'legs' else {}
    extra = list(legs_odom.get('foot_frames', []))
    if legs_odom.get('imu_frame', 'pelvis') != 'pelvis':
        extra.append(legs_odom['imu_frame'])
    names = cfg['joint_names']
    kin = ArmKinematics(args.urdf, joint_names=names, ee_frames=cfg['arms']['ee_frames'],
                        arm_joint_names=[names[ARM_START:ARM_START + 7], names[ARM_START + 7:]], extra_frames=extra)
    load = lambda key, n_in, n_out: TorchScriptNetwork(str(cfg_path.parent / key), n_in, n_out)
    controller = MarlController(cfg, kin, load(cfg['legs']['policy'], cfg['legs']['num_obs'], cfg['legs']['num_actions']),
                                load(cfg['arms']['policy'], cfg['arms']['num_obs'], cfg['arms']['num_actions']),
                                load(cfg['estimator']['path'], cfg['estimator']['num_inputs'], 6))
    kp, kd, default = controller.kp, controller.kd, controller.default
    limits = load_config(args.safety)['limits']
    assert list(JOINT_NAMES) == list(names), 'safety layer and policy joint orders differ'

    # the scene and the variant
    ctx = open_floor(m, d, resolver)
    ctx.update(resolver=resolver, kp=kp, kd=kd, implicit_pd=False, motor_dofs=resolver.motor_qvel)
    for mod in settings['mods']:
        MODS[mod](m, ctx)
    dt = float(m.opt.timestep)
    ctrl_every = int(round(cfg['control_dt'] / dt))
    if abs(ctrl_every * dt - cfg['control_dt']) > 1e-9:
        raise ValueError(f'control_dt {cfg["control_dt"]} is not a multiple of the physics step {dt}')
    delay_steps = int(round(settings['delay_ms'] / 1000.0 / dt))

    pelvis = resolver.body_id('pelvis')
    torso = resolver.body_id('torso_link')
    free = _free_joint(m, pelvis)
    qadr, vadr = m.jnt_qposadr[free], m.jnt_dofadr[free]
    feet = [resolver.body_id(f) for f in ('left_ankle_roll_link', 'right_ankle_roll_link')]
    foot_geoms = [[g for g in ctx['robot_geoms'] if m.geom_bodyid[g] == b and (m.geom_contype[g] or m.geom_conaffinity[g])]
                  for b in feet]
    qa, ga, aa = (resolver.sensor_adr[k][0] for k in ('imu_quat', 'imu_gyro', 'imu_acc'))
    floor = ctx['floor']

    def robot_state():
        sd = d.sensordata
        return RobotState(q=d.qpos[resolver.motor_qpos].copy(), dq=d.qvel[resolver.motor_qvel].copy(),
                          tau=sd[resolver.motor_tau].copy(), quat=sd[qa:qa + 4].copy(), gyro=sd[ga:ga + 3].copy(),
                          acc=sd[aa:aa + 3].copy())

    def stand(x, y, yaw):
        d.qvel[:] = 0.0
        d.qpos[qadr:qadr + 3] = (x, y, 1.0)
        d.qpos[qadr + 3:qadr + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        d.qpos[resolver.motor_qpos] = default
        mujoco.mj_forward(m, d)
        d.qpos[qadr + 2] += ctx['floor_top'] + 0.002 - _lowest_point(m, d, [g for gs in foot_geoms for g in gs])
        mujoco.mj_forward(m, d)
        return d.xpos[torso].copy() + np.array([0.0, 0.0, 0.01])

    def contact_forces():
        force = np.zeros(2)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for side, geoms in enumerate(foot_geoms):
                if (c.geom1 in geoms and c.geom2 == floor) or (c.geom2 in geoms and c.geom1 == floor):
                    mujoco.mj_contactForce(m, d, i, f6)
                    force[side] += f6[0]
        return force

    # the schedule
    events = build_events(args, cfg_path.parent)

    log = open(out / 'commander.jsonl', 'w', buffering=1)
    tele = open(out / 'telemetry.jsonl', 'w', buffering=1)
    frames = open(out / 'qpos.f32', 'wb')
    write = lambda rec: log.write(json.dumps(rec) + '\n')

    # run
    mujoco.mj_resetData(m, d)
    anchor = stand(*ctx['floor_xy'], 0.0)
    mujoco.mj_saveModel(m, str(out / 'model.mjb'), None)
    (out / 'model.json').write_text(json.dumps({'nq': int(m.nq), 'timestep': dt}))
    s = {'band': True, 'latched': default.copy(), 'phase': 'prepose', 'phase_t': 0.0, 'ev_i': -1, 'ev': None,
         'ev_start': 0.0, 'reach_since': None, 'reached': False, 'best': math.inf, 'cmd': np.zeros(3), 'mode': ''}
    pending = deque()
    estop_active = np.zeros((3, 27), dtype=bool)
    estops, falls = [], []
    next_tele, next_frame = 0.0, 0.0
    step, wall0 = 0, time.time()
    total_s = sum(e['duration'] + SETTLE_TO_ARM_S for e in events) + 2.0

    def finish_goal():
        ev = s['ev']
        write({'t': round(d.time, 3), 'type': 'goal_result', 'index': s['ev_i'], 'goal': ev['goal'], 'low': ev['low'],
               'needs_crouch': ev['needs_crouch'], 'reached': s['reached'],
               'best_error': None if math.isinf(s['best']) else round(s['best'], 4)})

    def start_event(i):
        """The commander's event switch: a goal replacing a goal needs no end call; a walk after a goal does."""
        prev = s['ev']
        nxt = events[i] if i < len(events) else None
        if prev is not None and prev['kind'] == 'arm_goal':
            finish_goal()
            if nxt is None or nxt['kind'] != 'arm_goal':
                controller.end_arm_goal()
        s['ev_i'], s['ev_start'], s['ev'] = i, d.time, None
        if nxt is None:
            return
        if nxt['kind'] == 'walk':
            s['cmd'] = np.array(nxt['cmd'])
            controller.set_velocity_command(s['cmd'])
            s['ev'] = nxt
            write({'t': round(d.time, 3), 'type': 'event', 'kind': 'walk', 'index': i, 'cmd': nxt['cmd'], 'duration': nxt['duration']})
        else:
            s['cmd'] = np.zeros(3)
            controller.set_velocity_command(s['cmd'])
            settle = 0.0 if controller.mode.value == 'arm_goal' else SETTLE_TO_ARM_S
            s['ev'] = {**nxt, 'settle': settle}
            controller.request_arm_goal(np.asarray(nxt['poses']))
            s['reach_since'], s['reached'], s['best'] = None, False, math.inf
            write({'t': round(d.time, 3), 'type': 'event', 'kind': 'arm_goal', 'index': i, 'goal': nxt['goal'],
                   'low': nxt['low'], 'needs_crouch': nxt['needs_crouch'], 'poses': nxt['poses'],
                   'duration': nxt['duration'], 'settle': settle})

    def policy_tick(t):
        state = robot_state()
        target = default.copy()
        if s['phase'] == 'prepose' and t >= s['phase_t'] + 1.0:
            controller.reset(state)
            s['phase'], s['phase_t'] = 'engaged', t
            write({'t': round(t, 3), 'type': 'deploy_state', 'state': 'engaged'})
        elif s['phase'] == 'engaged' and t >= s['phase_t'] + 1.0:
            s['band'] = False
            controller.reset(state)
            s['phase'] = 'free'
            write({'t': round(t, 3), 'type': 'deploy_state', 'state': 'band released'})
            start_event(s['ev_i'] + 1)
        if s['phase'] != 'prepose':
            target = controller.step(state).q
            if s['phase'] == 'free' and controller.mode.value != s['mode']:
                s['mode'] = controller.mode.value
                write({'t': round(t, 3), 'type': 'deploy_state', 'state': s['mode']})
        pending.append((step + delay_steps, np.clip(target, limits['q_clip_limits'][:, 0], limits['q_clip_limits'][:, 1])))
        ev = s['ev']
        if s['phase'] != 'free' or ev is None:
            return
        if ev['kind'] == 'arm_goal' and controller.mode.value == 'arm_goal':
            err = float(controller.wrist_errors().max())
            s['best'] = min(s['best'], err)
            if err < REACH_TOL:
                s['reach_since'] = t if s['reach_since'] is None else s['reach_since']
                s['reached'] = s['reached'] or t - s['reach_since'] >= REACH_HOLD
            else:
                s['reach_since'] = None
        if t - s['ev_start'] >= ev['duration'] + ev.get('settle', 0.0):
            start_event(s['ev_i'] + 1)

    def physics_step():
        nonlocal step
        while pending and pending[0][0] <= step:
            s['latched'] = pending.popleft()[1]
        d.xfrc_applied[torso, :3] = 0.0
        if s['band']:
            x, v = d.xpos[torso], d.qvel[vadr:vadr + 3]
            disp = np.linalg.norm(x - anchor)
            if disp > 1e-9:
                direction = (x - anchor) / disp
                d.xfrc_applied[torso, :3] = (-1e4 * disp - 1e3 * float(np.dot(v, direction))) * direction
        q, dq = d.qpos[resolver.motor_qpos], d.qvel[resolver.motor_qvel]
        d.ctrl[resolver.motor_ctrl] = s['latched'] if ctx['implicit_pd'] else kp * (s['latched'] - q) - kd * dq
        mujoco.mj_step(m, d)
        step += 1

    while not (s['phase'] == 'free' and s['ev'] is None and s['ev_i'] >= len(events)):
        if step % ctrl_every == 0:
            policy_tick(d.time)
        physics_step()
        free = s['phase'] == 'free'
        # e-stop checks on the new state (tau_est: the applied torque, clamped, as rt/lowstate carries it)
        q, dq, tau = d.qpos[resolver.motor_qpos], d.qvel[resolver.motor_qvel], d.sensordata[resolver.motor_tau]
        trip = np.stack([(q < limits['q_estop_limits'][:, 0]) | (q > limits['q_estop_limits'][:, 1]),
                         np.abs(dq) > limits['dq_estop_limits'], np.abs(tau) > limits['tau_estop_limits']]) & free
        for cause, j in zip(*np.nonzero(trip & ~estop_active)):
            estops.append({'t': round(d.time, 3), 'cause': ('position', 'velocity', 'torque')[cause], 'joint': names[j],
                           'value': round(float((q, dq, tau)[cause][j]), 4), 'event': s['ev_i'], 'mode': controller.mode.value})
        estop_active = trip
        if d.time >= next_tele:
            tele.write(json.dumps({
                'sim_time': round(float(d.time), 4),
                'pelvis_pos': [round(float(v), 5) for v in d.xpos[pelvis]],
                'pelvis_quat': [round(float(v), 5) for v in d.xquat[pelvis]],
                'q': [round(float(v), 5) for v in q], 'qd': [round(float(v), 4) for v in dq],
                'tau': [round(float(v), 3) for v in tau], 'band': s['band'],
                'feet': [[round(float(v), 5) for v in d.xpos[b]] for b in feet],
                'foot_force': [round(float(f), 1) for f in contact_forces()],
                'cmd': s['cmd'].round(3).tolist(), 'mode': controller.mode.value if free else s['phase'], 'event': s['ev_i'],
            }) + '\n')
            next_tele += TELEMETRY_DT
        if d.time >= next_frame:
            frames.write(np.concatenate([[d.time], d.qpos]).astype(np.float32).tobytes())
            next_frame += FRAME_DT
        # a fall: stand up where it is (pre-pose, engage, band release again), then the next event
        up = d.xmat[pelvis].reshape(3, 3)[:, 2]
        if free and (d.xpos[pelvis][2] < FALL_HEIGHT or math.acos(max(-1.0, min(1.0, up[2]))) > FALL_TILT):
            ev = s['ev']
            falls.append({'t': round(d.time, 3), 'event': s['ev_i'], 'mode': controller.mode.value,
                          'kind': ev['kind'] if ev else None, 'cmd': s['cmd'].round(3).tolist()})
            write({'t': round(d.time, 3), 'type': 'fall', 'event': s['ev_i']})
            if ev is not None and ev['kind'] == 'arm_goal':
                finish_goal()
            s['ev'] = None
            fwd = d.xmat[pelvis].reshape(3, 3)[:, 0]
            anchor = stand(d.xpos[pelvis][0], d.xpos[pelvis][1], math.atan2(fwd[1], fwd[0]))
            pending.clear()
            s.update(band=True, latched=default.copy(), phase='prepose', phase_t=d.time, cmd=np.zeros(3), mode='')
            controller.set_velocity_command(np.zeros(3))
            controller.reset(None)
        if d.time > total_s + 60.0 * (1 + len(falls)):
            raise RuntimeError('schedule overran')

    for f in (log, tele, frames):
        f.close()
    segments = score(out)
    result = {'variant': variant, 'settings': settings, 'timestep': dt, 'sim_s': round(d.time, 2),
              'wall_s': round(time.time() - wall0, 1), 'falls': falls, 'estops': estops, 'segments': segments}
    json.dump(result, open(out / 'segments.json', 'w'), indent=1)
    return result


# -- scoring -----------------------------------------------------------------------------------------------------
def _rotate_inverse(qw, v):
    w, xyz = qw[:, :1], -qw[:, 1:]
    t2 = 2.0 * np.cross(xyz, v)
    return v + w * t2 + np.cross(xyz, t2)


def score(out):
    """Per event in <out>/commander.jsonl, from <out>/telemetry.jsonl: velocity tracking (each walk after its first
    1.5 s), gait and reach. Needs only numpy, so it also scores recordings from other sims in the same format."""
    out = Path(out)
    tele = [json.loads(line) for line in open(out / 'telemetry.jsonl')]
    recs = [json.loads(line) for line in open(out / 'commander.jsonl')]
    t = np.array([r['sim_time'] for r in tele])
    pos = np.array([r['pelvis_pos'] for r in tele])
    quat = np.array([r['pelvis_quat'] for r in tele])
    feet = np.array([r['feet'] for r in tele])          # (T, 2, 3) ankle_roll_link origins, world
    force = np.array([r['foot_force'] for r in tele])   # (T, 2) floor normal force per foot, N
    ev_idx = np.array([r['event'] for r in tele])
    free = np.array([r['mode'] not in ('prepose', 'engaged') for r in tele])
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(0.25 / dt)))                   # 0.5 s central differences
    vel_w = np.zeros_like(pos)
    vel_w[k:-k] = (pos[2 * k:] - pos[:-2 * k]) / (t[2 * k:] - t[:-2 * k])[:, None]
    vel_b = _rotate_inverse(quat, vel_w)
    yaw = np.unwrap(np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)))
    yaw_rate = np.zeros_like(yaw)
    yaw_rate[k:-k] = (yaw[2 * k:] - yaw[:-2 * k]) / (t[2 * k:] - t[:-2 * k])
    foot_v = np.zeros(feet.shape[:2])
    foot_v[1:] = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) / np.diff(t)[:, None]
    loaded = force > 200.0                              # ~30% of the robot's weight
    events = {r['index']: r for r in recs if r['type'] == 'event'}
    results = {r['index']: r for r in recs if r['type'] == 'goal_result'}
    fell = {r['event'] for r in recs if r['type'] == 'fall'}
    # foot height at rest: both feet loaded, standing still
    still = free & loaded.all(axis=1) & (np.linalg.norm(vel_w[:, :2], axis=1) < 0.02)
    rest_z = np.median(feet[still, :, 2], axis=0) if still.any() else feet[free, :, 2].min(axis=0)
    first_free = t[np.flatnonzero(free)[0]] if free.any() else 0.0
    standing_z = float(np.median(pos[free & (t < first_free + 2.0), 2])) if free.any() else math.nan
    segs = []
    for i in sorted(events):
        ev = events[i]
        m_all = (ev_idx == i) & free
        cmd = np.array(ev.get('cmd', [0.0, 0.0, 0.0]))
        is_stand = ev['kind'] == 'walk' and not np.any(np.abs(cmd) > 1e-6)
        seg = {'index': i, 'kind': 'stand' if is_stand else ev['kind'], 'fell': i in fell}
        if ev['kind'] == 'walk':
            seg['cmd'] = cmd.round(3).tolist()
            m = m_all & (t >= ev['t'] + 1.5)
            if m.sum() < max(25, int(1.0 / dt)):
                segs.append(seg)
                continue
            speed = float(np.linalg.norm(cmd[:2]))
            seg['seconds'] = round(float(m.sum() * dt), 2)
            seg['v_body'] = vel_b[m, :2].mean(axis=0).round(3).tolist()
            seg['xy_error'] = round(float(np.linalg.norm(vel_b[m, :2] - cmd[:2], axis=1).mean()), 3)
            seg['yaw_rate'] = round(float(yaw_rate[m].mean()), 3)
            if speed > 0.05:
                along = float((vel_b[m, :2] @ (cmd[:2] / speed)).mean())
                seg['v_along'] = round(along, 3)
                seg['path_ratio'] = round(along / speed, 3)
            # gait: lift above the foot's rest height, unloaded share, cadence from the left-right load swing
            lift = feet[m, :, 2] - rest_z
            seg['lift_p90_m'] = round(float(np.percentile(lift, 90, axis=0).mean()), 3)
            seg['unloaded'] = round(float((force[m] < 20.0).mean()), 3)
            seg['double_loaded'] = round(float(loaded[m].all(axis=1).mean()), 3)
            swing = force[m, 0] - force[m, 1]
            swing = swing - swing.mean()
            freq = np.fft.rfftfreq(len(swing), dt)
            band = (freq > 0.4) & (freq < 4.0)
            if band.any():
                f_gait = float(freq[band][np.argmax(np.abs(np.fft.rfft(swing))[band])])
                seg['steps_per_s'] = round(2.0 * f_gait, 2)
                seg['stride_m'] = round(float(foot_v[m].mean()) / f_gait, 3)
            slip = foot_v[m][loaded[m]]
            seg['stance_slip_m_s'] = round(float(np.median(slip)), 3) if slip.size else None
        elif ev['kind'] == 'arm_goal':
            res = results.get(i, {})
            seg.update(goal=ev['goal'], low=ev['low'], needs_crouch=ev['needs_crouch'], reached=res.get('reached'),
                       best_error=res.get('best_error'))
            seg['pelvis_drop'] = round(standing_z - float(pos[m_all, 2].min()), 3) if m_all.any() else None
        segs.append(seg)
    return segs


# -- the comparison ------------------------------------------------------------------------------------------------
def summarize(results):
    rows = []
    mean = lambda xs: round(float(np.mean(xs)), 3) if xs else None
    for r in results:
        walks = [x for x in r['segments'] if x['kind'] == 'walk' and 'path_ratio' in x]
        fwd = [x for x in walks if x['cmd'][0] > 0.05 and abs(x['cmd'][1]) < 0.05 and abs(x['cmd'][2]) < 0.05]
        turns = [x for x in r['segments'] if x['kind'] == 'walk' and abs(x.get('cmd', [0, 0, 0])[2]) > 0.1 and 'yaw_rate' in x]
        reaches = [x for x in r['segments'] if x['kind'] == 'arm_goal']
        crouch = [x for x in reaches if x['needs_crouch']]
        pick = lambda segs, key: mean([x[key] for x in segs if x.get(key) is not None])
        rows.append({
            'variant': r['variant'],
            'path_ratio': pick(walks, 'path_ratio'), 'fwd_path_ratio': pick(fwd, 'path_ratio'),
            'fwd_steps_per_s': pick(fwd, 'steps_per_s'), 'fwd_stride_m': pick(fwd, 'stride_m'),
            'fwd_lift_m': pick(fwd, 'lift_p90_m'), 'fwd_unloaded': pick(fwd, 'unloaded'),
            'fwd_slip_m_s': pick(fwd, 'stance_slip_m_s'),
            'turn_ratio': mean([x['yaw_rate'] / x['cmd'][2] for x in turns]),
            'falls': len(r['falls']), 'estops': len(r['estops']),
            'reached': f"{sum(bool(x['reached']) for x in reaches)}/{len(reaches)}",
            'crouch_reached': f"{sum(bool(x['reached']) for x in crouch)}/{len(crouch)}",
            'crouch_drop_m': pick(crouch, 'pelvis_drop'),
        })
    return rows


def write_report(out, args, results, rows):
    lines = ['# Walking sim-to-sim test', '',
             f"Model `{args.model}`, policy `{args.policy}`, " +
             (f"events replayed from `{args.replay}`." if args.replay else f"schedule `{args.schedule}`."), '',
             '- path ratio: mean body-frame speed along the command over |command|, each walk after its first 1.5 s;',
             '- forward columns: the straight-ahead walks;',
             '- steps/s: both feet, from the left-right load swing; stride: one foot\'s travel per gait cycle;',
             '- lift: 90th percentile of the ankle above its rest height; unloaded: share of time a foot carries < 20 N;',
             '- slip: median horizontal speed of a foot carrying > 200 N.', '',
             '| variant | path ratio | fwd path ratio | steps/s | stride m | lift m | unloaded | slip m/s | turn ratio | falls | e-stop onsets | reached | crouch reached | crouch drop m |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['path_ratio']} | {r['fwd_path_ratio']} | {r['fwd_steps_per_s']} | {r['fwd_stride_m']} | "
                     f"{r['fwd_lift_m']} | {r['fwd_unloaded']} | {r['fwd_slip_m_s']} | {r['turn_ratio']} | {r['falls']} | "
                     f"{r['estops']} | {r['reached']} | {r['crouch_reached']} | {r['crouch_drop_m']} |")
    lines += ['', '## E-stop onsets (relax_safety_split at every physics step, counted, not latched)', '']
    for r in results:
        if r['estops']:
            causes = {}
            for e in r['estops']:
                key = f"{e['joint']} {e['cause']}"
                causes[key] = causes.get(key, 0) + 1
            lines.append(f"- {r['variant']}: " + ', '.join(f'{k} x{v}' for k, v in sorted(causes.items(), key=lambda kv: -kv[1])))
    lines += ['', '## Per walk', '', '| variant | cmd | v body | path ratio | steps/s | stride m | lift m | unloaded | slip m/s | fell |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        for x in r['segments']:
            if x['kind'] == 'walk' and 'seconds' in x:
                lines.append(f"| {r['variant']} | {x['cmd']} | {x.get('v_body')} | {x.get('path_ratio')} | {x.get('steps_per_s')} | "
                             f"{x.get('stride_m')} | {x.get('lift_p90_m')} | {x.get('unloaded')} | {x.get('stance_slip_m_s')} | {x['fell']} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')


def _job(packed):
    args, variant, out = packed
    try:
        return run_variant(args, variant, Path(out) / variant)
    except Exception as exc:  # report the variant, keep the others
        import traceback
        traceback.print_exc()
        return {'variant': variant, 'error': repr(exc)}


def main():
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.score:
        results = []
        for directory in args.score:
            meta = json.load(open(Path(directory) / 'segments.json')) if (Path(directory) / 'segments.json').exists() else {}
            results.append({'variant': meta.get('variant', Path(directory).name), 'falls': meta.get('falls', []),
                            'estops': meta.get('estops', []), 'segments': score(directory)})
        failed = []
    else:
        if not args.model:
            parser.error('--model is required unless --score is given')
        variants = [v for v in args.variants.split(',') if v]
        for v in variants:
            _variant_settings(v)
        with mp.get_context('spawn').Pool(min(args.jobs, len(variants))) as pool:
            results = pool.map(_job, [(args, v, str(out)) for v in variants])
        failed = [r for r in results if 'error' in r]
        results = [r for r in results if 'error' not in r]
    rows = summarize(results)
    json.dump({'args': vars(args), 'summary': rows, 'failed': failed}, open(out / 'metrics.json', 'w'), indent=1)
    write_report(out, args, results, rows)
    print((out / 'report.md').read_text())
    for f in failed:
        print(f"FAILED {f['variant']}: {f['error']}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
