"""Play one scripted scenario to one safety layer and record everything it writes on rt/lowcmd.

Usage (golem_ros container, core_ws sourced):
  python3 driver.py <old|new> <scenario> <domain> <out dir> [<old checkout>]

Starts the layer as a subprocess on <domain> (never 0), waits SETTLE s, then plays the
scenario on Unitree DDS: rt/lowstate at 500 Hz, the safety inputs at 200 Hz and the
hardware estop at 50 Hz, in the form each layer reads (old: h12/estop_status_raw; new:
the ROS Bools rt/estop/triggered and rt/estop/plugged_in). Both layers see the same
values. Writes <out>/<layer>_<scenario>.npz (every frame received, with its time from
the start of the scenario; negative while settling) and <layer>_<scenario>.log (the
layer's own output). Exit code 0 unless the layer could not be started.
"""
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                         ChannelSubscriber)
from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowCmd_ as LowCmdDefault,
                                        unitree_hg_msg_dds__LowState_ as LowStateDefault)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

LAYER, SCENARIO, DOMAIN, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
OLD_CHECKOUT = sys.argv[5] if len(sys.argv) > 5 else '/tmp/safety_diff/h12_safety_layer'
# SAFETY: domain 0 is the real robots' command bus; this harness never writes there.
if DOMAIN == 0:
    sys.exit('driver refuses domain 0')
SETTLE = 2.5  # s for the layer to start before the scenario's t = 0
N = 27  # H1-2 body slots
SLOTS = 35  # motor slots on the wire
DUMP = '/tmp/gs/install/golem_safety/lib/golem_safety/golem_safety_dump_config'


@dataclass
class RosBool(IdlStruct, typename='std_msgs::msg::dds_::Bool_'):
    """std_msgs/Bool as ROS 2 puts it on DDS."""

    data: bool


@dataclass
class EStopStatus(IdlStruct, typename='h12.estop.EstopStatusRaw'):
    """The old layer's raw estop sample (h12_safety_layer.core.estop_subscriber)."""

    triggered: bool
    plugged_in: bool
    sequence_id: int
    stamp_ms: int


def limits(config):
    """Return golem_safety's derived limits of an H1-2 config (the old ones are equal, limits.json)."""
    out = subprocess.run([DUMP, 'h12', config], capture_output=True, text=True, check=True)
    return {k: np.asarray(v, float) for k, v in json.loads(out.stdout)['limits'].items()}


def valid_cmd(q=0.1, kp=50.0, kd=2.0):
    """Return per-slot (mode, q, dq, tau, kp, kd) arrays of an in-limit command."""
    return dict(mode=np.ones(N), q=np.full(N, q), dq=np.zeros(N), tau=np.zeros(N),
                kp=np.full(N, kp), kd=np.full(N, kd))


# Each scenario: config, mode, estop use, duration (s) and a function of t (s) returning
# {'state': {...overrides} or None, 'lower'/'upper'/'full': cmd dict or None, 'extra': ...}.
def scenarios():
    """Return the scenario table."""
    lim = limits('default_safety_split')
    f32 = np.float32

    def over_cmd():
        """Return a command past every clip limit, with header and spare-slot payloads."""
        c = valid_cmd()
        c['q'] = np.where(np.arange(N) % 2 == 0, lim['q_clip_high'] + 0.5, lim['q_clip_low'] - 0.5)
        c['dq'] = np.full(N, 100.0)
        c['tau'] = np.full(N, -1000.0)
        c['kp'] = np.where(np.arange(N) % 2 == 0, 1000.0, -5.0)
        c['kd'] = np.where(np.arange(N) % 2 == 0, 100.0, -1.0)
        c['header'] = (7, 1)
        c['spare'] = 9.0
        return c

    def edge_cmd():
        """Return q at the float32 clip limits and one float32 step past them."""
        c = valid_cmd()
        hi = lim['q_clip_high'].astype(f32)
        c['q'] = np.where(np.arange(N) % 2 == 0, hi, np.nextafter(hi, f32(np.inf)))
        c['dq'] = lim['dq_clip'].astype(f32)
        c['kp'] = np.full(N, 500.0)
        return c

    def both(c):
        """Send c on both split inputs."""
        return {'lower': c, 'upper': c}

    def q_at(joint, value):
        """Return a state override putting one joint's q at value."""
        q = np.zeros(N)
        q[joint] = value
        return {'q': q}

    knee_hi = lim['q_estop_high'][3]
    s = {
        'startup_idle': ('default_safety_split', False, 1.0, lambda t: {}),
        'relay_clip_split': ('default_safety_split', False, 1.0,
                             lambda t: both(over_cmd() if t < 0.5 else edge_cmd())),
        'relay_clip_full': ('default_safety_full', False, 1.0, lambda t: {'full': over_cmd()}),
        'nan_unowned_slot': ('default_safety_split', False, 1.0, lambda t: {
            'lower': dict(valid_cmd(), nan_at=20) if t >= 0.5 else valid_cmd(),
            'upper': valid_cmd()}),
        'upper_silent': ('default_safety_split', False, 3.5, lambda t: {
            'lower': valid_cmd(0.05), 'upper': valid_cmd(0.2) if t < 0.5 else None}),
        'lower_silent': ('default_safety_split', False, 1.5, lambda t: {
            'lower': valid_cmd(0.05) if t < 0.5 else None, 'upper': valid_cmd(0.2)}),
        'estop_q_edge': ('default_safety_split', False, 2.0, lambda t: dict(
            both(valid_cmd()),
            state=q_at(3, 0.0 if t < 0.5 else f32(knee_hi) if t < 1.0
                       else knee_hi + 0.01))),
        'estop_q_below': ('default_safety_split', False, 1.5, lambda t: dict(
            both(valid_cmd()), state=q_at(3, np.nextafter(f32(knee_hi), f32(0)) if t >= 0.5
                                          else 0.0))),
        'estop_dq': ('default_safety_split', False, 1.5, lambda t: dict(
            both(valid_cmd()), state={'dq': np.where(np.arange(N) == 16, 50.0, 0.0)}
            if t >= 0.5 else None)),
        'state_nan': ('default_safety_split', False, 1.5, lambda t: dict(
            both(valid_cmd()), state={'tau': np.where(np.arange(N) == 5, np.nan, 0.0)}
            if t >= 0.5 else None)),
        'engage_mm0': ('default_safety_split', False, 2.0, lambda t: dict(
            both(valid_cmd()), state={'mode_machine': 0 if t < 1.0 else 6})),
        'foreign_mm': ('default_safety_split', False, 2.0, lambda t: dict(
            both(valid_cmd()), state={'mode_machine': 6 if t < 1.0 else 5})),
        'second_writer': ('default_safety_split', False, 1.5, lambda t: dict(
            both(valid_cmd()), second_lower=valid_cmd(0.3) if t >= 0.8 else None)),
        'hw_estop_silent': ('default_safety_split', True, 2.0, lambda t: dict(
            both(valid_cmd()), estop=(False, True) if 0.3 <= t < 1.0 else None)),
        'hw_estop_trigger': ('default_safety_split', True, 2.0, lambda t: dict(
            both(valid_cmd()), estop=(t >= 1.0, True) if t >= 0.3 else None)),
    }
    return s


def fill_cmd(msg, c):
    """Write command dict c into a LowCmd_."""
    mm, pr = c.get('header', (6, 0))
    msg.mode_machine, msg.mode_pr = mm, pr
    for i in range(SLOTS):
        m = msg.motor_cmd[i]
        if i < N:
            m.mode = int(c['mode'][i])
            m.q, m.dq, m.tau = float(c['q'][i]), float(c['dq'][i]), float(c['tau'][i])
            m.kp, m.kd = float(c['kp'][i]), float(c['kd'][i])
        else:
            v = c.get('spare', 0.0)
            m.mode, m.q, m.dq, m.tau, m.kp, m.kd = int(v), v, v, v, v, v
        m.reserve = int(c.get('spare', 0))
    if 'nan_at' in c:
        msg.motor_cmd[c['nan_at']].q = float('nan')
    msg.reserve = [int(c.get('spare', 0))] * 4
    msg.crc = 0


def start_layer(config, use_estop, log):
    """Start the layer under test on DOMAIN; return its Popen."""
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN))
    if LAYER == 'old':
        cmd = ['python3', '-u', '/tmp/safety_diff/old_runner.py', OLD_CHECKOUT,
               f'{OLD_CHECKOUT}/config/{config}.yaml', '1' if use_estop else '0']
    else:
        cmd = ['/tmp/gs/install/golem_safety/lib/golem_safety/safety_node', '--ros-args', '-p', 'robot:=h12',
               '-p', f'config:={config}', '-p', 'sim:=true',
               '-p', f'use_estop:={"true" if use_estop else "false"}']
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def main():
    """Run the scenario and save the recording."""
    config, use_estop, duration, script = scenarios()[SCENARIO]
    OUT.mkdir(parents=True, exist_ok=True)
    ChannelFactoryInitialize(DOMAIN)
    frames, lock = [], threading.Lock()
    t0 = [time.monotonic() + 1e9]

    def on_cmd(msg):
        """Record one rt/lowcmd frame."""
        row = np.array([[m.mode, m.q, m.dq, m.tau, m.kp, m.kd] for m in msg.motor_cmd])
        with lock:
            frames.append((time.monotonic() - t0[0], msg.mode_machine, msg.mode_pr,
                           row, list(msg.reserve), msg.crc))

    sub = ChannelSubscriber('rt/lowcmd', LowCmd_)
    sub.Init(on_cmd, 100)
    pubs = {k: ChannelPublisher(f'rt/safety/{t}', LowCmd_) for k, t in
            (('lower', 'lowcmd_lower_in'), ('upper', 'lowcmd_upper_in'), ('full', 'lowcmd_in'))}
    second = ChannelPublisher('rt/safety/lowcmd_lower_in', LowCmd_)
    state_pub = ChannelPublisher('rt/lowstate', LowState_)
    for p in (*pubs.values(), second, state_pub):
        p.Init()
    dp = DomainParticipant(DOMAIN)
    trig_w = DataWriter(dp, Topic(dp, 'rt/estop/triggered', RosBool))
    plug_w = DataWriter(dp, Topic(dp, 'rt/estop/plugged_in', RosBool))
    raw_w = DataWriter(dp, Topic(dp, 'h12/estop_status_raw', EStopStatus))

    log = open(OUT / f'{LAYER}_{SCENARIO}.log', 'w')
    proc = start_layer(config, use_estop, log)
    start = time.monotonic()
    t0[0] = start + SETTLE
    end = t0[0] + duration
    tick, seq, k = 0, 0, 0
    state, cmd = LowStateDefault(), LowCmdDefault()
    while (now := time.monotonic()) < end:
        t = now - t0[0]
        step = script(max(t, 0.0)) if t >= 0 else {}
        ov = step.get('state') or {}
        state.mode_machine = int(ov.get('mode_machine', 6))
        tick += 2
        state.tick = tick
        for i in range(N):
            ms = state.motor_state[i]
            ms.q = float(ov['q'][i]) if 'q' in ov else 0.0
            ms.dq = float(ov['dq'][i]) if 'dq' in ov else 0.0
            ms.tau_est = float(ov['tau'][i]) if 'tau' in ov else 0.0
        state_pub.Write(state)
        if k % 5 == 0:  # 200 Hz commands
            for key, pub in pubs.items():
                if step.get(key) is not None:
                    fill_cmd(cmd, step[key])
                    pub.Write(cmd)
            if step.get('second_lower') is not None:
                fill_cmd(cmd, step['second_lower'])
                second.Write(cmd)
        if k % 10 == 0 and step.get('estop') is not None:  # 50 Hz estop samples
            trig, plug = step['estop']
            seq += 1
            trig_w.write(RosBool(bool(trig)))
            plug_w.write(RosBool(bool(plug)))
            raw_w.write(EStopStatus(bool(trig), bool(plug), seq, int(time.time() * 1000)))
        k += 1
        time.sleep(max(0.0, start + k * 0.002 - time.monotonic()))
    time.sleep(0.2)
    os.killpg(proc.pid, 2)  # SIGINT to the layer's process group: its exit sequence
    try:
        proc.wait(3)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
    with lock:
        rec = list(frames)
    np.savez_compressed(
        OUT / f'{LAYER}_{SCENARIO}.npz',
        t=np.array([r[0] for r in rec]), mode_machine=np.array([r[1] for r in rec]),
        mode_pr=np.array([r[2] for r in rec]), motor=np.array([r[3] for r in rec]),
        reserve=np.array([r[4] for r in rec]), crc=np.array([r[5] for r in rec]),
        duration=duration)
    print(f'{LAYER} {SCENARIO}: {len(rec)} frames')


main()
