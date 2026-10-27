#!/usr/bin/env python3
"""Writes test/fixtures/lowcmd_crc.json, for test_lowcmd.cpp.

Each entry is a unitree_hg LowCmd built with the Unitree SDK, with the CRC the SDK's
CRC().Crc computes for it. The package code and its tests never import the SDK: they
compare against this file. Field values are rounded to float32 first, so the JSON holds
the exact value the message carries.

  commands      five commands with known field values (test LowCmd.CrcMatchesSdkFixtures)
  old_commands  what the two old safety layers write as their estop command, with the
                header the new layer stamps (mode_machine 6 for the H1-2, 5 for the G1)
                and crc_field set to the SDK's CRC, as a published command carries it:
                h12_estop is make_estop_cmd(LowCmdDefault()) of h12_safety_layer and
                g1_damp is make_estop_cmd(template, kd) of g1_safety_layer

Both old packages must import for the old_commands, so run this in the G1-mode ros
container after the workspace is built and sourced:

  python3 /home/code/core_ws/src/golem_safety/tools/crc_fixtures.py
"""
import json
from pathlib import Path
import random
import re

import numpy as np
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDefault
from unitree_sdk2py.utils.crc import CRC

from g1_safety_layer.core.safety_checks import make_estop_cmd as g1_make_estop_cmd
from h12_safety_layer.core.safety_checks import make_estop_cmd as h12_make_estop_cmd

OUTPUT = Path(__file__).resolve().parents[1] / 'test' / 'fixtures' / 'lowcmd_crc.json'
SLOTS = 35
H12_MODE_MACHINE = 6
G1_MODE_MACHINE = 5
FIELDS = ('mode', 'q', 'dq', 'tau', 'kp', 'kd')


def f32(value):
    """The float32 nearest `value`, as a Python float."""
    return float(np.float32(value))


def blank(mode_machine=0, mode_pr=0):
    cmd = LowCmdDefault()
    cmd.mode_pr = mode_pr
    cmd.mode_machine = mode_machine
    return cmd


def set_motor(cmd, slot, mode, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0):
    motor = cmd.motor_cmd[slot]
    motor.mode = mode
    motor.q = f32(q)
    motor.dq = f32(dq)
    motor.tau = f32(tau)
    motor.kp = f32(kp)
    motor.kd = f32(kd)
    motor.reserve = reserve


def entry(name, cmd, crc_field=None, **extra):
    """One JSON entry: the fields as columns, the CRC the SDK computes, and crc_field."""
    crc = CRC().Crc(cmd)
    out = {
        'name': name,
        'mode_pr': cmd.mode_pr,
        'mode_machine': cmd.mode_machine,
    }
    for field in FIELDS:
        out[field] = [getattr(cmd.motor_cmd[i], field) for i in range(SLOTS)]
    out['motor_reserve'] = [cmd.motor_cmd[i].reserve for i in range(SLOTS)]
    out['reserve'] = list(cmd.reserve)
    # The CRC does not cover the crc field, so a command's own value is arbitrary.
    out['crc_field'] = cmd.crc if crc_field is None else crc_field
    out['crc'] = crc
    out.update(extra)
    return out


def command_vectors():
    rng = random.Random(20261027)

    zero = blank()

    h12_stand = blank(H12_MODE_MACHINE)
    for slot in range(27):
        set_motor(
            h12_stand, slot, 1, q=rng.uniform(-0.5, 0.5), dq=rng.uniform(-0.1, 0.1),
            tau=rng.uniform(-5.0, 5.0), kp=rng.choice((60.0, 100.0, 200.0, 300.0)),
            kd=rng.choice((1.5, 2.5, 5.0, 8.0)))

    g1_damp = blank(G1_MODE_MACHINE)
    for slot in range(29):
        set_motor(g1_damp, slot, 1, kd=0.5 * (1 + slot % 16))

    g1_arm_sdk = blank(G1_MODE_MACHINE)
    set_motor(g1_arm_sdk, 29, 0, q=0.5)  # the arm_sdk blend weight
    for slot in range(12, 29):
        set_motor(
            g1_arm_sdk, slot, 1, q=rng.uniform(-1.0, 1.0), kp=rng.uniform(20.0, 80.0),
            kd=rng.uniform(1.0, 4.0))

    every_field = blank(mode_machine=200, mode_pr=1)
    for slot in range(SLOTS):
        set_motor(
            every_field, slot, rng.randrange(256),
            q=rng.uniform(-6.0, 6.0) * 10.0 ** rng.randint(-6, 3),
            dq=rng.uniform(-30.0, 30.0), tau=rng.uniform(-200.0, 200.0),
            kp=rng.uniform(0.0, 500.0), kd=rng.uniform(0.0, 50.0),
            reserve=rng.getrandbits(32))
    set_motor(every_field, 0, 255, q=-0.0, dq=3.0e38, tau=-3.0e38, kp=1.0e-30, kd=1.0e-45)
    for i in range(4):
        every_field.reserve[i] = rng.getrandbits(32)
    every_field.crc = 0xDEADBEEF

    return [
        entry('all_zero_default', zero),
        entry('h12_stand_like', h12_stand),
        entry('g1_damp_per_slot_kd', g1_damp),
        entry('g1_arm_sdk_weight', g1_arm_sdk, crc_field=0x12345678),
        entry('every_field_set', every_field),
    ]


def old_estop_commands():
    h12_template = blank(H12_MODE_MACHINE)
    h12_cmd = h12_make_estop_cmd(h12_template)
    h12_cmd.crc = CRC().Crc(h12_cmd)

    kd = [0.5 * (1 + i % 16) for i in range(29)]
    g1_template = blank(G1_MODE_MACHINE)
    g1_cmd = g1_make_estop_cmd(g1_template, np.asarray(kd))
    g1_cmd.crc = CRC().Crc(g1_cmd)

    return {
        'h12_estop': entry('h12_estop', h12_cmd),
        'g1_damp': entry('g1_damp', g1_cmd, kd_input=kd),
    }


def compact(text):
    """json.dumps output with each list of scalars on one line."""
    def one_line(match):
        return '[' + re.sub(r'\s*\n\s*', ' ', match.group(1)).strip() + ']'
    return re.sub(r'\[([^\[\]{}]*)\]', one_line, text)


def main():
    data = {
        'generator': 'tools/crc_fixtures.py',
        'commands': command_vectors(),
        'old_commands': old_estop_commands(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(compact(json.dumps(data, indent=1)) + '\n')
    print(f'wrote {OUTPUT}')


if __name__ == '__main__':
    main()
