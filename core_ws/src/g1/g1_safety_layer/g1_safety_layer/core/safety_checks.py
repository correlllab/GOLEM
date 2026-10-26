'''Safety checks for low_cmd and low_state messages'''

import copy
from typing import Any, Sequence

import numpy as np

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from g1_safety_layer.core.joint_limits import JOINT_NAMES, MOTOR_COUNT, WAIST_SLOTS

class CmdValidationError(Exception):
    '''Command payload has non-finite values, or a waist kp below arm_sdk.waist_kp_min'''

class EStopTriggered(Exception):
    '''E-stop condition was met from low_state feedback'''

def clone_cmd(msg: LowCmd_) -> LowCmd_:
    '''Create a deep copy of low_cmd before mutating'''
    return copy.deepcopy(msg)

def _cmd_arrays(msg: LowCmd_) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray([float(msg.motor_cmd[i].q) for i in range(MOTOR_COUNT)], dtype=np.float64)
    dq = np.asarray([float(msg.motor_cmd[i].dq) for i in range(MOTOR_COUNT)], dtype=np.float64)
    tau = np.asarray([float(msg.motor_cmd[i].tau) for i in range(MOTOR_COUNT)], dtype=np.float64)
    kp = np.asarray([float(msg.motor_cmd[i].kp) for i in range(MOTOR_COUNT)], dtype=np.float64)
    kd = np.asarray([float(msg.motor_cmd[i].kd) for i in range(MOTOR_COUNT)], dtype=np.float64)
    return q, dq, tau, kp, kd

def clip_low_cmd(msg: LowCmd_, limits: dict[str, np.ndarray]) -> tuple[LowCmd_, int]:
    '''Validate finite values and clip command to configured limits'''

    out = clone_cmd(msg)
    q_raw, dq_raw, tau_raw, kp_raw, kd_raw = _cmd_arrays(out)

    if not np.all(np.isfinite(q_raw)):
        raise CmdValidationError('q contains non-finite values')
    if not np.all(np.isfinite(dq_raw)):
        raise CmdValidationError('dq contains non-finite values')
    if not np.all(np.isfinite(tau_raw)):
        raise CmdValidationError('tau contains non-finite values')
    if not np.all(np.isfinite(kp_raw)):
        raise CmdValidationError('kp contains non-finite values')
    if not np.all(np.isfinite(kd_raw)):
        raise CmdValidationError('kd contains non-finite values')

    q_low = limits['q_clip_limits'][:, 0]
    q_high = limits['q_clip_limits'][:, 1]
    q_new = np.clip(q_raw, q_low, q_high)
    dq_new = np.clip(dq_raw, -limits['dq_clip_limits'], limits['dq_clip_limits'])
    tau_new = np.clip(tau_raw, -limits['tau_clip_limits'], limits['tau_clip_limits'])
    kp_new = np.clip(kp_raw, 0.0, limits['kp_clip_max'])
    kd_new = np.clip(kd_raw, 0.0, limits['kd_clip_max'])

    # changed = np.logical_or.reduce((
    #     q_new != q_raw,
    #     dq_new != dq_raw,
    #     tau_new != tau_raw,
    #     kp_new != kp_raw,
    #     kd_new != kd_raw,
    # ))
    # clipped_count = int(np.count_nonzero(changed))

    for i in range(MOTOR_COUNT):
        motor_cmd = out.motor_cmd[i]
        motor_cmd.q = float(q_new[i])
        motor_cmd.dq = float(dq_new[i])
        motor_cmd.tau = float(tau_new[i])
        motor_cmd.kp = float(kp_new[i])
        motor_cmd.kd = float(kd_new[i])

    return out


def clipped_gains(
    msg: LowCmd_, limits: dict[str, np.ndarray], slots: Sequence[int]
) -> list[tuple[int, str, float, float]]:
    '''List (slot, gain, commanded, max) for each kp/kd that clip_low_cmd lowers'''
    clipped = []
    for i in slots:
        motor_cmd = msg.motor_cmd[i]
        for gain, value, limit in (
            ('kp', float(motor_cmd.kp), float(limits['kp_clip_max'][i])),
            ('kd', float(motor_cmd.kd), float(limits['kd_clip_max'][i])),
        ):
            if value > limit:
                clipped.append((i, gain, value, limit))
    return clipped


def check_waist_kp(msg: LowCmd_, kp_min: float) -> None:
    '''
    Raise CmdValidationError if a waist slot's kp is below kp_min: a waist that
    soft cannot hold the torso up, and a stray publisher's default command
    leaves it at 0
    '''
    for i in WAIST_SLOTS:
        kp = float(msg.motor_cmd[i].kp)
        if kp < kp_min:
            raise CmdValidationError(
                f'motor {i} ({JOINT_NAMES[i]}) kp {kp} is below arm_sdk.waist_kp_min {kp_min}'
            )


def engage_refusal(
    msg: LowCmd_,
    slots: Sequence[int],
    state_q: np.ndarray | None,
    state_age: float | None,
    mode_machine: int | None,
    gate: dict[str, Any],
) -> str | None:
    '''
    Why the layer must not take control with msg yet, or None when it may:
    rt/lowstate must be fresh and report the expected mode_machine, and msg
    must be within max_position_error of the measured q on every given slot
    '''
    if state_q is None or state_age is None:
        return 'no rt/lowstate received yet'
    max_age = gate['max_state_age_seconds']
    if state_age > max_age:
        return f'rt/lowstate is {state_age:.3f} s old (> {max_age} s)'
    if mode_machine != gate['mode_machine']:
        return f'rt/lowstate mode_machine is {mode_machine}, expected {gate["mode_machine"]}'
    idx = np.asarray(list(slots), dtype=np.int64)
    q_cmd = np.asarray([float(msg.motor_cmd[i].q) for i in idx], dtype=np.float64)
    error = np.abs(q_cmd - state_q[idx])
    if not np.all(np.isfinite(error)):
        bad = int(idx[np.flatnonzero(~np.isfinite(error))[0]])
        return f'motor {bad} ({JOINT_NAMES[bad]}) has a non-finite q'
    worst = int(np.argmax(error))
    max_error = gate['max_position_error']
    if error[worst] > max_error:
        i = int(idx[worst])
        return (
            f'motor {i} ({JOINT_NAMES[i]}) command q {q_cmd[worst]:.3f} is '
            f'{error[worst]:.3f} rad from rt/lowstate q {state_q[i]:.3f} (> {max_error} rad)'
        )
    return None


def check_estop_limits(
    msg: LowState_,
    limits: dict[str, np.ndarray],
    slots: Sequence[int] = range(MOTOR_COUNT),
) -> None:
    '''Raise if low_state exceeds configured estop limits on the given motor slots'''
    idx = np.asarray(list(slots), dtype=np.int64)
    q = np.asarray([float(msg.motor_state[i].q) for i in idx], dtype=np.float64)
    dq = np.asarray([float(msg.motor_state[i].dq) for i in idx], dtype=np.float64)
    tau = np.asarray([float(msg.motor_state[i].tau_est) for i in idx], dtype=np.float64)
    # check for inf and nan
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)) or not np.all(np.isfinite(tau)):
        raise EStopTriggered('low_state contains non-finite values')

    q_low = limits['q_estop_limits'][idx, 0]
    q_high = limits['q_estop_limits'][idx, 1]
    # check joint position q limits
    bad_q = np.where(np.logical_or(q < q_low, q > q_high))[0]
    if bad_q.size > 0:
        i = int(idx[bad_q[0]])
        raise EStopTriggered(f'motor {i} ({JOINT_NAMES[i]}) q out of estop range: {q[bad_q[0]]}')
    # check joint velocity dq limits
    bad_dq = np.where(np.abs(dq) > limits['dq_estop_limits'][idx])[0]
    if bad_dq.size > 0:
        i = int(idx[bad_dq[0]])
        raise EStopTriggered(f'motor {i} ({JOINT_NAMES[i]}) dq out of estop range: {dq[bad_dq[0]]}')
    # check joint torque tau limits
    bad_tau = np.where(np.abs(tau) > limits['tau_estop_limits'][idx])[0]
    if bad_tau.size > 0:
        i = int(idx[bad_tau[0]])
        raise EStopTriggered(
            f'motor {i} ({JOINT_NAMES[i]}) tau out of estop range: {tau[bad_tau[0]]}')

def make_estop_cmd(template: LowCmd_, kd: np.ndarray | None = None) -> LowCmd_:
    '''
    Create an estop command preserving headers: every motor off, or with a
    per-joint kd, every motor on as a pure damper (kp = 0, q = dq = tau = 0)
    '''
    out = clone_cmd(template)
    for i in range(MOTOR_COUNT):
        motor_cmd = out.motor_cmd[i]
        motor_cmd.mode = 0 if kd is None else 1
        motor_cmd.q = 0.0
        motor_cmd.dq = 0.0
        motor_cmd.tau = 0.0
        motor_cmd.kp = 0.0
        motor_cmd.kd = 0.0 if kd is None else float(kd[i])
    return out
