'''rt/arm_sdk hand-over: ramps on the weight slot that splits control with Unitree'''

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from g1_safety_layer.core.safety_checks import clone_cmd
from g1_safety_layer.core.joint_limits import ARM_SDK_WEIGHT_SLOT, LEG_SLOTS


def ramp_fraction(elapsed: float, seconds: float) -> float:
    '''Fraction of a linear ramp completed after elapsed seconds, in [0, 1]'''
    if seconds <= 0.0:
        return 1.0
    return min(max(elapsed / seconds, 0.0), 1.0)


def engage_weight(now: float, engage_start: float, engage_seconds: float) -> float:
    '''Weight while taking the waist and arms from Unitree: ramps 0 -> 1'''
    return ramp_fraction(now - engage_start, engage_seconds)


def release_weight(
    now: float, release_start: float, start_weight: float, release_seconds: float
) -> float:
    '''Weight while handing the waist and arms back to Unitree: ramps start_weight -> 0'''
    return start_weight * (1.0 - ramp_fraction(now - release_start, release_seconds))


def with_weight(cmd: LowCmd_, weight: float) -> LowCmd_:
    '''Copy of cmd with the rt/arm_sdk weight slot set (1 = ours, 0 = Unitree's)'''
    out = clone_cmd(cmd)
    slot = out.motor_cmd[ARM_SDK_WEIGHT_SLOT]
    slot.mode = 0
    slot.q = float(weight)
    slot.dq = 0.0
    slot.tau = 0.0
    slot.kp = 0.0
    slot.kd = 0.0
    return out


def zero_legs(cmd: LowCmd_) -> LowCmd_:
    '''
    Zero every field of the leg slots (0-11) in place, as Unitree's rt/arm_sdk
    example leaves them: the legs are Unitree's in arm_sdk_mode
    '''
    for i in LEG_SLOTS:
        motor_cmd = cmd.motor_cmd[i]
        motor_cmd.mode = 0
        motor_cmd.q = 0.0
        motor_cmd.dq = 0.0
        motor_cmd.tau = 0.0
        motor_cmd.kp = 0.0
        motor_cmd.kd = 0.0
    return cmd
