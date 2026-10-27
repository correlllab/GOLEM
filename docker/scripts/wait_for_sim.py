#!/usr/bin/env python3
"""Wait for a fresh Unitree lowstate sample before starting ROS bringup, and
refuse it when the simulator runs another robot than GOLEM_ROBOT (h12 or g1,
default h12). Both robots' simulators publish the same rt/lowstate IDL, so on
a shared domain a G1 bringup would otherwise start against the H1-2 sim, or
the reverse."""
import os
import sys

# LowState_.mode_machine of each robot's simulators: the G1 sim reports the
# G1's (MODE_MACHINE in h1_robocasa/g1/unitree_interface.py); the H1-2 sims,
# RoboCasa and Isaac, leave the IDL default.
MODE_MACHINE = {'h12': 0, 'g1': 5}


def expected_mode_machine(robot: str) -> int:
    """The mode_machine that a simulator of robot ('h12' or 'g1') reports."""
    if robot not in MODE_MACHINE:
        raise ValueError(f"GOLEM_ROBOT must be h12 or g1, not {robot!r}")
    return MODE_MACHINE[robot]


def main():
    robot = os.environ.get('GOLEM_ROBOT', '').strip().lower() or 'h12'
    expected = expected_mode_machine(robot)
    domain = int(os.environ.get('ROS_DOMAIN_ID') or '1')
    timeout = float(os.environ.get('GOLEM_SIM_TIMEOUT', '180'))
    if not 1 <= domain <= 232 or timeout <= 0:
        raise ValueError('Expected simulation domain 1..232 and a positive GOLEM_SIM_TIMEOUT')
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(domain)
    subscriber = ChannelSubscriber('rt/lowstate', LowState_)
    subscriber.Init()
    print(f'[bringup] waiting up to {timeout:g}s for rt/lowstate on domain {domain}', flush=True)
    try:
        sample = subscriber.Read(timeout)
    finally:
        subscriber.Close()
    if sample is None:
        print('[bringup] no simulator sample received; inspect simulator logs', file=sys.stderr)
        return 1
    if sample.mode_machine != expected:
        print(f'[bringup] the simulator is not a {robot} (mode_machine {sample.mode_machine}): a {robot} '
              f'simulator reports {expected}. Stop the simulator on domain {domain}, or select the '
              'robot it runs (--g1/--h12 or GOLEM_ROBOT).', file=sys.stderr)
        return 1
    print('[bringup] simulator is publishing', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
