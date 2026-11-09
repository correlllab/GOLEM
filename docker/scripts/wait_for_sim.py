#!/usr/bin/env python3
"""Wait for a simulator's first rt/lowstate sample before the bringup starts.

launch_ros.sh runs it when GOLEM_WAIT_FOR_SIM=1 (docker_stack.sh's
docker-compose.stack.yml), and tests/stack's launch_ros_stack.sh runs it
directly. Both robots' simulators publish the same rt/lowstate type, so it
also refuses a simulator of the other robot, told apart by
LowState_.mode_machine.

Environment: GOLEM_ROBOT (h12 or g1, default h12), ROS_DOMAIN_ID (1..232,
default 1), GOLEM_SIM_TIMEOUT (s, > 0, default 180).
Exit status: 0 when a sample of the selected robot arrives; 1 on no sample
within the timeout or another robot's sample. A bad environment value raises
ValueError.
"""
import os
import sys

# LowState_.mode_machine each robot's simulators report: the real robot's value
# (golem_robocasa/unitree_interface.py and CL_isaaclab_sim for the H1-2,
# golem_robocasa/g1/unitree_interface.py for the G1). golem_safety's robot
# profiles (robots/<robot>.yaml) hold the same values.
MODE_MACHINE = {'h12': 6, 'g1': 5}


def expected_mode_machine(robot: str) -> int:
    """Return the mode_machine of robot's simulators; ValueError unless robot is h12 or g1."""
    if robot not in MODE_MACHINE:
        raise ValueError(f"GOLEM_ROBOT must be h12 or g1, not {robot!r}")
    return MODE_MACHINE[robot]


def main():
    """Wait for one rt/lowstate sample, check its robot and return the exit status."""
    robot = os.environ.get('GOLEM_ROBOT', '').strip().lower() or 'h12'
    expected = expected_mode_machine(robot)
    domain = int(os.environ.get('ROS_DOMAIN_ID') or '1')
    timeout = float(os.environ.get('GOLEM_SIM_TIMEOUT', '180'))
    if not 1 <= domain <= 232 or timeout <= 0:
        raise ValueError('Expected simulation domain 1..232 and a positive GOLEM_SIM_TIMEOUT')
    # Imported after the checks, so a bad environment fails without the SDK.
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
