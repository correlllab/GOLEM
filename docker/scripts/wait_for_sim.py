#!/usr/bin/env python3
"""Wait for a fresh Unitree lowstate sample before starting ROS bringup."""
import os
import sys


def main():
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
        if subscriber.Read(timeout) is None:
            print('[bringup] no simulator sample received; inspect simulator logs', file=sys.stderr)
            return 1
    finally:
        subscriber.Close()
    print('[bringup] simulator is publishing', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
