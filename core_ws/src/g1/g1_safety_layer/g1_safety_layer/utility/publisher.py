'''
Unitree SDK2 channel publisher example.
This module publishes `builtin_interfaces/Time_` timestamps to `demo/topic`.
The subscriber measures one-way latency by comparing against its local clock.
'''

import time

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.builtin_interfaces.msg.dds_ import Time_

from g1_safety_layer.core.config import init_channel_factory, load_config


def main(config_path: str) -> None:
    '''Initialize a channel publisher and send one message per second'''
    init_channel_factory(load_config(config_path))

    publisher = ChannelPublisher('demo/topic', Time_)
    publisher.Init()

    counter = 0
    while True:
        t = time.time()
        sec = int(t)
        nanosec = int((t - sec) * 1e9)
        publisher.Write(Time_(sec, nanosec))
        print(f'[PUB] seq={counter}  t={t:.6f}')
        counter += 1
        time.sleep(1.0)


if __name__ == '__main__':
    main('config/full.yaml')
