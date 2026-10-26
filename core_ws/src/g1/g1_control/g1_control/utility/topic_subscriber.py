import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from g1_control.utility.controller_config import (
    init_channel_factory_guard,
    load_controller_config,
)

TOPIC_ARM_SDK = 'rt/arm_sdk'

class TopicSubscriber:
    def __init__(self, topic_name, topic_type):
        '''
        Topic subscriber using ChannelSubscriber

        @param topic_name: name of the topic to subscribe to
        @param topic_type: type of the topic message (e.g., LowState_)
        '''
        self.topic_name = topic_name
        self.topic_type = topic_type
        self.message_count = 0
        self.last_received_time = None
        self.last_received_msg = None

        print(f'Initializing DDSSubscriber for topic: {self.topic_name}')

        # create subscriber for the specified topic type
        self.subscriber = ChannelSubscriber(self.topic_name, self.topic_type)
        self.subscriber.Init(self.topic_callback, 10)

        print(f'Successfully subscribed to topic: {self.topic_name}')

    def topic_callback(self, msg):
        self.message_count += 1
        self.last_received_time = time.time()
        self.last_received_msg = msg

    def get_stats(self):
        return {
            'topic_name': self.topic_name,
            'last_received_time': self.last_received_time,
            'last_received_msg': self.last_received_msg
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f'Topic: {stats["topic_name"]}')
        if stats['last_received_time']:
            print(f'Last message time: {stats["last_received_time"]:.6f}')
        else:
            print('No messages received yet')

    def shutdown(self):
        print(f'Shutting down subscriber for topic: {self.topic_name}')
        self.subscriber.Close()
        self.print_stats()
        print('DDSSubscriber shutdown complete.')

def main():
    print('Starting subscriber...')
    config = load_controller_config()
    init_channel_factory_guard(config)
    subscriber = TopicSubscriber(TOPIC_ARM_SDK, LowCmd_)

    try:
        print('Listening for messages... (Press Ctrl+C to stop)')
        while True:
            time.sleep(0.1)
            if subscriber.message_count > 0:
                msg = subscriber.last_received_msg
                print(msg.motor_cmd[17])
    except KeyboardInterrupt:
        print('Received keyboard interrupt, shutting down...')
    finally:
        subscriber.shutdown()

if __name__ == '__main__':
    main()
