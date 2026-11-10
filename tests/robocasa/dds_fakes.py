"""Stand in for unitree_sdk2py's DDS channels and cyclonedds' ddspy_write, so that no DDS participant opens.

FakePublisher has the private attribute chain of unitree_sdk2py's ChannelPublisher, ending in a
stand-in DataWriter, so that unitree_common.LowStateWriter writes its bytes through
fake_ddspy_write into the publisher's `sent`; samples handed to its Write (the SDK path) go to
`samples`. FakeSubscriber keeps its handler, so that a test delivers frames through it.
fake_dds puts both, and fake_ddspy_write, in place of a module's channels for one test. Used by
test_h12_unitree_interface.py, test_h12_mode_machine.py and test_lowstate_writer.py.
"""
import os
import types
from unittest import mock

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

import unitree_common


class FakePublisher:
    """Stand in for ChannelPublisher, keeping what is written to it."""

    by_ref = {}   # stand-in DataWriter entity handle -> publisher

    def __init__(self, topic=None, kind=None, writer=True):
        """Keep the topic; open nothing. writer=False leaves out the DataWriter, as on an SDK without one."""
        self.topic, self.sent, self.samples, self.closed = topic, [], [], False
        if writer:
            self.ref = len(FakePublisher.by_ref) + 1
            FakePublisher.by_ref[self.ref] = self
            dds_writer = types.SimpleNamespace(_ref=self.ref, _use_version_2=None)
            self._ChannelPublisher__channel = types.SimpleNamespace(
                _Channel__writer=types.SimpleNamespace(_Writer__writer=dds_writer))

    def Init(self):
        """Do nothing: there is no channel."""

    def Write(self, sample):
        """Keep an SDK sample (the SDK path's write)."""
        self.samples.append(sample)

    def Close(self):
        """Mark the publisher closed."""
        self.closed = True

    def last(self):
        """Return the last bytes written through its DataWriter, decoded by the SDK's LowState_."""
        return LowState_.deserialize(self.sent[-1])


def fake_ddspy_write(ref, data):
    """Keep the bytes written to a FakePublisher's DataWriter, and report success."""
    FakePublisher.by_ref[ref].sent.append(bytes(data))
    return 0


class FakeSubscriber:
    """Stand in for ChannelSubscriber, keeping the handler so that a test delivers frames through it."""

    by_topic = {}

    def __init__(self, topic, kind):
        """Register the subscriber by topic; open nothing."""
        self.topic, self.handler, self.closed = topic, None, False
        FakeSubscriber.by_topic[topic] = self

    def Init(self, handler, queue_size):
        """Keep the handler."""
        self.handler = handler

    def Close(self):
        """Mark the subscriber closed."""
        self.closed = True


def fake_dds(test, module):
    """Replace module's DDS channels and cyclonedds' ddspy_write with the fakes until test ends.

    ROS_DOMAIN_ID is 199 meanwhile: SimInterface refuses a missing value and the real robots' 0
    when it opens its channels, and the fakes reach no network.
    """
    FakePublisher.by_ref = {}
    FakeSubscriber.by_topic = {}
    patchers = [mock.patch.dict(os.environ, {'ROS_DOMAIN_ID': '199'})]
    patchers += [mock.patch.object(target, name, fake) for target, name, fake in (
        (module, 'ChannelFactoryInitialize', lambda *args, **kwargs: None),
        (module, 'ChannelPublisher', FakePublisher),
        (module, 'ChannelSubscriber', FakeSubscriber),
        (unitree_common, 'ddspy_write', fake_ddspy_write))]
    for patcher in patchers:
        patcher.start()
        test.addCleanup(patcher.stop)
