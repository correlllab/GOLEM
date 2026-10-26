'''pytest setup shared by the tests'''

import threading

from cyclonedds.idl._main import IDL

# cyclonedds-python 0.10 serializes every sample of a type through one buffer
# of that type, without a lock. The in-process DDS tests write LowCmd_ inputs
# from their own threads while the safety layer's publisher loop writes its
# LowCmd_, and two writes at once can mix the two samples: in a probe with two
# threads writing back to back, about 3 % of the samples came out with the
# other thread's fields, and in a test the layer's rt/lowcmd once carried an
# input's mode_machine 0. The safety node writes LowCmd_ from its publisher
# loop only; the tests write one sample at a time
_SERIALIZE = IDL.serialize
_SERIALIZE_LOCK = threading.Lock()


def _serialize_one_at_a_time(self, *args, **kwargs):
    with _SERIALIZE_LOCK:
        return _SERIALIZE(self, *args, **kwargs)


IDL.serialize = _serialize_one_at_a_time
