'''pytest setup shared by the tests'''

import gc

# the DDS tests close every channel they open, but a test that fails while
# setting up may not. When the cyclic garbage collector reclaims an unclosed
# Unitree channel, cyclonedds-python frees its listener but keeps its DDS
# entity, and the next channel on that topic can crash the process; with the
# collector off, an unclosed channel stays alive until the session ends
gc.disable()
