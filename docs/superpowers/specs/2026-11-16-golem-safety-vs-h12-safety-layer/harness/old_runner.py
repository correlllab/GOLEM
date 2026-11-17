"""Run h12_safety_layer's SafetyLayer on a sim domain, never domain 0.

Usage: ROS_DOMAIN_ID=<n != 0> python3 old_runner.py <old checkout> <config.yaml> <use_estop 0|1>

The old layer's DDS channel factory reads ROS_DOMAIN_ID; its raw estop reader
opens a DomainParticipant on the default domain (0), so it is pinned here to
ROS_DOMAIN_ID as well. Logging to disk is off. Prints the layer's events.
"""
import os
import sys

DOMAIN = int(os.environ.get('ROS_DOMAIN_ID', '0'))
# SAFETY: domain 0 is the real robots' command bus; this harness never writes there.
if DOMAIN == 0:
    sys.exit('old_runner refuses ROS_DOMAIN_ID 0')
sys.path.insert(0, sys.argv[1])

from cyclonedds.domain import DomainParticipant  # noqa: E402
import h12_safety_layer.core.estop_subscriber as es  # noqa: E402
from h12_safety_layer.core.config import init_channel_factory, load_config  # noqa: E402
from h12_safety_layer.core.safety_layer import SafetyLayer  # noqa: E402

es.DomainParticipant = lambda: DomainParticipant(DOMAIN)
config = load_config(sys.argv[2])
config['estop']['enabled'] = sys.argv[3] == '1'
config['logging']['enabled'] = False
init_channel_factory(config)
SafetyLayer(config).start()
