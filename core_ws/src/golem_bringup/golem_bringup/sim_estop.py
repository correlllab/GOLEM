"""Name the estop box the H1-2 sims serve, for estop_node's serial_port."""
import os

# The sims' estop servers listen on BASE_PORT + ROS_DOMAIN_ID; keep in step with
# golem_robocasa/estop_server.py and CL_isaaclab_sim's estop_server.py.
BASE_PORT = 15700


def sim_estop_url(environ=None) -> str:
    """Return socket://127.0.0.1:<port> of the sim on this shell's ROS_DOMAIN_ID (1 when unset)."""
    env = os.environ if environ is None else environ
    domain = env.get('ROS_DOMAIN_ID', '').strip() or '1'
    return f'socket://127.0.0.1:{BASE_PORT + int(domain)}'
