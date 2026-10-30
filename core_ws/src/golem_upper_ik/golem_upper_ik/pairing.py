'''
Which golem_safety and golem_upper_ik configs may run together, for each robot: the one place
for these checks, used by the bringups. Configs are told apart by name; a directory or a
.yaml suffix is ignored.

- H1-2: each safety config runs with one controller config (H12_SAFETY_CONTROLLER_CONFIGS),
  as the H1-2 bringups launch them; a known controller config with another known safety
  config is refused. Nothing is refused for being a sim or a real config, since one config
  serves both (lmg_sim runs relax_safety_split).
- G1: the safety config must relay the upper-body channel, and sim configs (named sim*) and
  real ones do not mix. Each safety config runs with one controller config
  (G1_SAFETY_CONTROLLER_CONFIGS), one for the real robot and one for the sim, so the sim/real
  rule keeps each known controller config to its own safety config.

A config outside the tables is not known by its name, so its contents decide; only the G1
refuses an unknown safety config, which may not relay the upper-body channel.
'''
from golem_upper_ik.robots import ROBOTS

__all__ = [
    'config_stem', 'default_controller_config', 'is_sim_config', 'pairing_refusal',
    'G1_CONTROLLER_CONFIGS', 'G1_SAFETY_CONTROLLER_CONFIGS', 'G1_UPPER_CHANNEL_SAFETY_CONFIGS',
    'H12_CONTROLLER_CONFIGS', 'H12_SAFETY_CONTROLLER_CONFIGS',
]

# The controller config each H1-2 safety config runs with. The *_split ones write the
# upper-body input (/safety/lowcmd_upper_in), safety_full the full-body input
# (/safety/lowcmd_in), and sim_safety_split's limits suit the sim. debug, which
# joint_state_publisher reads, is in no pair.
H12_SAFETY_CONTROLLER_CONFIGS = {
    'relax_safety_split': 'safety_split',
    'default_safety_split': 'safety_split',
    'tight_safety_split': 'safety_split',
    'sim_safety_split': 'sim_safety_split',
    'relax_safety_full': 'safety_full',
    'default_safety_full': 'safety_full',
    'tight_safety_full': 'safety_full',
    'locomotion_safety_full': 'safety_full',
}
H12_CONTROLLER_CONFIGS = ('safety_split', 'sim_safety_split', 'safety_full')

# G1 safety configs that relay the upper-body channel the controller writes
G1_UPPER_CHANNEL_SAFETY_CONFIGS = ('split', 'sim_split')
# the G1 controller configs; both hold the waist stiff for a walking policy
G1_CONTROLLER_CONFIGS = ('g1_split', 'sim_split')
# the controller config each G1 safety config runs with: its speed limits suit the real
# robot or the sim, and its topics.safety_out is the safety config's output, which the follow
# gate reads
G1_SAFETY_CONTROLLER_CONFIGS = {
    'split': 'g1_split',
    'sim_split': 'sim_split',
}


def _check_robot(robot: str) -> None:
    if robot not in ROBOTS:
        raise ValueError(f"unknown robot '{robot}': one of {', '.join(ROBOTS)}")


def config_stem(config_name: str) -> str:
    '''A config name without its directory or .yaml suffix'''
    stem = config_name.rsplit('/', 1)[-1]
    return stem[:-5] if stem.endswith('.yaml') else stem


def is_sim_config(config_name: str) -> bool:
    '''True for a G1 sim config (sim_split, sim_custom, ...), by its name'''
    return config_stem(config_name).startswith('sim')


def default_controller_config(robot: str, safety_config: str) -> str | None:
    '''The controller config that runs with a safety config of robot, or None if unknown'''
    _check_robot(robot)
    table = H12_SAFETY_CONTROLLER_CONFIGS if robot == 'h12' else G1_SAFETY_CONTROLLER_CONFIGS
    return table.get(config_stem(safety_config))


def pairing_refusal(robot: str, controller_config: str, safety_config: str,
                    sim: bool | None = None) -> str | None:
    '''
    Why the controller and safety configs of robot must not run together, or None.
    sim is True in the sim, False on the real robot and None when unknown. On the G1, an
    unknown sim refuses only configs that mix sim and real. The H1-2 checks the pair only
    '''
    _check_robot(robot)
    if robot == 'h12':
        return _h12_refusal(config_stem(controller_config), config_stem(safety_config))
    return _g1_refusal(config_stem(controller_config), config_stem(safety_config), sim)


def _h12_refusal(controller: str, safety: str) -> str | None:
    paired = H12_SAFETY_CONTROLLER_CONFIGS.get(safety)
    if paired is None or controller not in H12_CONTROLLER_CONFIGS or controller == paired:
        return None
    return (f'safety config "{safety}" runs with controller config "{paired}", not '
            f'"{controller}": the controller must write the input the safety config relays, '
            'with the limits it was tuned with')


def _g1_refusal(controller: str, safety: str, sim: bool | None) -> str | None:
    if safety not in G1_UPPER_CHANNEL_SAFETY_CONFIGS:
        return (f'safety config "{safety}" must relay the upper-body channel: '
                f'one of {", ".join(G1_UPPER_CHANNEL_SAFETY_CONFIGS)}')
    # a custom controller config is known to be a sim one only by its name, and is not
    # matched to the safety config: its gains and safety_out are not known by its name
    controller_known = controller in G1_CONTROLLER_CONFIGS or is_sim_config(controller)
    if sim is not None:
        for kind, name, known in (('safety', safety, True),
                                  ('controller', controller, controller_known)):
            if known and is_sim_config(name) != sim:
                if sim:
                    return f'{kind} config "{name}" is for the real robot, not the sim'
                return f'{kind} config "{name}" is a sim config; the real robot never runs one'
    elif controller_known and is_sim_config(controller) != is_sim_config(safety):
        return (f'controller config "{controller}" and safety config "{safety}" mix sim and '
                'real: use sim_split.yaml with sim_*, g1_split.yaml with the others')
    return None
