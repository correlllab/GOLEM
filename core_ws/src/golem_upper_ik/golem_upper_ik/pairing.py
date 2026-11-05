'''
Which golem_safety and golem_upper_ik configs may run together, for each robot: the one place
for these checks, used by the bringups. Configs are told apart by name; a directory or a
.yaml suffix is ignored.

Each robot's config/<robot>/pairing.yaml pairs each of its safety configs with the one
controller config that runs with it. One rule serves every robot, checked in this order:

- the safety config is one of the robot's pairs;
- the controller config is the one paired with it;
- with sim known, neither is a sim config (named sim*) on the real robot or a real config in
  the sim.
'''
import os

from ament_index_python.packages import get_package_share_directory
import yaml

from golem_upper_ik.profile import robot_config

__all__ = ['config_stem', 'default_controller_config', 'is_sim_config', 'pairing_refusal']


def config_stem(config_name: str) -> str:
    '''A config name without its directory or .yaml suffix'''
    stem = config_name.rsplit('/', 1)[-1]
    return stem[:-5] if stem.endswith('.yaml') else stem


def is_sim_config(config_name: str) -> bool:
    '''True for a sim config (sim_split, sim_safety_split, sim_custom, ...), by its name'''
    return config_stem(config_name).startswith('sim')


def _pairs(robot: str) -> dict[str, str]:
    '''The safety config -> controller config stems of robot; ValueError for an unknown robot'''
    robot_config(robot)
    path = os.path.join(get_package_share_directory('golem_upper_ik'), 'config', robot,
                        'pairing.yaml')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'robot {robot} has no pairing config: {path} does not exist')
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    pairs = raw.get('pairs') if isinstance(raw, dict) and list(raw) == ['pairs'] else None
    if not isinstance(pairs, dict) or not all(
            isinstance(safety, str) and isinstance(controller, str)
            for safety, controller in pairs.items()):
        raise ValueError(f"{path}: the file must hold only 'pairs', a mapping of each safety "
                         'config name to its controller config name')
    return pairs


def _unpaired(robot: str, safety: str, pairs: dict[str, str]) -> str:
    return (f'safety config "{safety}" is not one of the {robot_config(robot)["display_name"]} '
            f'pairs: {", ".join(pairs)}')


def default_controller_config(robot: str, safety_config: str) -> str:
    '''
    The controller config that runs with a safety config of robot. ValueError for an unknown
    robot or a safety config that is in none of its pairs
    '''
    pairs = _pairs(robot)
    safety = config_stem(safety_config)
    if safety not in pairs:
        raise ValueError(_unpaired(robot, safety, pairs))
    return pairs[safety]


def pairing_refusal(robot: str, controller_config: str, safety_config: str,
                    sim: bool | None = None) -> str | None:
    '''
    Why the controller and safety configs of robot must not run together, or None. sim is
    True in the sim, False on the real robot and None when unknown (no sim check)
    '''
    pairs = _pairs(robot)
    safety = config_stem(safety_config)
    controller = config_stem(controller_config)
    if safety not in pairs:
        return _unpaired(robot, safety, pairs)
    if controller != pairs[safety]:
        return (f'safety config "{safety}" runs with controller config "{pairs[safety]}", not '
                f'"{controller}": the controller must write the input the safety config '
                'relays, with the limits it was tuned with')
    if sim is not None:
        for kind, name in (('safety', safety), ('controller', controller)):
            if is_sim_config(name) != sim:
                if sim:
                    return f'{kind} config "{name}" is for the real robot, not the sim'
                return f'{kind} config "{name}" is a sim config; the real robot never runs one'
    return None
