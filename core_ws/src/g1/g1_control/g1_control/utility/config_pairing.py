'''
Which g1_safety_layer and controller configs may run together: the one place
for these checks, used by robot_safety_launch.py and g1_bringup's
g1_core.launch.py. Configs are told apart by name; sim configs start with sim.
Each known safety config runs with one controller config
(SAFETY_CONTROLLER_CONFIGS)
'''

# g1_safety_layer configs that relay the upper-body channel this controller publishes
UPPER_CHANNEL_SAFETY_CONFIGS = ('arm_sdk', 'split', 'sim_arm_sdk', 'sim_split')
# this package's configs; the *_split ones hold the waist stiff for a walking policy
CONTROLLER_CONFIGS = ('g1', 'g1_split', 'sim', 'sim_split')
# the controller config each safety config runs with: its gains suit the mode
# (the *_split ones hold the waist stiff, whoever drives the legs) and its
# topics.safety_out is the safety config's topics.low_cmd_out, which the follow
# gate reads
SAFETY_CONTROLLER_CONFIGS = {
    'arm_sdk': 'g1',
    'split': 'g1_split',
    'sim_arm_sdk': 'sim',
    'sim_split': 'sim_split',
}
# with a walking policy on the legs: the safety layer's split mode and a split
# controller config, on the real robot (False) and in the sim (True)
LOCOMOTION_SAFETY_CONFIGS = {False: 'split', True: 'sim_split'}
LOCOMOTION_CONTROLLER_CONFIGS = {False: 'g1_split', True: 'sim_split'}


def config_stem(config_name: str) -> str:
    '''Return a config name without directory or .yaml suffix'''
    stem = config_name.rsplit('/', 1)[-1]
    return stem[:-5] if stem.endswith('.yaml') else stem


def is_sim_config(config_name: str) -> bool:
    '''True for a sim config (sim, sim_split, sim_arm_sdk, ...), by its name'''
    return config_stem(config_name).startswith('sim')


def default_controller_config(safety_config: str) -> str | None:
    '''The controller config that runs with a safety config, or None if unknown'''
    return SAFETY_CONTROLLER_CONFIGS.get(config_stem(safety_config))


def pairing_refusal(controller_config: str, safety_config: str,
                    sim: bool | None = None, locomotion: bool = False) -> str | None:
    '''
    Why the controller and safety configs must not run together, or None.
    sim is True in the sim, False on the real robot and None when unknown, in
    which case only configs that mix sim and real are refused. locomotion (a
    walking policy on the legs) needs sim to be known
    '''
    controller = config_stem(controller_config)
    safety = config_stem(safety_config)
    if safety not in UPPER_CHANNEL_SAFETY_CONFIGS:
        return (f'safety config "{safety}" must relay the upper-body channel: '
                f'one of {", ".join(UPPER_CHANNEL_SAFETY_CONFIGS)}')
    if locomotion:
        split_safety = LOCOMOTION_SAFETY_CONFIGS[sim]
        split_controller = LOCOMOTION_CONTROLLER_CONFIGS[sim]
        if safety != split_safety:
            return ('use_locomotion:=true drives the legs through g1_safety_layer split mode, '
                    f'so it needs safety_config:={split_safety}, not {safety}')
        if controller != split_controller:
            return (f'use_locomotion:=true needs controller_config:={split_controller}, '
                    f'whose stiff waist the walking policy expects, not {controller}')
    # a custom controller config is only known to be a sim one by its name
    controller_known = controller in CONTROLLER_CONFIGS or is_sim_config(controller)
    if sim is not None:
        for kind, name, known in (('safety', safety, True),
                                  ('controller', controller, controller_known)):
            if known and is_sim_config(name) != sim:
                if sim:
                    return f'{kind} config "{name}" is for the real robot, not the sim'
                return f'{kind} config "{name}" is a sim config; the real robot never runs one'
    elif controller_known and is_sim_config(controller) != is_sim_config(safety):
        return (f'controller config "{controller}" and safety config "{safety}" mix sim and '
                'real: use sim.yaml or sim_split.yaml with sim_*, g1.yaml or g1_split.yaml '
                'with the others')
    # a custom controller config's gains and safety_out are not known by its name
    paired = SAFETY_CONTROLLER_CONFIGS[safety]
    if controller in CONTROLLER_CONFIGS and controller != paired:
        return (f'safety config "{safety}" runs with controller config "{paired}", not '
                f'"{controller}": the waist gains suit the mode, and the follow gate reads '
                'the topic the safety layer writes')
    return None
