'''
A start-up move whose IK step has no solution stops the controller with an error that names
the closest self-collision pairs against controller.d_min, on both robots. On the G1, from
arms_down (each hand and wrist sphere inside its hip-pitch sphere) the barrier's quadratic
program has no solution at the first step toward home.
'''
from types import SimpleNamespace

import numpy as np
import pytest
from pink.exceptions import NoSolutionFound

ROBOTS = ('h12', 'g1')


class Logger:
    def __init__(self):
        self.errors = []

    def info(self, message):
        pass

    def error(self, message):
        self.errors.append(message)


def controller_at(model, q_reduced, step):
    '''An UpperController with only what a start-up move reads, its IK at q_reduced'''
    from golem_upper_ik.core.controller.upper_controller import UpperController
    controller = UpperController.__new__(UpperController)
    controller.robot_model = model
    controller.d_min = 0.02
    controller.dt = 0.002
    controller._logger = Logger()
    controller.ik_solver = SimpleNamespace(q_reduced=np.asarray(q_reduced, dtype=float))
    controller.goto_reduced_configuration = step
    return controller


@pytest.mark.parametrize('robot', ROBOTS)
def test_no_solution_names_the_closest_pairs(models, robot):
    '''Every pair keeps d_min at home: the error says so and logs the closest pairs'''
    model = models(robot)
    home = model.profile.named_configs['home']

    def no_solution(_):
        raise NoSolutionFound(None, None)

    controller = controller_at(model, home, no_solution)
    with pytest.raises(RuntimeError, match='every self-collision pair keeps '
                                           r'controller.d_min \(20 mm\)') as raised:
        controller._move_to_reduced_configuration(home, deadline=float('inf'),
                                                  desc='Moving home')
    assert 'the IK has no solution' in str(raised.value)
    assert isinstance(raised.value.__cause__, NoSolutionFound)
    [logged] = controller._logger.errors
    first, second, distance = model.closest_collision_pairs(home)[0]
    assert f'{first} - {second} {distance * 1000:.1f} mm' in logged


def test_g1_start_inside_a_pair_says_self_collision(models):
    '''From arms_down the first IK step toward home has no solution: self-collision'''
    from golem_upper_ik.core.ik_solver import IKSolver
    from golem_upper_ik.core.robot_model import FixedState
    model = models('g1')
    named = model.profile.named_configs
    q = model.zero_q_body
    q[model.reduced_mask] = named['arms_down']
    try:
        model.state_subscriber = FixedState(q)
        solver = IKSolver(model, named, dt=0.002, d_min=0.02)
        solver.update_configurations(q)
        controller = controller_at(model, named['arms_down'],
                                   solver.goto_reduced_configuration)
        controller.ik_solver = solver
        with pytest.raises(RuntimeError, match='inside controller.d_min') as raised:
            controller._move_to_reduced_configuration(named['home'], deadline=float('inf'),
                                                      desc='Moving home')
    finally:
        model.state_subscriber = None
    assert 'the arms start too close' in str(raised.value)
    [logged] = controller._logger.errors
    # the hands and wrists are inside the hip-pitch spheres
    assert 'hip_pitch' in logged
    assert model.closest_collision_pairs(named['arms_down'])[0][2] < 0


def test_closest_pairs_are_sorted_and_counted(models):
    model = models('g1')
    pairs = model.closest_collision_pairs(model.profile.named_configs['home'], count=5)
    assert len(pairs) == 5
    distances = [distance for _, _, distance in pairs]
    assert distances == sorted(distances)
