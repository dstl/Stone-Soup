import pytest
import numpy as np
from datetime import datetime, timedelta
import itertools as it
import copy

from ....types.state import StateVector, State, StateVectors
from ....platform import FixedPlatform
from ..move_position_action import GridActionGenerator, NStepDirectionalGridActionGenerator
from ...actionable_movable import GridActionableMovable


@pytest.mark.parametrize(
    'generator_params, state, position_mapping',
    [
        (
                {'n_steps': 1,
                 'step_size': 1,
                 'action_mapping': (0, 1),
                 'action_space': None,
                 'resolution': 1},
                StateVector([0., 0., 0.]),  # state
                (0, 1, 2)  # position_mapping
        ), (
                {'n_steps': 1,
                 'step_size': 1,
                 'action_mapping': (0, 1),
                 'action_space': None,
                 'resolution': 1},
                StateVector([0., 0.]),  # state
                (0, 1)  # position_mapping
        ), (
                {'n_steps': 1,
                 'step_size': 1,
                 'action_mapping': (1,),
                 'action_space': None,
                 'resolution': 1},
                StateVector([0., 0.]),  # state
                (0, 1)  # position_mapping
        ), (
                {'n_steps': 2,
                 'step_size': 1,
                 'action_mapping': (0, 1),
                 'action_space': None,
                 'resolution': 1},
                StateVector([0., 0., 0.]),  # state
                (0, 1, 2)  # position_mapping
        ), (
                {'n_steps': 2,
                 'step_size': 1,
                 'action_mapping': (0, 1),
                 'action_space': None,
                 'resolution': None},
                StateVector([0., 0., 0.]),  # state
                (0, 1, 2)  # position_mapping
        ), (
                {'n_steps': 2,
                 'step_size': 1,
                 'action_mapping': (0, 1),
                 'action_space': StateVectors([[0, 5], [-1, 5]]),
                 'resolution': 1},
                StateVector([0., 0., 0.]),  # state
                (0, 1, 2)  # position_mapping
        ), (
                {'n_steps': 2,
                 'step_size': 1,
                 'action_mapping': (0, 1),
                 'action_space': StateVectors([[0, 5], [-1, 5], [-1, 1]]),
                 'resolution': 1},
                StateVector([0., 0., 0.]),  # state
                (0, 1, 2)  # position_mapping
        )
    ],
    ids=["1_step_3D", "1_step_2D", "1_step_2D_with_1D_action", "2_step_3D",
         "2_step_2D_default_res", "2_step_2D_constrained", "2_step_3D_constrained",
         ]
)
def test_n_step_directional_grid_action_gen(generator_params, state, position_mapping):

    start_timestamp = datetime.now()
    end_timestamp = start_timestamp + timedelta(seconds=1)

    n_steps, step_size, action_mapping, action_space, resolution = \
        (generator_params.get(key) for key in ['n_steps',
                                               'step_size',
                                               'action_mapping',
                                               'action_space',
                                               'resolution'])

    if resolution is None:
        generator_params.pop('resolution')
        resolution = 1  # if none, it should use default

    platform = FixedPlatform(
        movement_controller=GridActionableMovable(
            states=[State(state, timestamp=start_timestamp)],
            position_mapping=position_mapping,
            generator=GridActionGenerator))  # Dummy platform for populating the owner category

    generator = NStepDirectionalGridActionGenerator(owner=platform,
                                                    attribute='position',
                                                    start_time=start_timestamp,
                                                    end_time=end_timestamp,
                                                    **generator_params)

    # Check that parameters have been set correctly
    assert generator.n_steps == n_steps
    assert generator.step_size == step_size
    assert generator.action_mapping == action_mapping
    assert np.all(generator.action_space == action_space)
    assert generator.resolution == resolution

    # Check that actions are generated correctly
    generator_set = set()
    generator_set.add(generator)
    move_position_actions = list(it.product(*generator_set))
    actions = []
    for elements in move_position_actions:
        actions.append(elements[0].target_value)

    deltas = np.linspace(-1*n_steps*step_size*resolution,
                         n_steps*step_size*resolution,
                         2*n_steps+1)

    eval_actions = [state]
    for dim in action_mapping:
        for delta in deltas:
            if delta == 0:
                continue
            eval_action = copy.copy(state)
            eval_action[dim] += delta
            if action_space is not None and (np.any(eval_action < action_space[:, 0]) or
                                             np.any(eval_action > action_space[:, 1])):
                continue
            else:
                eval_actions.append(eval_action)

    assert np.all(np.isclose(actions, eval_actions))
