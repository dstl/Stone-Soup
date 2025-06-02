import pytest
import numpy as np
from datetime import datetime, timedelta
import itertools as it
import copy

from ....types.state import StateVector, State, StateVectors
from ....platform import FixedPlatform
from ...grid import NStepDirectionalGridMovable
from ...sample import CircleSampleActionableMovable


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
                 'action_mapping': (0, 1, 2),
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
        movement_controller=NStepDirectionalGridMovable(
            states=[State(state, timestamp=start_timestamp)],
            position_mapping=position_mapping,
            **generator_params))  # Dummy platform for initiating the generator

    generator = platform.actions(start_timestamp).pop()

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

            if action_space is None or \
                    (np.all(eval_action[action_mapping, :] >= action_space[:, [0]])
                     and np.all(eval_action[action_mapping, :] <= action_space[:, [1]])):
                eval_actions.append(eval_action)

    assert np.all(np.isclose(actions, eval_actions))

    platform.add_actions(move_position_actions[1])
    platform.act(end_timestamp)

    assert np.all(np.isclose(platform.position, actions[1]))


@pytest.mark.parametrize(
    'gen_param_dict, state, position_mapping',
    [
        (
            {'n_samples': None,
             'action_space': None,
             'action_mapping': (0, 1),
             'maximum_travel': 4},
            StateVector([0.0, 0.0]),  # state
            (0, 1)  # position_mapping
        ), (
            {'n_samples': 15,
             'action_space': StateVectors([[-5, 5], [-5, 5]]),
             'action_mapping': (0, 1),
             'maximum_travel': None},
            StateVector([0.0, 0.0]),  # state
            (0, 1)  # position_mapping
        ), (
            {'n_samples': 20,
             'action_space': StateVectors([[-10, 5], [-10, 5]]),
             'action_mapping': (0, 1),
             'maximum_travel': 10},
            StateVector([2.0, 3.0, 2.0]),  # state
            (0, 1)  # position_mapping
        ), (
            {'n_samples': 20,
             'action_space': StateVectors([[-5, 5], [-5, 5]]),
             'action_mapping': (0, 1),
             'maximum_travel': 10},
            StateVector([6.0, 6.0]),  # state
            (0, 1)  # position_mapping
        ), (
            {'n_samples': 20,
             'action_space': StateVectors([[-5, 5], [-5, 5], [-5, 5]]),
             'action_mapping': (0, 1),
             'maximum_travel': 10},
            StateVector([0.0, 0.0]),  # state
            (0, 1)  # position_mapping
        ), (
            {'n_samples': 20,
             'action_space': StateVectors([[-5, 5], [-5, 5], [-5, 5]]),
             'action_mapping': (0, 1, 2),
             'maximum_travel': 10},
            StateVector([0.0, 0.0, 0.0]),  # state
            (0, 1, 2)  # position_mapping
        ), (
            {'n_samples': 20,
             'action_space': StateVectors([[-5, 5], [-5, 5]]),
             'action_mapping': (0, 1),
             'maximum_travel': 10},
            StateVector([0.0, 0.0, 0.0]),  # state
            (0, 1, 2)  # position_mapping
        )
    ],

    ids=["defauls_n_samples_no_space", "default_travel_in_space", "max_travel_larger_than_space",
         "outside_space_start", "incompatible_action_space_and_mapping", "3d_action_value_error",
         "3d_position_2d_action"]
)
def test_circle_sample_action_gen(gen_param_dict, state, position_mapping):

    start_timestamp = datetime.now()
    end_timestamp = start_timestamp + timedelta(seconds=1)

    n_samples, action_space, action_mapping, maximum_travel = \
        (gen_param_dict.get(key) for key in ['n_samples',
                                             'action_space',
                                             'action_mapping',
                                             'maximum_travel'])

    if n_samples is None:
        gen_param_dict.pop('n_samples')
        n_samples = 10  # if none, it should use default

    if action_space is None:
        gen_param_dict.pop('action_space')

    if maximum_travel is None:
        gen_param_dict.pop('maximum_travel')
        maximum_travel = 1  # if non it should use default

    platform = \
        FixedPlatform(
            movement_controller=CircleSampleActionableMovable(
                states=[State(state, timestamp=start_timestamp)],
                position_mapping=position_mapping,
                **gen_param_dict))  # Dummy platform for initiating the generator

    # Test error raises for invalid action space, action mapping or incompatibility of the two
    if action_space is not None:
        if (len(action_space) != len(action_mapping)) or \
                (np.any(state[action_mapping, :] < action_space[:, [0]]) or
                 np.any(state[action_mapping, :] > action_space[:, [1]])) or \
                (len(action_mapping) != 2):

            with pytest.raises(ValueError):
                generator = platform.actions(start_timestamp).pop()
            return

    generator = platform.actions(start_timestamp).pop()

    # Check that parameters have been set correctly
    assert generator.n_samples == n_samples
    assert generator.action_mapping == action_mapping
    assert np.all(generator.action_space == action_space)
    assert generator.maximum_travel == maximum_travel

    # Test generator default action
    assert np.all(generator.default_action.target_value == state[position_mapping, :])

    # Check that actions are generated correctly
    generator_set = set()
    generator_set.add(generator)
    move_position_actions = list(it.product(*generator_set))

    # Check that the correct number of actions have been generated
    assert len(move_position_actions) == n_samples + 1

    # Verify each action is within range of the current platform position
    # and within the action space
    for action in move_position_actions:

        temp_platform = copy.deepcopy(platform)
        temp_platform.add_actions(action)
        temp_platform.act(end_timestamp)
        assert np.linalg.norm(temp_platform.position - platform.position,
                              axis=0) <= maximum_travel

        if action_space is not None:
            assert (temp_platform.position[0] > action_space[0, 0] and
                    temp_platform.position[0] < action_space[0, 1])
            assert (temp_platform.position[1] > action_space[1, 0] and
                    temp_platform.position[1] < action_space[1, 1])
