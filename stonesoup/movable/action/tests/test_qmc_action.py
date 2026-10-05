from datetime import datetime, timedelta

import numpy as np
from numpy.testing import assert_allclose

from ....platform import FixedPlatform
from ....types.state import State, StateVector, StateVectors
from ...sample import CircleSampleActionableMovable


def test_circle_sample_sobol_actions():
    start_time = datetime.now()
    state = StateVector([0., 0.])
    platform = FixedPlatform(
        movement_controller=CircleSampleActionableMovable(
            states=[State(state, timestamp=start_time)],
            position_mapping=(0, 1),
            n_samples=4,
            maximum_travel=2.,
            use_qmc=True,
        )
    )

    generator = platform.actions(start_time + timedelta(seconds=1)).pop()
    actions = [action.target_value for action in generator]

    assert generator.use_qmc
    assert len(actions) == 5
    assert_allclose(actions[0], state)

    # The origin of the Sobol sequence is skipped because the generator already
    # yields a default action that remains at the current position.
    sobol_samples = np.array([
        [0.5, 0.5],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.375, 0.375],
    ])
    radii = 2 * np.sqrt(sobol_samples[:, 0])
    angles = 2 * np.pi * sobol_samples[:, 1]
    expected = np.column_stack((radii * np.sin(angles), radii * np.cos(angles)))
    observed = np.array([np.asarray(action).ravel() for action in actions[1:]])

    assert_allclose(observed, expected, atol=1e-12)


def test_circle_sample_sobol_respects_action_space():
    start_time = datetime.now()
    action_space = StateVectors([[-1., 1.], [-1., 1.]])
    platform = FixedPlatform(
        movement_controller=CircleSampleActionableMovable(
            states=[State(StateVector([0., 0.]), timestamp=start_time)],
            position_mapping=(0, 1),
            n_samples=4,
            maximum_travel=2.,
            action_space=action_space,
            use_qmc=True,
        )
    )

    generator = platform.actions(start_time + timedelta(seconds=1)).pop()
    actions = [action.target_value for action in generator]
    observed = np.hstack(actions)

    assert len(actions) == 5
    assert np.all(observed >= action_space[:, [0]])
    assert np.all(observed <= action_space[:, [1]])


def test_circle_sample_uses_pseudorandom_sampling_by_default():
    start_time = datetime.now()
    platform = FixedPlatform(
        movement_controller=CircleSampleActionableMovable(
            states=[State(StateVector([0., 0.]), timestamp=start_time)],
            position_mapping=(0, 1),
        )
    )

    generator = platform.actions(start_time + timedelta(seconds=1)).pop()

    assert not generator.use_qmc
