# coding: utf-8

from datetime import datetime, timedelta

import numpy as np

from ..categorical import MarkovianTransitionModel
from ....types.array import StateVector
from ....types.state import CategoricalState


def test_categorical_transition_model():
    # 3 categories
    F = np.array([[50, 5, 30],
                  [25, 90, 30],
                  [25, 5, 30]])

    model = MarkovianTransitionModel(F)

    # Test normalised
    expected_array = np.array([[2 / 4, 1 / 20, 1 / 3],
                               [1 / 4, 18 / 20, 1 / 3],
                               [1 / 4, 1 / 20, 1 / 3]])
    assert np.allclose(model.transition_matrix, expected_array)

    # Test ndim
    assert model.ndim == 3
    assert model.ndim_state == 3

    state = CategoricalState([80, 10, 10], timestamp=datetime.now())

    # Test function (noiseless)
    new_vector = model.function(state, time_interval=timedelta(seconds=1), noise=False)
    assert isinstance(new_vector, StateVector)
    assert new_vector.shape[0] == 3
    assert np.isclose(np.sum(new_vector), 1)

    # Test function (noisy)
    new_vector = model.function(state, time_interval=timedelta(seconds=1), noise=True)
    assert isinstance(new_vector, StateVector)
    assert new_vector.shape[0] == 3
    assert np.count_nonzero(new_vector) == 1  # basis vector

    # Test 0 time-interval function
    new_vector = model.function(state, time_interval=timedelta(seconds=0))
    assert np.allclose(new_vector, state.state_vector)

    # Test no time-interval
    new_vector = model.function(state)
    assert np.allclose(new_vector, state.state_vector)
