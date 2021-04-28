# coding: utf-8

import numpy as np
import pytest

from ..categorical import CategoricalTransitionModel
from ....types.array import StateVectors, StateVector, Matrix, CovarianceMatrix
from ....types.state import State, CategoricalState


def create_random_multinomial(ndim):
    """Create a state with random state vector representing a categorical distribution across
    `ndim` possible categories."""
    total = 0
    sv = list()
    for i in range(ndim - 1):
        x = np.random.uniform(0, 1 - total)
        sv.append(x)
        total += x
    sv.append(1 - total)
    return CategoricalState(sv)


def test_basic_time_invariant():
    # 2 possible categories
    F = Matrix([[0.9, 0.1],
                [0.2, 0.8]])
    Q = CovarianceMatrix(np.eye(2))

    model = CategoricalTransitionModel(F, Q)

    # test ndim state
    assert model.ndim == F.shape[0]
    assert model.ndim_state == F.shape[0]

    # test function noise error
    with pytest.raises(ValueError, match="Noise is generated via random sampling, and defined "
                                         "noise is not implemented"):
        model.function(create_random_multinomial(2), noise=[0.5, 0.5])

    # test function
    for _ in range(3):

        state = create_random_multinomial(2)
        fp = F @ state.state_vector

        if any(fp == 1):
            exp_noiseless = fp * 1.0
            exp_noiseless[fp == 1] = np.finfo(np.float64).max
            exp_noiseless[fp == 0] = np.finfo(np.float64).min
        else:
            exp_noiseless = np.log(fp / (1 - fp)) + 0

        exp_noiseless = 1 / (1 + np.exp(-exp_noiseless))
        exp_noiseless = exp_noiseless / np.sum(exp_noiseless)

        actual_noiseless = model.function(state, noise=False)

        assert len(actual_noiseless) == 2  # model ndim
        assert np.isclose(sum(actual_noiseless), 1)  # test normalised
        assert np.array_equal(exp_noiseless, actual_noiseless)  # test expected

    # in case fp == 1 not hit
    id_model = CategoricalTransitionModel(np.eye(2), Q)
    state = State(StateVector([1, 0]))
    fp = np.eye(2) @ state.state_vector

    exp_noiseless = fp * 1.0
    exp_noiseless[fp == 1] = np.finfo(np.float64).max
    exp_noiseless[fp == 0] = np.finfo(np.float64).min

    exp_noiseless = 1 / (1 + np.exp(-exp_noiseless))
    exp_noiseless = exp_noiseless / np.sum(exp_noiseless)

    actual_noiseless = id_model.function(state, noise=False)

    assert len(actual_noiseless) == 2  # model ndim
    assert np.isclose(sum(actual_noiseless), 1)  # test normalised
    assert np.array_equal(exp_noiseless, actual_noiseless)  # test expected

    # test rvs
    for i in range(1, 4):
        rvs = model.rvs(num_samples=i)
        assert isinstance(rvs, StateVectors)
        assert len(rvs.T) == i
        for elem in rvs.T:
            assert len(elem) == 2  # same as model ndim

    # test pdf
    for _ in range(3):
        state1 = create_random_multinomial(2)
        state2 = create_random_multinomial(2)
        exp_value = (F @ state2.state_vector).T @ state1.state_vector
        actual_value = model.pdf(state1, state2)
        assert np.array_equal(actual_value, exp_value)
