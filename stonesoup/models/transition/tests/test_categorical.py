# coding: utf-8

import numpy as np
import pytest

from ..categorical import CategoricalTransitionModel
from ....types.array import StateVectors, CovarianceMatrix
from ....types.state import State, CategoricalState


def create_categorical(ndim):
    """Create a state with random state vector representing a categorical distribution across
    `ndim` possible categories."""
    total = 0
    sv = list()
    for i in range(ndim - 1):
        x = np.random.uniform(0, 1 - total)
        sv.append(x)
        total += x
    sv.append(1 - total)  # add probability that is left
    sv = np.array(sv)
    np.random.shuffle(sv)  # shuffle state vector
    return sv


def create_categorical_matrix(num_rows, num_cols):
    """Create a matrix with normalised rows"""
    matrix = list()
    for i in range(num_rows):
        matrix.append(create_categorical(num_cols))
    return np.array(matrix)


@pytest.mark.parametrize('ndim', (2, 3, 4))
def test_categorical_transition_model(ndim):
    # test invalid matrix
    with pytest.raises(ValueError, match="Column 0 of transition matrix does not sum to 1"):
        CategoricalTransitionModel(2 * np.eye(2))

    # ndim possible categories
    F = create_categorical_matrix(ndim, ndim).T  # normalised columns
    Q = CovarianceMatrix(np.eye(ndim))

    model = CategoricalTransitionModel(F, Q)

    # test ndim state
    assert model.ndim == F.shape[0]
    assert model.ndim_state == F.shape[0]

    # test function noise error
    with pytest.raises(ValueError, match="Noise is generated via random sampling, and defined "
                                         "noise is not implemented"):
        prior = CategoricalState(create_categorical(2))
        model.function(prior, noise=ndim * [0.5])

    states = [CategoricalState(create_categorical(ndim)) for _ in range(5)]

    # Test function
    for state in states:

        fp = F @ state.state_vector

        # Test noiseless

        if any(fp == 1):
            exp_noiseless = fp * 1.0
            exp_noiseless[fp == 1] = np.finfo(np.float64).max
            exp_noiseless[fp == 0] = np.finfo(np.float64).min
        else:
            exp_noiseless = np.log(fp / (1 - fp)) + 0

        exp_noiseless = 1 / (1 + np.exp(-exp_noiseless))
        exp_noiseless = exp_noiseless / np.sum(exp_noiseless)

        actual_noiseless = model.function(state, noise=False)

        assert len(actual_noiseless) == ndim
        assert np.isclose(sum(actual_noiseless), 1)  # test normalised
        assert np.array_equal(exp_noiseless, actual_noiseless)  # test is expected

        # Test noisy
        noisy = model.function(state, noise=True)
        assert len(noisy) == ndim
        assert np.isclose(sum(noisy), 1)  # test normalised

        # Test pdf
        other_state = CategoricalState(create_categorical(ndim))
        exp_value = (F @ other_state.state_vector).T @ state.state_vector
        actual_value = model.pdf(state, other_state)
        assert np.array_equal(actual_value, exp_value)

    # Test edge case
    id_model = CategoricalTransitionModel(np.eye(ndim), Q)
    state = State(np.eye(ndim)[0])  # vector looks like (1, 0, ..., 0) - ndim elements
    fp = np.eye(ndim) @ state.state_vector

    exp_noiseless = fp * 1.0
    exp_noiseless[fp == 1] = np.finfo(np.float64).max
    exp_noiseless[fp == 0] = np.finfo(np.float64).min

    exp_noiseless = 1 / (1 + np.exp(-exp_noiseless))
    exp_noiseless = exp_noiseless / np.sum(exp_noiseless)

    actual_noiseless = id_model.function(state, noise=False)

    assert len(actual_noiseless) == ndim
    assert np.isclose(sum(actual_noiseless), 1)  # test normalised
    assert np.array_equal(exp_noiseless, actual_noiseless)  # test is expected

    # test rvs
    for i in range(1, 4):
        rvs = model.rvs(num_samples=i)
        assert isinstance(rvs, StateVectors)
        assert len(rvs.T) == i
        for elem in rvs.T:
            assert len(elem) == ndim
