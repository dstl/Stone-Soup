# coding: utf-8
from numbers import Real

import numpy as np
import pytest

from ..classification import BasicTimeInvariantClassificationTransitionModel
from ..linear import LinearGaussianTimeInvariantTransitionModel
from ....measures import ObservationAccuracy
from ....types.state import State


def test_linear_gaussian():
    F = np.eye(3)
    Q = np.eye(3)
    model = LinearGaussianTimeInvariantTransitionModel(
        transition_matrix=F, covariance_matrix=Q)

    x_1 = np.ones([3, 1])
    x_2 = F @ x_1

    assert F.shape[0] == model.ndim_state
    assert np.array_equal(F, model.matrix())
    assert np.array_equal(Q, model.covar())
    assert np.array_equal(x_2, model.function(State(x_1),
                                              noise=np.zeros([3, 1])))
    assert isinstance(model.rvs(), np.ndarray)
    assert isinstance(model.pdf(State(x_2), State(x_1)), Real)

    model = LinearGaussianTimeInvariantTransitionModel(transition_matrix=F, covariance_matrix=None)
    with pytest.raises(ValueError, match="Cannot generate rvs from None-type covariance"):
        model.rvs()
    with pytest.raises(ValueError, match="Cannot generate pdf from None-type covariance"):
        model.pdf(State([0]), State([0]))


def create_random_multinomial(ndim):
    total = 0
    sv = list()
    for i in range(ndim - 1):
        x = np.random.uniform(0, 1 - total)
        sv.append(x)
        total += x
    sv.append(1 - total)
    return State(sv)


def test_basic_time_invariant():
    F = np.array([[0.9, 0.1],
                  [0.2, 0.8]])
    Q = np.eye(2)

    model = BasicTimeInvariantClassificationTransitionModel(F, Q)

    # test ndim
    assert model.ndim == F.shape[0]
    assert model.ndim_state == F.shape[0]

    # test function
    for _ in range(3):
        state = create_random_multinomial(2)
        Fx = F @ state.state_vector
        noiseless_expected = Fx / sum(Fx)
        noise = Q @ Fx
        noisy_expected = (Fx + noise) / sum(Fx + noise)

        noiseless_result = model.function(state, noise=False)
        noisy_result = model.function(state, noise=True)

        # test expected value
        assert np.array_equal(noiseless_expected, noiseless_result)
        assert np.array_equal(noisy_expected, noisy_result)

        # test normalised
        assert np.isclose(sum(noiseless_result), 1)
        assert np.isclose(sum(noisy_result), 1)

    # test missing noise
    with pytest.raises(AttributeError,
                       match="Require a defined transition noise matrix to generate noise"):
        noiseless_model = BasicTimeInvariantClassificationTransitionModel(F)
        noiseless_model.function(create_random_multinomial(2), noise=True)

    # test pdf
    measure = ObservationAccuracy()
    for _ in range(3):
        state1 = create_random_multinomial(2)
        state2 = create_random_multinomial(2)
        assert model.pdf(state1, state2) == measure(state1, state2)

    # test rvs
    with pytest.raises(NotImplementedError,
                       match="Noise generation for classification-based state transitions is not "
                             "implemented"):
        model.rvs()
