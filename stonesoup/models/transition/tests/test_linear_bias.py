from numbers import Real

import numpy as np

from ..linear import LinearTransitionModel
from ....types.state import State
import pytest


def test_linear_transition():
    F = np.eye(3)
    Q = np.eye(3)
    bias = np.ones([3, 1])
    model = LinearTransitionModel(
        transition_matrix=F, bias_value=bias, covariance_matrix=Q)

    x_1 = np.ones([3, 1])
    x_2 = F @ x_1 + bias

    assert F.shape[0] == model.ndim_state
    assert np.array_equal(F, model.matrix())
    assert np.array_equal(Q, model.covar())
    assert np.array_equal(x_2, model.function(State(x_1),
                                              noise=np.zeros([3, 1])))
    assert isinstance(model.rvs(), np.ndarray)
    assert isinstance(model.pdf(State(x_2), State(x_1)), Real)

    model = LinearTransitionModel(transition_matrix=F, bias_value=bias, covariance_matrix=None)
    with pytest.raises(ValueError, match="Cannot generate rvs from None-type covariance"):
        model.rvs()
    with pytest.raises(ValueError, match="Cannot generate pdf from None-type covariance"):
        model.pdf(State([0]), State([0]))
