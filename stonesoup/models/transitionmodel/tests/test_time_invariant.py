# coding: utf-8
import numpy as np

from stonesoup.models.transitionmodel.linear import LinearGaussianTimeInvariant

def test_linear_gaussian():
    F = np.eye(3)
    Q = np.eye(3)
    model = LinearGaussianTimeInvariant(transition_matrix=F, covariance_matrix=Q)

    x_1 = np.ones([3,1])
    x_2 = F @ x_1

    assert np.array_equal(F.shape[0], model.ndim_state)
    assert np.array_equal(F, model.matrix())
    assert np.array_equal(Q, model.covar())
    assert np.array_equal(x_2, model.function(x_1, noise=np.zeros([3,1])))
    assert type(model.rvs()) is np.ndarray
    assert type(model.pdf(x_2, x_1)) is np.float64

test_linear_gaussian()
