# coding: utf-8
import pytest
import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.models.measurement.nonlinear import Polar2CartesianGaussian
from stonesoup.functions import jacobian as compute_jac


def h(state_vector):
    x = state_vector[0]
    y = state_vector[1]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([[phi], [rho]])


@pytest.mark.parametrize(
    "h, R, ndim_state, mapping",
    [
        (   # 1D meas, 2D state
            h,
            np.array([[0.015, 0],
                      [0, 0.1]]),
            2,
            np.array([0, 1]),
        )
    ],
    ids=["standard"]
)
def test_p2cgmodel(h, R, ndim_state, mapping):
    """ Polar2CartGaussian Measurement Model test """

    # State related variables
    state_vec = np.array([[0], [1]])

    # Create and a measurement model object
    model = Polar2CartesianGaussian(ndim_state=ndim_state,
                                    mapping=mapping,
                                    noise_covar=R)

    # Project a state throught the model
    # (without noise)
    meas_pred_wo_noise = model.function(state_vec, noise=0)
    assert np.array_equal(meas_pred_wo_noise, h(state_vec))

    # Ensure ```lg.transfer_function()``` returns H
    def fun(x):
        return model.function(state_vec, noise=0)
    H = compute_jac(fun, state_vec)
    assert np.array_equal(H, model.jacobian(state_vec))

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, model.covar())

    # Project a state throught the model
    # (without noise)
    meas_pred_wo_noise = model.function(state_vec, noise=0)
    assert np.array_equal(meas_pred_wo_noise, h(state_vec))

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = model.pdf(meas_pred_wo_noise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_wo_noise.T,
        mean=np.array(h(state_vec)).ravel(),
        cov=R).T)

    # Propagate a state vector throught the model
    # (with internal noise)
    meas_pred_w_inoise = model.function(state_vec)
    print(meas_pred_w_inoise)
    assert not np.array_equal(meas_pred_w_inoise, h(state_vec))

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(meas_pred_w_inoise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_w_inoise.T,
        mean=np.array(h(state_vec)).ravel(),
        cov=R).T)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = model.rvs()
    meas_pred_w_enoise = model.function(state_vec,
                                        noise=noise)
    assert np.array_equal(meas_pred_w_enoise, h(state_vec)+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model.pdf(meas_pred_w_enoise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_w_enoise.T,
        mean=np.array(h(state_vec)).ravel(),
        cov=R).T)
