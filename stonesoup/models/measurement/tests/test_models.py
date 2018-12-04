# -*- coding: utf-8 -*-
import pytest
import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.models.measurement.nonlinear \
    import RangeBearingGaussianToCartesian
from stonesoup.models.measurement.nonlinear \
    import RangeBearingElevationGaussianToCartesian
from stonesoup.models.measurement.nonlinear \
    import BearingElevationGaussianToCartesian
from stonesoup.functions import jacobian as compute_jac


def h2d(state_vector):
    x = state_vector[0][0]
    y = state_vector[1][0]

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return np.array([[phi], [rho]])


def h3d(state_vector):
    x = state_vector[0][0]
    y = state_vector[1][0]
    z = state_vector[2][0]

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)

    return np.array([[theta], [phi], [rho]])


def hbearing(state_vector):
    x = state_vector[0][0]
    y = state_vector[1][0]
    z = state_vector[2][0]

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)

    return np.array([[theta], [phi]])


@pytest.mark.parametrize(
    "h, ModelClass, state_vec, R , mapping",
    [
        (   # 2D meas, 2D state
            h2d,
            RangeBearingGaussianToCartesian,
            np.array([[0], [1]]),
            np.array([[0.015, 0],
                      [0, 0.1]]),
            np.array([0, 1]),
        ),
        (   # 3D meas, 3D state
            h3d,
            RangeBearingElevationGaussianToCartesian,
            np.array([[1], [2], [2]]),
            np.array([[0.05, 0, 0],
                      [0, 0.015, 0],
                      [0, 0, 0.1]]),
            np.array([0, 1, 2]),
        ),
        (   # 2D meas, 3D state
            hbearing,
            BearingElevationGaussianToCartesian,
            np.array([[1], [2], [3]]),
            np.array([[0.05, 0],
                      [0, 0.015]]),
            np.array([0, 1, 2]),
        )
    ],
    ids=["standard", "RBE", "BearingsOnly"]
)
def test_models(h, ModelClass, state_vec, R, mapping):
    """ RangeBearingGaussianToCartesian Measurement Model test """

    ndim_state = state_vec.size

    # Create and a measurement model object
    model = ModelClass(ndim_state=ndim_state,
                       mapping=mapping,
                       noise_covar=R)

    # Project a state throught the model
    # (without noise)
    meas_pred_wo_noise = model.function(state_vec, noise=0)
    assert np.array_equal(meas_pred_wo_noise, h(state_vec))

    # Ensure ```lg.transfer_function()``` returns H
    def fun(x):
        return model.function(x, noise=0)
    H = compute_jac(fun, state_vec)
    assert np.array_equal(H, model.jacobian(state_vec))
    # Check Jacobian has proper dimensions
    assert H.shape == (model.ndim_meas, ndim_state)

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

    # Propagate a state vector through the model
    # (with internal noise)
    meas_pred_w_inoise = model.function(state_vec)
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
