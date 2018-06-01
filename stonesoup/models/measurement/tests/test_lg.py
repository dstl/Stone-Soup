# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.models.measurement.linear import LinearGaussian1D


def test_lgmodel1D():
    """ LinearGaussian1D Measurement Model test """

    # State related variables
    state_vec = np.array([[3.0], [1.0]])

    # Model-related components
    noise_covar = 0.1  # m/s^2
    H = np.array([[1, 0]])
    R = np.array([[noise_covar]])

    # Create and a Constant Velocity model object
    lg = LinearGaussian1D(ndim_state=2,
                          noise_covar=noise_covar,
                          mapping=0)

    # Ensure ```lg.transfer_function()``` returns H
    assert np.array_equal(H, lg.matrix())

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, lg.covar())

    # Project a state throught the model
    # (without noise)
    meas_pred_wo_noise = lg.function(state_vec, noise=0)
    assert np.array_equal(meas_pred_wo_noise, H@state_vec)

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = lg.pdf(meas_pred_wo_noise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_wo_noise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R).T)

    # Propagate a state vector throught the model
    # (with internal noise)
    meas_pred_w_inoise = lg.function(state_vec)
    assert not np.array_equal(meas_pred_w_inoise, H@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(meas_pred_w_inoise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_w_inoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R).T)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = lg.rvs()
    meas_pred_w_enoise = lg.function(state_vec,
                                     noise=noise)
    assert np.array_equal(meas_pred_w_enoise, H@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(meas_pred_w_enoise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_w_enoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R).T)
