
# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.measurementmodel.base import LinearGaussian1D


def test_lgmodel1D():
    """ LinearGaussian1D Measurement Model test """

    # State related variables
    state_mean = np.array([[3.0], [1.0]])

    # Model-related components
    noise_var = 0.1  # m/s^2
    H = np.array([[1, 0]])
    R = noise_var

    # Create and a Constant Velocity model object
    lg = LinearGaussian1D(ndim_state=2,
                          noise_var=noise_var,
                          mapping=0)

    # Ensure ```lg.eval()``` returns H
    assert np.array_equal(H, lg.eval()),\
        "ERROR: Call to lg.eval() did not return H"

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, lg.covar()),\
        "ERROR: Call to lg.covar() did not return R"

    # Project a state vector throught the model
    # (without noise)
    meas_pred_mean_wo_noise = lg.eval(state_mean)
    assert np.array_equal(meas_pred_mean_wo_noise,
                          lg.eval()@state_mean), \
        "ERROR: Call to cv.eval(state_mean) did not return expected results!"

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = lg.pdf(meas_pred_mean_wo_noise, state_mean)
    test_mean = H.dot(state_mean[:, 0]).T
    assert np.array_equal(prob[0, 0], multivariate_normal.pdf(
        meas_pred_mean_wo_noise.T, mean=test_mean, cov=R).T)

    # Project a state vector throught the model
    # (with noise)
    meas_pred_mean_w_noise = lg.eval(state_mean)
    assert np.array_equal(meas_pred_mean_w_noise,
                          lg.eval()@state_mean), \
        "ERROR: Call to cv.eval(state_mean) did not return expected results!"

    # Evaluate the likelihood of the predicted measurement, given the state
    # (with noise)
    prob = lg.pdf(meas_pred_mean_w_noise, state_mean)
    Np1 = meas_pred_mean_w_noise.shape[1]
    Np2 = state_mean.shape[1]
    for i in range(0, Np1):
        for j in range(0, Np2):
            assert np.array_equal(
                prob[i, j],
                multivariate_normal.pdf(
                    meas_pred_mean_w_noise[:, i].T,
                    mean=meas_pred_mean_wo_noise[:, j].T,
                    cov=R
                ).T
            )
