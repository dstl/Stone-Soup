
# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.transitionmodel.base import ConstantVelocity1D


def test_cvmodel1D():
    """ ConstanVelocity1D Transition Model test """

    # State related variables
    state_mean = np.array([[3.0], [1.0]])

    # Model-related components
    time_variant = 1  # sec
    noise_diff_coeff = 0.001  # m/s^2
    F = np.array([[1, time_variant], [0, 1]])
    Q = np.array([[np.power(time_variant, 3)/3, np.power(time_variant, 2)/2],
                  [np.power(time_variant, 2)/2, time_variant]])\
        * noise_diff_coeff

    # Create and a Constant Velocity model object
    cv = ConstantVelocity1D(noise_diff_coeff=noise_diff_coeff,
                            time_variant=time_variant)

    # Ensure ```cv.eval()``` returns F
    assert np.array_equal(F, cv.eval()),\
        "ERROR: Call to cv.eval() did not return F"

    # Ensure ```cv.covar()``` returns Q
    assert np.array_equal(Q, cv.covar()),\
        "ERROR: Call to cv.covar() did not return Q"

    # Propagate a state vector throught the model
    # (without noise)
    new_state_mean_wo_noise = cv.eval(state_mean)
    assert np.array_equal(new_state_mean_wo_noise,
                          cv.eval()@state_mean), \
        "ERROR: Call to cv.eval(state_mean) did not return expected results!"

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = cv.pdf(new_state_mean_wo_noise, state_mean)
    test_mean = F.dot(state_mean[:, 0]).T
    assert np.array_equal(prob[0, 0], multivariate_normal.pdf(
        new_state_mean_wo_noise.T, mean=test_mean, cov=Q).T)

    # Propagate a state vector throught the model
    # (with noise)
    new_state_mean_w_noise = cv.eval(state_mean)
    assert np.array_equal(new_state_mean_w_noise,
                          cv.eval()@state_mean), \
        "ERROR: Call to cv.eval(state_mean) did not return expected results!"

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = cv.pdf(new_state_mean_w_noise, state_mean)
    Np1 = new_state_mean_w_noise.shape[1]
    Np2 = state_mean.shape[1]
    for i in range(0, Np1):
        for j in range(0, Np2):
            assert np.array_equal(
                prob[i, j],
                multivariate_normal.pdf(
                    new_state_mean_w_noise[:, i].T,
                    mean=new_state_mean_wo_noise[:, j].T,
                    cov=Q
                ).T
            )
