# coding: utf-8

import datetime

import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.models.transitionmodel.linear import ConstantVelocity1D


def test_cvmodel1D():
    """ ConstanVelocity1D Transition Model test """

    # State related variables
    state_vec = np.array([[3.0], [1.0]])
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeff = 0.001  # m/s^2
    F = np.array([[1, timediff], [0, 1]])
    Q = np.array([[np.power(timediff, 3)/3,
                   np.power(timediff, 2)/2],
                  [np.power(timediff, 2)/2,
                   timediff]]) * noise_diff_coeff

    # Create and a Constant Velocity model object
    cv = ConstantVelocity1D(noise_diff_coeff=noise_diff_coeff)

    # Ensure ```cv.transfer_function(time_interval)``` returns F
    assert np.array_equal(F, cv.matrix(
        timestamp=new_timestamp, time_interval=time_interval))

    # Ensure ```cv.covar(time_interval)``` returns Q
    assert np.array_equal(Q, cv.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector throught the model
    # (without noise)
    new_state_vec_wo_noise = cv.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=0)
    assert np.array_equal(new_state_vec_wo_noise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = cv.pdf(new_state_vec_wo_noise,
                  state_vec,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert np.array_equal(prob, multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q).T)

    # Propagate a state vector throught the model
    # (with internal noise)
    new_state_vec_w_inoise = cv.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.array_equal(new_state_vec_w_inoise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = cv.pdf(new_state_vec_w_inoise,
                  state_vec,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert np.array_equal(prob, multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q).T)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = cv.random(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = cv.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.array_equal(new_state_vec_w_enoise, F@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = cv.pdf(new_state_vec_w_enoise, state_vec,
                  timestamp=new_timestamp, time_interval=time_interval)
    assert np.array_equal(prob, multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q).T)
