# coding: utf-8
import datetime

from pytest import approx
import numpy as np
from scipy.stats import multivariate_normal

from ..linear import RandomWalk


def test_rwodel():
    """ RandomWalk Transition Model test """

    # State related variables
    state_vec = np.array([[3.0]])
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeff = 0.001  # m/s^2
    F = np.array([[1]])
    Q = np.array([[timediff]]) * noise_diff_coeff

    # Create and a Random Walk model object
    rw = RandomWalk(noise_diff_coeff=noise_diff_coeff)

    # Ensure ```rw.transfer_function(time_interval)``` returns F
    assert np.array_equal(F, rw.matrix(
        timestamp=new_timestamp, time_interval=time_interval))

    # Ensure ```rw.covar(time_interval)``` returns Q
    assert np.array_equal(Q, rw.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector through the model
    # (without noise)
    new_state_vec_wo_noise = rw.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=0)
    assert np.array_equal(new_state_vec_wo_noise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = rw.pdf(new_state_vec_wo_noise,
                  state_vec,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector throught the model
    # (with internal noise)
    new_state_vec_w_inoise = rw.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.array_equal(new_state_vec_w_inoise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = rw.pdf(new_state_vec_w_inoise,
                  state_vec,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = rw.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = rw.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.array_equal(new_state_vec_w_enoise, F@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = rw.pdf(new_state_vec_w_enoise, state_vec,
                  timestamp=new_timestamp, time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)
