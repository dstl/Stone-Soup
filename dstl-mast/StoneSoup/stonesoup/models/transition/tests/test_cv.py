import datetime

from pytest import approx
import numpy as np
from scipy.stats import multivariate_normal

from ..linear import ConstantVelocity
from ....types.state import State


def test_cvmodel():
    """ ConstanVelocity Transition Model test """

    # State related variables
    state = State(np.array([[3.0], [1.0]]))
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeff = 0.001  # m/s^2
    F = np.array([[1, timediff], [0, 1]])
    Q = np.array([[timediff**3 / 3,
                   timediff**2 / 2],
                  [timediff**2 / 2,
                   timediff]]) * noise_diff_coeff

    # Create and a Constant Velocity model object
    cv = ConstantVelocity(noise_diff_coeff=noise_diff_coeff)

    # Ensure ```cv.transfer_function(time_interval)``` returns F
    assert np.array_equal(F, cv.matrix(
        timestamp=new_timestamp, time_interval=time_interval))

    # Ensure ```cv.covar(time_interval)``` returns Q
    assert np.array_equal(Q, cv.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector through the model
    # (without noise)
    new_state_vec_wo_noise = cv.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert np.array_equal(new_state_vec_wo_noise, F@state.state_vector)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = cv.pdf(State(new_state_vec_wo_noise),
                  state,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F@state.state_vector).ravel(),
        cov=Q)

    # Propagate a state vector throught the model
    # (with internal noise)
    new_state_vec_w_inoise = cv.function(
        state,
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.array_equal(new_state_vec_w_inoise, F@state.state_vector)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = cv.pdf(State(new_state_vec_w_inoise),
                  state,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=np.array(F@state.state_vector).ravel(),
        cov=Q)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = cv.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = cv.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.array_equal(new_state_vec_w_enoise, F@state.state_vector+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = cv.pdf(State(new_state_vec_w_enoise), state,
                  timestamp=new_timestamp, time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F@state.state_vector).ravel(),
        cov=Q)
