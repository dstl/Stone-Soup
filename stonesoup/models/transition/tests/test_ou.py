import datetime

from pytest import approx
import pytest
import numpy as np
from scipy.stats import multivariate_normal

from stonesoup.models.transition.linear import OrnsteinUhlenbeck
from ....types.state import State


@pytest.mark.parametrize('sign', [1, -1])
def test_oumodel(sign):
    """ OrnsteinUhlenbeck Transition Model test """

    # State related variables
    state = State(np.array([[3.0], [1.0]]))
    old_timestamp = datetime.datetime.now()
    timediff = 1 * sign  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    q = 0.001  # m/s^2
    k = 0.1
    dt = time_interval.total_seconds()

    exp_kdt = np.exp(-k*dt)
    exp_2kdt = np.exp(-2*k*dt)

    F = np.array([[1, (1 - exp_kdt)/k],
                  [0, exp_kdt]])

    q11 = q/k ** 2*(dt - 2/k*(1 - exp_kdt)
                    + 1/(2*k)*(1 - exp_2kdt)) * sign
    q12 = q/k*((1 - exp_kdt)/k
               - 1/(2*k)*(1 - exp_2kdt))
    q22 = q/(2*k)*(1 - exp_2kdt) * sign

    Q = np.array([[q11, q12],
                  [q12, q22]])

    # Create and a Constant Velocity model object
    ou = OrnsteinUhlenbeck(noise_diff_coeff=q, damping_coeff=k)

    # Ensure ```ou.transfer_function(time_interval)``` returns F
    assert np.allclose(F, ou.matrix(
        timestamp=new_timestamp, time_interval=time_interval), rtol=1e-10)
    # Ensure ```ou.covar(time_interval)``` returns Q
    assert np.allclose(Q, ou.covar(
        timestamp=new_timestamp, time_interval=time_interval), rtol=1e-10)

    # Propagate a state vector throught the model
    # (without noise)
    new_state_vec_wo_noise = ou.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert np.allclose(new_state_vec_wo_noise, F @ state.state_vector,
                       rtol=1e-10)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = ou.pdf(State(new_state_vec_wo_noise),
                  state,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F @ state.state_vector).ravel(),
        cov=Q)

    # Propagate a state vector through the model
    # (with internal noise)
    new_state_vec_w_inoise = ou.function(
        state,
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.allclose(new_state_vec_w_inoise, F @ state.state_vector,
                           rtol=1e-10)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = ou.pdf(State(new_state_vec_w_inoise),
                  state,
                  timestamp=new_timestamp,
                  time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=np.array(F @ state.state_vector).ravel(),
        cov=Q)

    # Propagate a state vector through the model
    # (with external noise)
    noise = ou.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = ou.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.allclose(new_state_vec_w_enoise, F @ state.state_vector + noise,
                       rtol=1e-10)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = ou.pdf(State(new_state_vec_w_enoise), state,
                  timestamp=new_timestamp, time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F @ state.state_vector).ravel(),
        cov=Q)
