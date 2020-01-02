# coding: utf-8
import datetime

from pytest import approx
import scipy as sp
from scipy.stats import multivariate_normal

from ..linear import ConstantTurn
from ....types.state import State


def test_ctmodel():
    """ ConstantTurn Transition Model test """
    state = State(sp.array([[3.0], [1.0], [2.0], [1.0]]))
    noise_diff_coeffs = sp.array([0.01, 0.01])
    turn_rate = 0.1
    base(ConstantTurn, state, noise_diff_coeffs, turn_rate)


def base(model, state, noise_diff_coeffs, turn_rate):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create an ConstantTurn model object
    model = model
    model_obj = model(noise_diff_coeffs=noise_diff_coeffs, turn_rate=turn_rate)

    # State related variables
    state_vec = state.state_vector
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeffs = noise_diff_coeffs  # m/s^3
    turn_rate = turn_rate
    turn_ratedt = turn_rate*timediff
    F = sp.array(
            [[1, sp.sin(turn_ratedt) / turn_rate,
              0, -(1 - sp.cos(turn_ratedt)) / turn_rate],
             [0, sp.cos(turn_ratedt),
              0, -sp.sin(turn_ratedt)],
             [0, (1 - sp.cos(turn_ratedt)) / turn_rate,
              1, sp.sin(turn_ratedt) / turn_rate],
             [0, sp.sin(turn_ratedt),
              0, sp.cos(turn_ratedt)]])

    qx = noise_diff_coeffs[0]
    qy = noise_diff_coeffs[1]
    Q = sp.array([[sp.power(qx, 2) * sp.power(timediff, 3) / 3,
                   sp.power(qx, 2) * sp.power(timediff, 2) / 2,
                   0,
                   0],
                  [sp.power(qx, 2) * sp.power(timediff, 2) / 2,
                   sp.power(qx, 2) * timediff,
                   0,
                   0],
                  [0,
                   0,
                   sp.power(qy, 2) * sp.power(timediff, 3) / 3,
                   sp.power(qy, 2) * sp.power(timediff, 2) / 2],
                  [0,
                   0,
                   sp.power(qy, 2) * sp.power(timediff, 2) / 2,
                   sp.power(qy, 2) * timediff]])

    # Ensure ```model_obj.transfer_function(time_interval)``` returns F
    assert sp.array_equal(F, model_obj.matrix(
        timestamp=new_timestamp, time_interval=time_interval))

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    assert sp.array_equal(Q, model_obj.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector through the model
    # (without noise)
    new_state_vec_wo_noise = model_obj.function(
        state, noise=0,
        timestamp=new_timestamp,
        time_interval=time_interval)

    assert sp.array_equal(new_state_vec_wo_noise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = model_obj.pdf(State(new_state_vec_wo_noise),
                         state,
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=sp.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not sp.array_equal(new_state_vec_w_inoise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(State(new_state_vec_w_inoise),
                         state,
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=sp.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector through the model
    # (with external noise)
    noise = model_obj.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert sp.array_equal(new_state_vec_w_enoise, F@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(State(new_state_vec_w_enoise), state,
                         timestamp=new_timestamp, time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=sp.array(F@state_vec).ravel(),
        cov=Q)
