# coding: utf-8
import datetime

import scipy as sp
from scipy.stats import multivariate_normal

from ..linear import ConstantTurnSandwich, ConstantVelocity


def test_ctmodel():
    """ ConstantTurnSandwich Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [2.0], [1.0], [1.0], [1.0]])
    turn_noise_diff_coeffs = sp.array([0.01, 0.01])
    turn_rate = 0.1
    noise_diff_cv = 0.1
    model_list = [ConstantVelocity(noise_diff_cv)]
    base(ConstantTurnSandwich, state_vec, turn_noise_diff_coeffs, turn_rate,
         noise_diff_cv, model_list)


def base(model, state_vec, turn_noise_diff_coeffs, turn_rate, noise_diff_cv,
         model_list):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create an ConstantTurn model object
    model = model
    model_obj = model(turn_noise_diff_coeffs=turn_noise_diff_coeffs,
                      turn_rate=turn_rate, model_list=model_list)

    # State related variables
    state_vec = state_vec
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    turn_noise_diff_coeffs = turn_noise_diff_coeffs  # m/s^3
    noise_diff_cv = noise_diff_cv
    turn_rate = turn_rate
    turn_ratedt = turn_rate*timediff
    F = sp.array(
            [[1, sp.sin(turn_ratedt) / turn_rate, 0, 0,
              0, -(1 - sp.cos(turn_ratedt)) / turn_rate],
             [0, sp.cos(turn_ratedt), 0, 0,
              0, -sp.sin(turn_ratedt)],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, (1 - sp.cos(turn_ratedt)) / turn_rate, 0, 0,
              1, sp.sin(turn_ratedt) / turn_rate],
             [0, sp.sin(turn_ratedt), 0, 0,
              0, sp.cos(turn_ratedt)]])
    q = noise_diff_cv
    qx = turn_noise_diff_coeffs[0]
    qy = turn_noise_diff_coeffs[1]
    Q = sp.array([[qx * sp.power(timediff, 3) / 3,
                   qx * sp.power(timediff, 2) / 2,
                   0, 0,
                   0, 0],
                  [qx * sp.power(timediff, 2) / 2,
                   qx * timediff,
                   0, 0,
                   0, 0],
                  [0,
                   0,
                   q * sp.power(timediff, 3) / 3,
                   q * sp.power(timediff, 2) / 2,
                   0,
                   0],
                  [0,
                   0,
                   q * sp.power(timediff, 2) / 2,
                   q * timediff,
                   0,
                   0],
                  [0, 0,
                   0, 0,
                   qy * sp.power(timediff, 3) / 3,
                   qy * sp.power(timediff, 2) / 2],
                  [0, 0,
                   0, 0,
                   qy * sp.power(timediff, 2) / 2,
                   qy * timediff]])

    # Ensure ```model_obj.transfer_function(time_interval)``` returns F
    assert sp.array_equal(F, model_obj.matrix(
        timestamp=new_timestamp, time_interval=time_interval))

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    assert sp.array_equal(Q, model_obj.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector throught the model
    # (without noise)
    new_state_vec_wo_noise = model_obj.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=0)
    assert sp.array_equal(new_state_vec_wo_noise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = model_obj.pdf(new_state_vec_wo_noise,
                         state_vec,
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert sp.array_equal(prob, multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=sp.array(F@state_vec).ravel(),
        cov=Q).T)

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not sp.array_equal(new_state_vec_w_inoise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(new_state_vec_w_inoise,
                         state_vec,
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert sp.array_equal(prob, multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=sp.array(F@state_vec).ravel(),
        cov=Q).T)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = model_obj.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = model_obj.function(
        state_vec,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert sp.array_equal(new_state_vec_w_enoise, F@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(new_state_vec_w_enoise, state_vec,
                         timestamp=new_timestamp, time_interval=time_interval)
    assert sp.array_equal(prob, multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=sp.array(F@state_vec).ravel(),
        cov=Q).T)
