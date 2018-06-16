# coding: utf-8

import datetime

import scipy as sp
from scipy.stats import multivariate_normal

from stonesoup.models.transitionmodel.linear import ConstantTurn


def test_ctmodel():
    """ ConstantTurn Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [2.0], [1.0]])
    noise_diff_coeffs = sp.array([[0.01], [0.01]])
    omega = 0.1
    base(ConstantTurn, state_vec, noise_diff_coeffs, omega)


def base(model, state_vec, noise_diff_coeffs, omega):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create and a ConstantTurn model object
    model = model
    model_obj = model(noise_diff_coeffs=noise_diff_coeffs, omega=omega)

    # State related variables
    state_vec = state_vec
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeffs = noise_diff_coeffs  # m/s^3
    omega = omega
    omegadt = omega*timediff
    F = sp.array(
            [[1, sp.sin(omegadt) / omega,
              0, -(1 - sp.cos(omegadt)) / omega],
             [0, sp.cos(omegadt),
              0, -sp.sin(omegadt)],
             [0, (1 - sp.cos(omegadt)) / omega,
              1, sp.sin(omegadt) / omega],
             [0, sp.sin(omegadt),
              0, sp.cos(omegadt)]])
    
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
