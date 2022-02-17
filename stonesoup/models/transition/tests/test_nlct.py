# coding: utf-8
import datetime

import numpy as np
from ..nonlinear import ConstantTurn
from ....types.state import State


def test_ctmodel(CT_model):
    """ ConstantTurn Transition Model test """
    state = State(np.array([[3.0], [1.0], [2.0], [1.0], [-0.05]]))
    linear_noise_coeffs = np.array([0.1, 0.1])
    turn_noise_coeff = 0.01
    base(CT_model, ConstantTurn, state, linear_noise_coeffs, turn_noise_coeff)


def base(CT_model, model, state, linear_noise_coeffs, turn_noise_coeff):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create an ConstantTurn model object
    model = model
    model_obj = model(linear_noise_coeffs=linear_noise_coeffs,
                      turn_noise_coeff=turn_noise_coeff)

    # State related variables
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = datetime.timedelta(seconds=timediff)

    # Model-related components
    F = CT_model.function(state, time_interval=time_interval)

    Q = CT_model.covar(linear_noise_coeffs, turn_noise_coeff, time_interval)

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    assert np.array_equal(Q, model_obj.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector through the mode (without noise)
    new_state_vec_wo_noise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval)

    assert np.array_equal(new_state_vec_wo_noise, F)

    # Eliminated the pdf based tests since for nonlinear models these will no
    # longer be Gaussian

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        state,
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.array_equal(new_state_vec_w_inoise, F)

    # Propagate a state vector through the model
    # (with external noise)
    noise = model_obj.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.array_equal(new_state_vec_w_enoise, F + noise)
