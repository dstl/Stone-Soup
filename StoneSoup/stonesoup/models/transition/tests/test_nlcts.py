import datetime
import copy

import numpy as np

from stonesoup.models.transition.nonlinear import ConstantTurnSandwich
from stonesoup.models.transition.base import CombinedGaussianTransitionModel
from stonesoup.models.transition.linear import ConstantVelocity, ConstantAcceleration
from stonesoup.types.state import State


def test_ctsmodel(CT_model):
    """ ConstantTurn Transition Model test """
    state = State(np.array([[3.0], [1.0], [10.], [-1.], [-8.], [-1.], [0.5], [2.0], [1.0],
                            [-0.05]]))
    linear_noise_coeffs = np.array([0.1, 0.1])
    turn_noise_coeff = 0.01
    base(CT_model, ConstantTurnSandwich, state, linear_noise_coeffs, turn_noise_coeff)


def base(ct_model, model, state, linear_noise_coeffs, turn_noise_coeff):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """
    cv_coeff = 1.1
    ca_coeff = 0.4

    cv_model = ConstantVelocity(cv_coeff)
    ca_model = ConstantAcceleration(ca_coeff)

    model_list = [cv_model, ca_model]
    comb_model = CombinedGaussianTransitionModel(model_list=model_list)
    # Create an ConstantTurnSandwich model object
    model = model
    model_obj = model(linear_noise_coeffs=linear_noise_coeffs,
                      turn_noise_coeff=turn_noise_coeff, model_list=model_list)

    # State related variables
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = datetime.timedelta(seconds=timediff)
    state2 = copy.deepcopy(state)

    # Model-related components
    # Calculate indices for CT component - needed to extract ct elements from the sandwich
    idx = [0, 1]
    tmp = state.state_vector.shape[0] - 3
    idx.extend(list(range(tmp, tmp + 3)))

    state2.state_vector = state.state_vector[idx, :]
    state_out1 = ct_model.function(state2, time_interval=time_interval)
    state2.state_vector = state.state_vector[2:-3, :]
    state_out2 = comb_model.function(state2, time_interval=time_interval)

    # Propagate a state vector through the model (without noise)
    model_out = model_obj.function(state, time_interval=time_interval)
    assert np.array_equal(model_out[idx, :], state_out1)
    assert np.array_equal(model_out[2:-3, :], state_out2)

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    # Only checking block diagonal components
    Q1 = ct_model.covar(linear_noise_coeffs, turn_noise_coeff, time_interval)
    Q2 = comb_model.covar(timestamp=new_timestamp, time_interval=time_interval)
    model_Q = model_obj.covar(timestamp=new_timestamp, time_interval=time_interval)
    assert np.array_equal(model_Q[0:2, 0:2], Q1[0:2, 0:2])
    assert np.array_equal(model_Q[-3:, -3:], Q1[-3:, -3:])
    assert np.array_equal(model_Q[2:-3, 2:-3], Q2)

    # Eliminated the pdf based tests since for nonlinear models these will no
    # longer be Gaussian

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        state,
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.array_equal(new_state_vec_w_inoise, model_out)

    # Propagate a state vector through the model (with external noise)
    noise = model_obj.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.array_equal(new_state_vec_w_enoise, model_out + noise)
