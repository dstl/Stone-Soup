import datetime

from pytest import approx
import pytest
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal
from ....types.state import State

from ..linear import ConstantAcceleration
from ..base import CombinedGaussianTransitionModel


@pytest.fixture(params=[1, 2, 3])
def ca_model_params(request):
    if request.param == 1:
        state_vec = np.array([[3.0], [1.0], [0.1]])
        noise_diff_coeffs = np.array([[0.01]])
    elif request.param == 2:
        state_vec = np.array([[3.0], [1.0], [0.1],
                              [2.0], [2.0], [0.2]])
        noise_diff_coeffs = np.array([0.01, 0.02])
    else:
        state_vec = np.array([[3.0], [1.0], [0.1],
                              [2.0], [2.0], [0.2],
                              [4.0], [0.5], [0.05]])
        noise_diff_coeffs = np.array([0.01, 0.02, 0.005])
    return state_vec, noise_diff_coeffs


@pytest.mark.parametrize('sign', [1, -1])
def test_ca(ca_model_params, sign):
    state_vec, noise_diff_coeffs = ca_model_params
    # Create a 1D ConstantAcceleration or an n-dimensional
    # CombinedLinearGaussianTransitionModel object
    dim = len(state_vec) // 3  # pos, vel, acc for each dimension
    if dim == 1:
        model_obj = ConstantAcceleration(noise_diff_coeff=noise_diff_coeffs[0])
    else:
        model_list = [ConstantAcceleration(
            noise_diff_coeff=noise_diff_coeffs[i]) for i in range(0, dim)]
        model_obj = CombinedGaussianTransitionModel(model_list)

    # State related variables
    old_timestamp = datetime.datetime.now()
    timediff = 1 * sign  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeffs = noise_diff_coeffs  # m/s^3
    base_mat = np.array([[1, timediff, timediff**2 / 2],
                         [0, 1, timediff],
                         [0, 0, 1]])
    mat_list = [base_mat for num in range(0, dim)]
    F = sp.linalg.block_diag(*mat_list)

    base_covar = np.array([[abs(timediff)**5 / 20,
                            abs(timediff)**4 / 8,
                            abs(timediff)**3 / 6],
                           [abs(timediff)**4 / 8,
                            abs(timediff)**3 / 3,
                            abs(timediff)**2 / 2],
                           [abs(timediff)**3 / 6,
                            abs(timediff)**2 / 2,
                            abs(timediff)]])
    covar_list = [base_covar*noise_diff_coeffs[i]
                  for i in range(0, dim)]
    Q = sp.linalg.block_diag(*covar_list)

    # Ensure ```model_obj.transfer_function(time_interval)``` returns F
    assert F == approx(model_obj.jacobian(State(state_vec),
                                          time_interval=time_interval))

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    assert np.array_equal(Q, model_obj.covar(
        timestamp=new_timestamp, time_interval=time_interval))

    # Propagate a state vector through the model
    # (without noise)
    new_state_vec_wo_noise = model_obj.function(
        State(state_vec),
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert np.allclose(new_state_vec_wo_noise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = model_obj.pdf(State(new_state_vec_wo_noise),
                         State(state_vec),
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        State(state_vec),
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.allclose(new_state_vec_w_inoise, F@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(State(new_state_vec_w_inoise),
                         State(state_vec),
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector through the model
    # (with external noise)
    noise = model_obj.rvs(timestamp=new_timestamp, time_interval=time_interval)
    new_state_vec_w_enoise = model_obj.function(
        State(state_vec),
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.allclose(new_state_vec_w_enoise, F@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(State(new_state_vec_w_enoise), State(state_vec),
                         timestamp=new_timestamp, time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)
