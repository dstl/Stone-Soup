# coding: utf-8
import datetime

from pytest import approx
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

from ..linear import SingerApproximate, CombinedLinearGaussianTransitionModel
from ....types.state import State


def test_singer1dmodel_approximate():
    """ SingerApproximate 1D Transition Model test for small timediff """
    state = State(np.array([[3.0], [1.0], [0.1]]))
    noise_diff_coeffs = np.array([0.01])
    damping_coeffs = np.array([0.1])
    base(state, noise_diff_coeffs, damping_coeffs, timediff=0.4)


def test_singer2dmodel_approximate():
    """ SingerApproximate 2D Transition Model test for small timediff """
    state = State(np.array([[3.0], [1.0], [0.1], [2.0], [2.0], [0.2]]))
    noise_diff_coeffs = np.array([0.01, 0.02])
    damping_coeffs = np.array([0.1, 0.1])

    base(state, noise_diff_coeffs, damping_coeffs, timediff=0.4)


def test_singer3dmodel_approximate():
    """ SingerApproximate 3D Transition Model test for small timediff """
    state = State(np.array([[3.0], [1.0], [0.1], [2.0], [2.0], [0.2], [4.0],
                            [0.5], [0.05]]))
    noise_diff_coeffs = np.array([0.01, 0.02, 0.005])
    damping_coeffs = np.array([0.1, 0.1, 0.1])
    base(state, noise_diff_coeffs, damping_coeffs, timediff=0.4)


def base(state, noise_diff_coeffs, damping_coeffs, timediff=1.0):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    state_vec = state.state_vector

    # Create a 1D Singer or an n-dimensional
    # CombinedLinearGaussianTransitionModel object
    dim = len(state_vec) // 3  # pos, vel, acc for each dimension
    if dim == 1:
        model_obj = SingerApproximate(noise_diff_coeff=noise_diff_coeffs[0],
                                      damping_coeff=damping_coeffs[0])
    else:
        model_list = [
            SingerApproximate(
                noise_diff_coeff=noise_diff_coeffs[i],
                damping_coeff=damping_coeffs[i])
            for i in range(0, dim)
        ]
        model_obj = CombinedLinearGaussianTransitionModel(model_list)

    # State related variables
    state_vec = state_vec
    old_timestamp = datetime.datetime.now()
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeffs = noise_diff_coeffs  # m/s^3
    mat_list = []
    covar_list = []

    for i in range(0, dim):
        damping_coeff = damping_coeffs[i]
        damping_coeffdt = damping_coeff * timediff
        noise_diff_coeff = noise_diff_coeffs[i]

        mat_list.append(np.array(
            [[1,
              timediff,
              (damping_coeffdt - 1 + np.exp(-damping_coeffdt)) /
              damping_coeff**2],
             [0,
              1,
              (1 - np.exp(-damping_coeffdt)) / damping_coeff],
             [0,
              0,
              np.exp(-damping_coeffdt)]]))

        covar_list.append(np.array(
            [[timediff**5 / 20,
              timediff**4 / 8,
              timediff**3 / 6],
             [timediff**4 / 8,
              timediff**3 / 3,
              timediff**2 / 2],
             [timediff**3 / 6,
              timediff**2 / 2,
              timediff]]) * noise_diff_coeff)

    F = sp.linalg.block_diag(*mat_list)
    Q = sp.linalg.block_diag(*covar_list)

    # Ensure ```model_obj.transfer_function(time_interval)``` returns F
    assert np.allclose(F, model_obj.matrix(
        timestamp=new_timestamp, time_interval=time_interval), rtol=1e-10)

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    assert np.allclose(Q, model_obj.covar(
        timestamp=new_timestamp, time_interval=time_interval), rtol=1e-10)

    # Propagate a state vector through the model
    # (without noise)
    new_state_vec_wo_noise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert np.allclose(new_state_vec_wo_noise, F@state_vec, rtol=1e-10)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = model_obj.pdf(State(new_state_vec_wo_noise),
                         state,
                         timestamp=new_timestamp,
                         time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        state,
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval)
    assert not np.allclose(new_state_vec_w_inoise, F@state_vec, rtol=1e-10)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(State(new_state_vec_w_inoise),
                         state,
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
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise)
    assert np.allclose(new_state_vec_w_enoise, F@state_vec+noise, rtol=1e-10)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(State(new_state_vec_w_enoise), state,
                         timestamp=new_timestamp, time_interval=time_interval)
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F@state_vec).ravel(),
        cov=Q)
