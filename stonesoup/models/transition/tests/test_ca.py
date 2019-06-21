# coding: utf-8

import datetime

import scipy as sp
from scipy.stats import multivariate_normal

from ..linear import (
    ConstantAcceleration, CombinedLinearGaussianTransitionModel)


def test_cam1dodel():
    """ ConstantAcceleration Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1]])
    noise_diff_coeffs = sp.array([[0.01]])
    base(state_vec, noise_diff_coeffs)


def test_ca2dmodel():
    """ ConstantAcceleration2D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1],
                          [2.0], [2.0], [0.2]])
    noise_diff_coeffs = sp.array([0.01, 0.02])
    base(state_vec, noise_diff_coeffs)


def test_ca3dmodel():
    """ ConstantAcceleration3D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1],
                          [2.0], [2.0], [0.2],
                          [4.0], [0.5], [0.05]])
    noise_diff_coeffs = sp.array([0.01, 0.02, 0.005])
    base(state_vec, noise_diff_coeffs)


def base(state_vec, noise_diff_coeffs):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create a 1D ConstantAcceleration or an n-dimensional
    # CombinedLinearGaussianTransitionModel object
    dim = len(state_vec) // 3  # pos, vel, acc for each dimension
    if dim == 1:
        model_obj = ConstantAcceleration(noise_diff_coeff=noise_diff_coeffs[0])
    else:
        model_list = [ConstantAcceleration(
            noise_diff_coeff=noise_diff_coeffs[i]) for i in range(0, dim)]
        model_obj = CombinedLinearGaussianTransitionModel(model_list)

    # State related variables
    state_vec = state_vec
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeffs = noise_diff_coeffs  # m/s^3
    base_mat = sp.array([[1, timediff, sp.power(timediff, 2) / 2],
                         [0, 1, timediff],
                         [0, 0, 1]])
    mat_list = [base_mat for num in range(0, dim)]
    F = sp.linalg.block_diag(*mat_list)

    base_covar = sp.array([[sp.power(timediff, 5) / 20,
                            sp.power(timediff, 4) / 8,
                            sp.power(timediff, 3) / 6],
                           [sp.power(timediff, 4) / 8,
                            sp.power(timediff, 3) / 3,
                            sp.power(timediff, 2) / 2],
                           [sp.power(timediff, 3) / 6,
                            sp.power(timediff, 2) / 2,
                            timediff]])
    covar_list = [base_covar*noise_diff_coeffs[i]
                  for i in range(0, dim)]
    Q = sp.linalg.block_diag(*covar_list)

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
