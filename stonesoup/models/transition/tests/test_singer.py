# coding: utf-8

import datetime

import scipy as sp
from scipy.stats import multivariate_normal

from stonesoup.models.transition.linear import \
    (SingerModel1D, SingerModel2D, SingerModel3D)


def test_singer1dmodel():
    """ ConstantAcceleration1D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1]])
    noise_diff_coeffs = sp.array([[0.01]])
    alphas = sp.array([[0.1]])
    base(SingerModel1D, state_vec, noise_diff_coeffs, alphas)


def test_singer2dmodel():
    """ ConstantAcceleration2D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1],
                          [2.0], [2.0], [0.2]])
    noise_diff_coeffs = sp.array([[0.01], [0.02]])
    alphas = sp.array([[0.1], [0.1]])
    base(SingerModel2D, state_vec, noise_diff_coeffs, alphas)


def test_singer3dmodel():
    """ ConstantAcceleration3D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1],
                          [2.0], [2.0], [0.2],
                          [4.0], [0.5], [0.05]])
    noise_diff_coeffs = sp.array([[0.01], [0.02], [0.005]])
    alphas = sp.array([[0.1], [0.1], [0.1]])
    base(SingerModel3D, state_vec, noise_diff_coeffs, alphas)


def base(model, state_vec, noise_diff_coeffs, alphas):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create and an arbitrary dimension ConstantAcceleration model object
    model = model
    model_obj = model(noise_diff_coeffs=noise_diff_coeffs, alphas=alphas)
    dimension = model_obj.ndim_state // 3 # pos, vel, acc for each dimension

    # State related variables
    state_vec = state_vec
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    noise_diff_coeffs = noise_diff_coeffs  # m/s^3
    mat_list = []

    for i in range(0, dimension):
        alpha = alphas[i]
        alphadt = alpha * timediff
        mat_list.append(sp.array(
            [[1,
              timediff,
              (alphadt - 1 + sp.exp(-alphadt)) / sp.power(alpha, 2)],
             [0,
              1,
              (1 - sp.exp(-alphadt)) / alpha],
             [0,
              0,
              sp.exp(-alphadt)]]))
    F = sp.linalg.block_diag(*mat_list)

    covar_list = []
    for i in range(0, dimension):
        alpha = alphas[i]
        noise_diff_coeff = noise_diff_coeffs[i]
        constant_multiplier = 2 * alpha * sp.power(noise_diff_coeff, 2)

        covar_list.append(sp.array(
            [[sp.power(timediff, 5) / 20,
              sp.power(timediff, 4) / 8,
              sp.power(timediff, 3) / 6],
             [sp.power(timediff, 4) / 8,
              sp.power(timediff, 3) / 3,
              sp.power(timediff, 2) / 2],
             [sp.power(timediff, 3) / 6,
              sp.power(timediff, 2) / 2,
              timediff]]) * constant_multiplier)
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
