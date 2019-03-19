# coding: utf-8

import datetime
import scipy as sp
from scipy.stats import multivariate_normal

from stonesoup.models.transition.linear import \
    (Singer, CombinedLinearGaussianTransitionModel)


def test_singer1dmodel():
    """ Singer 1D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1]])
    noise_diff_coeffs = sp.array([0.01])
    recips_decorr_times = sp.array([0.1])
    base(state_vec, noise_diff_coeffs, recips_decorr_times)


def test_singer2dmodel():
    """ Singer 2D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1],
                          [2.0], [2.0], [0.2]])
    noise_diff_coeffs = sp.array([0.01, 0.02])
    recips_decorr_times = sp.array([0.1, 0.1])

    base(state_vec, noise_diff_coeffs, recips_decorr_times)


def test_singer3dmodel():
    """ Singer 3D Transition Model test """
    state_vec = sp.array([[3.0], [1.0], [0.1],
                          [2.0], [2.0], [0.2],
                          [4.0], [0.5], [0.05]])
    noise_diff_coeffs = sp.array([0.01, 0.02, 0.005])
    recips_decorr_times = sp.array([0.1, 0.1, 0.1])
    base(state_vec, noise_diff_coeffs, recips_decorr_times)


def base(state_vec, noise_diff_coeffs, recips_decorr_times, timediff=1.0):
    """ Base test for n-dimensional ConstantAcceleration Transition Models """

    # Create a 1D Singer or an n-dimensional
    # CombinedLinearGaussianTransitionModel object
    dim = len(state_vec) // 3  # pos, vel, acc for each dimension
    if dim == 1:
        model_obj = Singer(noise_diff_coeff=noise_diff_coeffs[0],
                           recip_decorr_time=recips_decorr_times[0])
    else:
        model_list = [Singer(noise_diff_coeff=noise_diff_coeffs[i],
                             recip_decorr_time=recips_decorr_times[i])
                      for i in range(0, dim)]
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
        recip_decorr_time = recips_decorr_times[i]
        recip_decorr_timedt = recip_decorr_time * timediff
        noise_diff_coeff = noise_diff_coeffs[i]

        mat_list.append(sp.array(
            [[1,
              timediff,
              (recip_decorr_timedt - 1 + sp.exp(-recip_decorr_timedt)) /
              sp.power(recip_decorr_time, 2)],
             [0,
              1,
              (1 - sp.exp(-recip_decorr_timedt)) / recip_decorr_time],
             [0,
              0,
              sp.exp(-recip_decorr_timedt)]]))

        alpha_time = recip_decorr_time * timediff
        e_neg_at = sp.exp(-alpha_time)
        e_neg2_at = sp.exp(-2 * alpha_time)
        covar_list.append(sp.array(
            [[((1 - e_neg2_at) +
               2 * alpha_time +
               (2 * sp.power(alpha_time, 3)) / 3 -
               2 * sp.power(alpha_time, 2) -
               4 * alpha_time * e_neg_at) /
              (2 * sp.power(recip_decorr_time, 5)),
              sp.power(alpha_time - (1 - e_neg_at), 2) /
              (2 * sp.power(recip_decorr_time, 4)),
              ((1 - e_neg2_at) - 2 * alpha_time * e_neg_at) /
              (2 * sp.power(recip_decorr_time, 3))],
             [sp.power(alpha_time - (1 - e_neg_at), 2) /
              (2 * sp.power(recip_decorr_time, 4)),
              (2 * alpha_time - 4 * (1 - e_neg_at) + (1 - e_neg2_at)) /
              (2 * sp.power(recip_decorr_time, 3)),
              sp.power(1 - e_neg_at, 2) /
              (2 * sp.power(recip_decorr_time, 2))],
             [((1 - e_neg2_at) - 2 * alpha_time * e_neg_at) /
              (2 * sp.power(recip_decorr_time, 3)),
              sp.power(1 - e_neg_at, 2) /
              (2 * sp.power(recip_decorr_time, 2)),
              (1 - e_neg2_at) / (2 * recip_decorr_time)]]
        ) * noise_diff_coeff)

    F = sp.linalg.block_diag(*mat_list)
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
