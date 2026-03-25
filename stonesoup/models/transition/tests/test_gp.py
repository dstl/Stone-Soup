import datetime

from pytest import approx
import pytest
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal

from ..linear import MarkovianGP
from ..base import CombinedGaussianTransitionModel
from ....types.state import State
from ....types.track import Track

import inspect


def se_kernel(ti, tj, length_scale, variance):
    """Helper for unit tests: single squared exponential kernel element k(ti, tj)."""
    return variance * np.exp(-0.5 * (ti - tj) ** 2 / (length_scale**2))


@pytest.fixture(params=[1])
def gp_model_params(request):
    if request.param == 1:
        state = State(np.array([[3.0], [1.0], [0.1]]))
        length_scales = np.array([1])
        variances = np.array([0.1])
    elif request.param == 2:
        state = State(np.array([[3.0], [1.0], [0.1], [2.0], [2.0], [0.2]]))
        length_scales = np.array([0.01, 0.02])
        variances = np.array([0.1, 0.1])
    else:
        state = State(
            np.array([[3.0], [1.0], [0.1], [2.0], [2.0], [0.2], [4.0], [0.5], [0.05]])
        )
        length_scales = np.array([0.01, 0.02, 0.005])
        variances = np.array([0.1, 0.1, 0.1])
    return state, length_scales, variances


#### check one by one, see why pdf says singular


@pytest.mark.parametrize("sign", [1, -1])
def test_gp(gp_model_params, sign):
    state, length_scales, variances = gp_model_params
    timediff = 1 * sign
    state_vec = state.state_vector

    # Create a 1D GP or an n-dimensional
    # CombinedGaussianTransitionModel object
    window_size = 3
    dim = len(length_scales)
    jitter = 1e-6  # put here?
    if dim == 1:
        kernel_params = {
            "length_scale": length_scales[0],
            "kernel_variance": variances[0],
        }
        model_obj = MarkovianGP(window_size=window_size, kernel_params=kernel_params)
    else:
        model_list = []
        for i in range(dim):
            kernel_params = {
                "length_scale": length_scales[i],
                "kernel_variance": variances[i],
            }
            model_list.append(
                MarkovianGP(window_size=window_size, kernel_params=kernel_params)
            )
        model_obj = CombinedGaussianTransitionModel(model_list)

    # State related variables
    old_timestamp = datetime.datetime.now()
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # create track required for constructing transition and process noise covar matrix
    start_time = datetime.datetime.now()
    states = [
        State(
            [[i + 1.0]], timestamp=start_time + datetime.timedelta(seconds=i * timediff)
        )
        for i in range(window_size)
    ]
    track = Track(states)

    # Model-related components
    mat_list = []
    covar_list = []

    for i in range(0, dim):
        l = length_scales[i]
        var = variances[i]

        # Training times (window of size 3)
        train_times = [2 * timediff, timediff, 0]

        # Prediction time (one step ahead)
        t_star = 3 * timediff

        # Kernel matrix for training window
        K = np.array(
            [
                [
                    se_kernel(train_times[i], train_times[j], l, var)
                    for j in range(window_size)
                ]
                for i in range(window_size)
            ]
        )
        K += jitter * np.eye(window_size)

        # Cross-covariance vector for prediction
        k_star = np.array([se_kernel(ti, t_star, l, var) for ti in train_times])

        # Compute GP weights and predictive variance manually
        weights = np.linalg.solve(K, k_star)
        # print(weights)
        v_star = se_kernel(t_star, t_star, l, var) - k_star.T @ weights

        # Transition matrix F for this dimension (Markovian GP)
        F_dim = np.array([[weights[0], weights[1], weights[2]], [1, 0, 0], [0, 1, 0]])
        mat_list.append(F_dim)

        # Covariance matrix Q for this dimension
        Q_dim = np.zeros((window_size, window_size))
        Q_dim[0, 0] = v_star

        # temporary

        covar_list.append(Q_dim)

    F = sp.linalg.block_diag(*mat_list)
    Q = sp.linalg.block_diag(*covar_list)

    # Ensure ```model_obj.transfer_function(time_interval)``` returns F
    assert np.allclose(
        F,
        model_obj.jacobian(State(state_vec), time_interval=time_interval, track=track),
        rtol=1e-6,
    )

    # Ensure ```model_obj.covar(time_interval)``` returns Q
    assert np.allclose(
        Q,
        model_obj.covar(
            timestamp=new_timestamp, time_interval=time_interval, track=track
        ),
        rtol=1e-6,
    )

    # Propagate a state vector through the model
    # (without noise)
    new_state_vec_wo_noise = model_obj.function(
        state, timestamp=new_timestamp, time_interval=time_interval, track=track
    )
    assert np.allclose(new_state_vec_wo_noise, F @ state_vec, rtol=1e-6)

    # Evaluate the likelihood of the predicted state, given the prior
    # (without noise)
    prob = model_obj.pdf(
        State(new_state_vec_wo_noise),
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        track=track,
        allow_singular=True,
    )
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_wo_noise.T,
        mean=np.array(F @ state_vec).ravel(),
        cov=Q,
        allow_singular=True,
    )

    # Propagate a state vector throughout the model
    # (with internal noise)
    new_state_vec_w_inoise = model_obj.function(
        state,
        noise=True,
        timestamp=new_timestamp,
        time_interval=time_interval,
        track=track,
    )
    assert not np.allclose(new_state_vec_w_inoise, F @ state_vec, rtol=1e-6)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(
        State(new_state_vec_w_inoise),
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        track=track,
    )
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_inoise.T,
        mean=np.array(F @ state_vec).ravel(),
        cov=Q,
        allow_singular=True,
    )

    # Propagate a state vector throught the model
    # (with external noise)
    noise = model_obj.rvs(
        timestamp=new_timestamp, time_interval=time_interval, track=track
    )
    new_state_vec_w_enoise = model_obj.function(
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        noise=noise,
        track=track,
    )
    assert np.allclose(new_state_vec_w_enoise, F @ state_vec + noise, rtol=1e-6)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = model_obj.pdf(
        State(new_state_vec_w_enoise),
        state,
        timestamp=new_timestamp,
        time_interval=time_interval,
        track=track,
    )
    assert approx(prob) == multivariate_normal.pdf(
        new_state_vec_w_enoise.T,
        mean=np.array(F @ state_vec).ravel(),
        cov=Q,
        allow_singular=True,
    )
