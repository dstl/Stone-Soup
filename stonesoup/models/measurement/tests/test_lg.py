# coding: utf-8
import pytest
import numpy as np
from scipy.stats import multivariate_normal

from ..linear import LinearGaussian


@pytest.mark.parametrize(
    "H, R, ndim_state, mapping",
    [
        (       # 1D meas, 2D state
                np.array([[1, 0]]),
                np.array([[0.1]]),
                2,
                [0],
        ),
        (       # 2D meas, 4D state
                np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),
                np.diag([0.1, 0.1]),
                4,
                [0, 2],
        ),
        (       # 4D meas, 2D state
                np.array([[1, 0], [0, 0], [0, 1], [0, 0]]),
                np.diag([0.1, 0.1, 0.1, 0.1]),
                2,
                [0, None, 1, None],
        ),
    ],
    ids=["1D_meas:2D_state", "2D_meas:4D_state", "4D_meas:2D_state"]
)
def test_lgmodel(H, R, ndim_state, mapping):
    """ LinearGaussian 1D Measurement Model test """

    # State related variables
    state_vec = np.array([[n] for n in range(ndim_state)])

    # Create and a Constant Velocity model object
    lg = LinearGaussian(ndim_state=ndim_state,
                        noise_covar=R,
                        mapping=mapping)

    # Ensure ```lg.transfer_function()``` returns H
    assert np.array_equal(H, lg.matrix())

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, lg.covar())

    # Project a state throught the model
    # (without noise)
    meas_pred_wo_noise = lg.function(state_vec, noise=0)
    assert np.array_equal(meas_pred_wo_noise, H@state_vec)

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = lg.pdf(meas_pred_wo_noise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_wo_noise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R).T)

    # Propagate a state vector throught the model
    # (with internal noise)
    meas_pred_w_inoise = lg.function(state_vec)
    assert not np.array_equal(meas_pred_w_inoise, H@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(meas_pred_w_inoise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_w_inoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R).T)

    # Propagate a state vector throught the model
    # (with external noise)
    noise = lg.rvs()
    meas_pred_w_enoise = lg.function(state_vec,
                                     noise=noise)
    assert np.array_equal(meas_pred_w_enoise, H@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(meas_pred_w_enoise, state_vec)
    assert np.array_equal(prob, multivariate_normal.pdf(
        meas_pred_w_enoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R).T)
