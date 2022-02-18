# coding: utf-8
import pytest
from pytest import approx
import numpy as np
from scipy.stats import multivariate_normal

from ..linear import LinearGaussian
from ....types.state import State


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
    state = State(state_vec)

    # Create and a Constant Velocity model object
    lg = LinearGaussian(ndim_state=ndim_state,
                        noise_covar=R,
                        mapping=mapping)

    # Ensure ```lg.transfer_function()``` returns H
    assert np.array_equal(H, lg.matrix())

    # Ensure lg.jacobian() returns H
    assert np.array_equal(H, lg.jacobian(state=state))

    # Ensure ```lg.covar()``` returns R
    assert np.array_equal(R, lg.covar())

    # Project a state through the model
    # (without noise)
    meas_pred_wo_noise = lg.function(state)
    assert np.array_equal(meas_pred_wo_noise, H@state_vec)

    # Evaluate the likelihood of the predicted measurement, given the state
    # (without noise)
    prob = lg.pdf(State(meas_pred_wo_noise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_wo_noise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with internal noise)
    meas_pred_w_inoise = lg.function(state, noise=lg.rvs())
    assert not np.array_equal(meas_pred_w_inoise, H@state_vec)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(State(meas_pred_w_inoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_w_inoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R)

    # Propagate a state vector through the model
    # (with external noise)
    noise = lg.rvs()
    meas_pred_w_enoise = lg.function(state,
                                     noise=noise)
    assert np.array_equal(meas_pred_w_enoise, H@state_vec+noise)

    # Evaluate the likelihood of the predicted state, given the prior
    # (with noise)
    prob = lg.pdf(State(meas_pred_w_enoise), state)
    assert approx(prob) == multivariate_normal.pdf(
        meas_pred_w_enoise.T,
        mean=np.array(H@state_vec).ravel(),
        cov=R)

    # Test random seed give consistent results
    lg1 = LinearGaussian(ndim_state=ndim_state,
                         noise_covar=R,
                         mapping=mapping,
                         seed=1)
    lg2 = LinearGaussian(ndim_state=ndim_state,
                         noise_covar=R,
                         mapping=mapping,
                         seed=1)

    # Check first values produced by seed match
    for _ in range(3):
        assert all(lg1.rvs() == lg2.rvs())
