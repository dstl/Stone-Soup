# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import numpy as np

from stonesoup.types import Track
from stonesoup.types.detection import Detection
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater,\
    SqrtKalmanUpdater
from stonesoup.measurementmodel.base import LinearGaussian1D
from stonesoup.types.state import GaussianState


def test_kalman():

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_var=0.04)

    # Define predicted state
    mean_pred = np.array([[-6.45], [0.7]])
    covar_pred = np.array([[4.1123, 0.0013],
                           [0.0013, 0.0365]])
    state_pred = GaussianState(mean_pred, covar_pred)
    meas_pred = GaussianState(
        lg.eval()@state_pred.mean,
        lg.eval()@state_pred.covar@lg.eval().T+lg.covar())
    cross_covar = state_pred.covar@lg.eval().T
    meas = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_kalman_gain = cross_covar@np.linalg.inv(meas_pred.covar)
    eval_state_post = GaussianState(
        state_pred.mean + eval_kalman_gain@(meas.state_vector-meas_pred.mean),
        state_pred.covar - eval_kalman_gain@meas_pred.covar@eval_kalman_gain.T)

    # Initialise a kalman updater
    updater = KalmanUpdater(meas_model=lg)

    # Perform and assert state update
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas,
                                             cross_covar=cross_covar)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))

    # Re-initialise a kalman updater
    updater = KalmanUpdater(meas_model=lg)

    # Perform and assert state update (without providing cross-covariance)
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))


def test_extendedkalman():

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_var=0.04)

    # Define predicted state
    mean_pred = np.array([[-6.45], [0.7]])
    covar_pred = np.array([[4.1123, 0.0013],
                           [0.0013, 0.0365]])
    state_pred = GaussianState(mean_pred, covar_pred)
    meas_pred = GaussianState(
        lg.eval()@state_pred.mean,
        lg.eval()@state_pred.covar@lg.eval().T+lg.covar())
    cross_covar = state_pred.covar@lg.eval().T
    meas = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_kalman_gain = cross_covar@np.linalg.inv(meas_pred.covar)
    eval_state_post = GaussianState(
        state_pred.mean + eval_kalman_gain@(meas.state_vector-meas_pred.mean),
        state_pred.covar - eval_kalman_gain@meas_pred.covar@eval_kalman_gain.T)

    # Initialise an extended kalman updater
    updater = ExtendedKalmanUpdater(meas_model=lg)

    # Perform and assert state update
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas,
                                             cross_covar=cross_covar)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))

    # Re-initialise an extended kalman updater
    updater = ExtendedKalmanUpdater(meas_model=lg)

    # Perform and assert state update (without providing cross-covariance)
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))


# def test_sqrtkalman():
#     """Square Root Kalman Updater test"""

#     # TODO: Better test data
#     estimate = GaussianState(np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4,
#                              np.array([[2.2128, 0, 0, 0],
#                                        [0.0002, 2.2130, 0, 0],
#                                        [0.3897, -0.00004, 0.0128, 0],
#                                        [0, 0.3897, 0.0013, 0.0135]]) * 1e3)
#     detection = GaussianState(np.array([[2.4378], [1.0072]]) * 1e4,
#                               np.array([[19.8607, 0], [-35.8829, 23.1799]]))
#     track = Track()
#     track.states.append(estimate)
#     state, innov = SqrtKalmanUpdater.update(track, detection)

#     assert np.allclose(state.state_vector, np.array([
#         [2.4375], [1.0078], [0.7553], [0.0014]]) * 1e4, 1)
#     assert np.allclose(state.covar, np.array([
#         [19.8573, 0, 0, 0],
#         [-35.8728, 23.1786, 0, 0],
#         [3.4975, -0.0004, 12.8034, 0],
#         [-6.3167, 4.0812, 1.2637, 13.5487]]), 0.1)

#     assert np.allclose(innov.state_vector, np.array(
#         [[4.2891], [0.0078]]) * 1e4, 1)
#     assert np.allclose(innov.covar, np.array([
#         [2.2129, 0],
#         [-0.0001, 2.2134]]) * 1e3, 1)
