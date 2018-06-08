# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import numpy as np

from stonesoup.types.detection import Detection
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from stonesoup.models.measurementmodel.linear import LinearGaussian
from stonesoup.types.state import GaussianState


def test_kalman():

    # Initialise a measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))

    # Define predicted state
    mean_pred = np.array([[-6.45], [0.7]])
    covar_pred = np.array([[4.1123, 0.0013],
                           [0.0013, 0.0365]])
    state_pred = GaussianState(mean_pred, covar_pred)
    meas_pred = GaussianState(
        lg.matrix()@state_pred.mean,
        lg.matrix()@state_pred.covar@lg.matrix().T+lg.covar())
    cross_covar = state_pred.covar@lg.matrix().T
    meas = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_kalman_gain = cross_covar@np.linalg.inv(meas_pred.covar)
    eval_state_post = GaussianState(
        state_pred.mean + eval_kalman_gain@(meas.state_vector-meas_pred.mean),
        state_pred.covar - eval_kalman_gain@meas_pred.covar@eval_kalman_gain.T)

    # Initialise a kalman updater
    updater = KalmanUpdater(measurement_model=lg)

    # Perform and assert state update
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas,
                                             cross_covar=cross_covar)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))
    assert(state_post.timestamp == state_pred.timestamp)

    # Re-initialise a kalman updater
    updater = KalmanUpdater(measurement_model=lg)

    # Perform and assert state update (without providing cross-covariance)
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))
    assert(state_post.timestamp == state_pred.timestamp)


def test_extendedkalman():

    # Initialise a measurement model
    lg = LinearGaussian(ndim_state=2, mapping=[0],
                        noise_covar=np.array([[0.04]]))

    # Define predicted state
    mean_pred = np.array([[-6.45], [0.7]])
    covar_pred = np.array([[4.1123, 0.0013],
                           [0.0013, 0.0365]])
    state_pred = GaussianState(mean_pred, covar_pred)
    meas_pred = GaussianState(
        lg.matrix()@state_pred.mean,
        lg.matrix()@state_pred.covar@lg.matrix().T+lg.covar())
    cross_covar = state_pred.covar@lg.matrix().T
    meas = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_kalman_gain = cross_covar@np.linalg.inv(meas_pred.covar)
    eval_state_post = GaussianState(
        state_pred.mean + eval_kalman_gain@(meas.state_vector-meas_pred.mean),
        state_pred.covar - eval_kalman_gain@meas_pred.covar@eval_kalman_gain.T)

    # Initialise a kalman updater
    updater = ExtendedKalmanUpdater(measurement_model=lg)

    # Perform and assert state update
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas,
                                             cross_covar=cross_covar)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))
    assert(state_post.timestamp == state_pred.timestamp)

    # Re-initialise a kalman updater
    updater = KalmanUpdater(measurement_model=lg)

    # Perform and assert state update (without providing cross-covariance)
    state_post, kalman_gain = updater.update(state_pred=state_pred,
                                             meas_pred=meas_pred,
                                             meas=meas)
    assert(np.array_equal(state_post.mean, eval_state_post.mean))
    assert(np.array_equal(state_post.covar, eval_state_post.covar))
    assert(np.array_equal(kalman_gain, eval_kalman_gain))
    assert(state_post.timestamp == state_pred.timestamp)
