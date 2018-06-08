# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import numpy as np

from stonesoup.types.detection import Detection
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from stonesoup.models.measurementmodel.linear import LinearGaussian1D
from stonesoup.types.state import GaussianState


def test_kalman():

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_covar=0.04)

    # Define predicted state
    prediction = GaussianState(np.array([[-6.45], [0.7]]),
                               np.array([[4.1123, 0.0013],
                                         [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    meas_pred = GaussianState(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar())
    cross_covar = prediction.covar@lg.matrix().T
    kalman_gain = cross_covar@np.linalg.inv(meas_pred.covar)
    eval_posterior = GaussianState(
        prediction.mean +
        kalman_gain@(measurement.state_vector-meas_pred.mean),
        prediction.covar - kalman_gain@meas_pred.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = KalmanUpdater(measurement_model=lg)

    # Perform and assert state update (without providing cross-covariance)
    posterior = updater.update(prediction=prediction,
                               measurement=measurement)
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.array_equal(posterior.covar, eval_posterior.covar))
    assert(posterior.timestamp == prediction.timestamp)


def test_extendedkalman():

    # Initialise a measurement model
    lg = LinearGaussian1D(ndim_state=2, mapping=0, noise_covar=0.04)

    # Define predicted state
    prediction = GaussianState(np.array([[-6.45], [0.7]]),
                               np.array([[4.1123, 0.0013],
                                         [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    meas_pred = GaussianState(
        lg.matrix()@prediction.mean,
        lg.matrix()@prediction.covar@lg.matrix().T+lg.covar())
    cross_covar = prediction.covar@lg.matrix().T
    kalman_gain = cross_covar@np.linalg.inv(meas_pred.covar)
    eval_posterior = GaussianState(
        prediction.mean +
        kalman_gain@(measurement.state_vector-meas_pred.mean),
        prediction.covar - kalman_gain@meas_pred.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = ExtendedKalmanUpdater(measurement_model=lg)

    # Perform and assert state update (without providing cross-covariance)
    posterior = updater.update(prediction=prediction,
                               measurement=measurement)
    assert(np.array_equal(posterior.mean, eval_posterior.mean))
    assert(np.array_equal(posterior.covar, eval_posterior.covar))
    assert(posterior.timestamp == prediction.timestamp)
