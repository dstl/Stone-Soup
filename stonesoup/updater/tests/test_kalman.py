# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction)
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import (
    KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater)


@pytest.mark.parametrize(
    "UpdaterClass, measurement_model, prediction, measurement",
    [
        (   # Standard Kalman
            KalmanUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]]))
        ),
        (   # Extended Kalman
            ExtendedKalmanUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]]))
        ),
        (   # Unscented Kalman
            UnscentedKalmanUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]]))
        ),
    ],
    ids=["standard", "extended", "unscented"]
)
def test_kalman(UpdaterClass, measurement_model, prediction, measurement):

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix()@prediction.mean,
        measurement_model.matrix()@prediction.covar
        @measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar@measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar@np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = UpdaterClass(measurement_model=measurement_model)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert(np.allclose(measurement_prediction.mean,
                       eval_measurement_prediction.mean,
                       0, atol=1.e-14))
    assert(np.allclose(measurement_prediction.covar,
                       eval_measurement_prediction.covar,
                       0, atol=1.e-14))
    assert(np.allclose(measurement_prediction.cross_covar,
                       eval_measurement_prediction.cross_covar,
                       0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert(np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert(np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert(np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)


def test_sqrt_kalman():
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[4.1123, 0.0013],
                                                   [0.0013, 0.0365]]))
    measurement = Detection(np.array([[-6.23]]))

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    updater = KalmanUpdater(measurement_model=measurement_model)

    # First test that the square root form does nothing wrong
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))

    updater.sqrt_form = True
    posterior_s = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))

    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.allclose(posterior_s.covar, eval_posterior.covar, 0,
                        atol=1.e-14))

    # Next create a prediction with a covariance that will cause problems
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[1e24, 1e-24],
                                                   [1e-24, 1e24]]))

    updater.sqrt_form = False
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))

    updater.sqrt_form = True
    posterior_s = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))

    # The new posterior will  be
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        np.array([[0.04, 0],
                  [0, 1e24]]))  # Accessed by looking through the Decimal() quantities...
    # It's actually [0.039999999999 1e-48], [1e-24 1e24 + 1e-48]] ish

    assert (not np.allclose(posterior.covar, posterior_s.covar, 0, atol=1.e-14))
    assert (not np.allclose(posterior.covar, eval_posterior.covar, rtol=1.e-2))
    assert (np.allclose(posterior_s.covar, eval_posterior.covar, rtol=1.e-2))
