# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import pytest
import numpy as np

from ...models.measurement.linear import LinearGaussian
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction)
from ...types.state import GaussianState
from ...updater.kalman import (
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
        )
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
    measurement_prediction = updater.get_measurement_prediction(prediction)
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
