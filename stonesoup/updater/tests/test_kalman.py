"""Test for updater.kalman module"""
import datetime

import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction, ASDGaussianStatePrediction,
    ASDGaussianMeasurementPrediction)
from stonesoup.types.state import GaussianState, SqrtGaussianState
from stonesoup.updater.kalman import (KalmanUpdater,
                                      ExtendedKalmanUpdater,
                                      UnscentedKalmanUpdater,
                                      SqrtKalmanUpdater,
                                      IteratedKalmanUpdater,
                                      ASDKalmanUpdater)


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
        (   # Iterated Kalman
            IteratedKalmanUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]]))
        ),
    ],
    ids=["standard", "extended", "unscented", "iterated"]
)
def test_kalman(UpdaterClass, measurement_model, prediction, measurement):

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
        - kalman_gain@eval_measurement_prediction.covar @ kalman_gain.T)

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
    sqrt_prediction = SqrtGaussianState(prediction.state_vector,
                                        np.linalg.cholesky(prediction.covar))
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

    # Test Square root form returns the same as standard form
    updater = KalmanUpdater(measurement_model=measurement_model)
    sqrt_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=False)
    qr_updater = SqrtKalmanUpdater(measurement_model=measurement_model, qr_method=True)

    posterior = updater.update(SingleHypothesis(prediction=prediction,
                                                measurement=measurement))
    posterior_s = sqrt_updater.update(SingleHypothesis(prediction=sqrt_prediction,
                                                       measurement=measurement))
    posterior_q = qr_updater.update(SingleHypothesis(prediction=sqrt_prediction,
                                                     measurement=measurement))

    assert np.allclose(posterior_s.mean, eval_posterior.mean, 0, atol=1.e-14)
    assert np.allclose(posterior_q.mean, eval_posterior.mean, 0, atol=1.e-14)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14)
    assert np.allclose(eval_posterior.covar,
                       posterior_s.sqrt_covar@posterior_s.sqrt_covar.T, 0,
                       atol=1.e-14)
    assert np.allclose(posterior.covar,
                       posterior_s.sqrt_covar@posterior_s.sqrt_covar.T, 0,
                       atol=1.e-14)
    assert np.allclose(posterior.covar,
                       posterior_q.sqrt_covar@posterior_q.sqrt_covar.T, 0,
                       atol=1.e-14)
    # I'm not sure this is going to be true in all cases. Keep in order to find edge cases
    assert np.allclose(posterior_s.covar, posterior_q.covar, 0, atol=1.e-14)

    # Next create a prediction with a covariance that will cause problems
    prediction = GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                         np.array([[1e24, 1e-24],
                                                   [1e-24, 1e24]]))
    sqrt_prediction = SqrtGaussianState(prediction.state_vector,
                                        np.linalg.cholesky(prediction.covar))

    posterior = updater.update(SingleHypothesis(prediction=prediction,
                                                measurement=measurement))
    posterior_s = sqrt_updater.update(SingleHypothesis(
        prediction=sqrt_prediction, measurement=measurement))
    posterior_q = qr_updater.update(SingleHypothesis(prediction=sqrt_prediction,
                                                     measurement=measurement))

    # The new posterior will  be
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        np.array([[0.04, 0],
                  [0, 1e24]]))  # Accessed by looking through the Decimal() quantities...
    # It's actually [0.039999999999 1e-48], [1e-24 1e24 + 1e-48]] ish

    # Test that the square root form succeeds where the standard form fails
    assert not np.allclose(posterior.covar, eval_posterior.covar, rtol=5.e-3)
    assert np.allclose(posterior_s.sqrt_covar@posterior_s.sqrt_covar.T,
                       eval_posterior.covar, rtol=5.e-3)
    assert np.allclose(posterior_q.sqrt_covar@posterior_s.sqrt_covar.T,
                       eval_posterior.covar, rtol=5.e-3)


def test_asdkalman():
    timestamp = datetime.datetime.now()
    measurement_model = LinearGaussian(
        ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = ASDGaussianStatePrediction(
        np.array([[-6.45], [0.7]]),
        multi_covar=np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        timestamps=[timestamp],
        correlation_matrices={timestamp: {'P': np.eye(2)}},
        act_timestamp=timestamp)
    measurement = Detection(np.array([[-6.23]]), timestamp=timestamp)

    # Calculate evaluation variables
    Pi = np.block([[np.eye(prediction.ndim)],
                   [np.zeros((prediction.multi_covar.shape[0] -
                              prediction.ndim, prediction.ndim))]])
    cross_cov = prediction.multi_covar @ Pi @ measurement_model.matrix().T
    eval_measurement_prediction = ASDGaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        multi_covar=measurement_model.matrix()@prediction.covar
        @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=cross_cov, timestamps=prediction.timestamps)

    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = ASDKalmanUpdater(measurement_model=measurement_model)

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
