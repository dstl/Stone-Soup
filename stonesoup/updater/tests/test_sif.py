import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction)
from stonesoup.types.state import GaussianState
from stonesoup.updater.slidinginnovation import (
    SlidingInnovationUpdater, ExtendedSlidingInnovationUpdater)


@pytest.mark.parametrize(
    "UpdaterClass, measurement_model, prediction, measurement, layer_width",
    [
        (   # Standard Kalman
            SlidingInnovationUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]])),
            10*np.array([0.04])  # 10 x diag(R)
        ),
        (   # Extended Kalman
            ExtendedSlidingInnovationUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]])),
            10 * np.array([0.04])  # 10 x diag(R)
        ),
    ],
    ids=["standard", "extended"]
)
def test_sif(UpdaterClass, measurement_model, prediction, measurement, layer_width):

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    innovation_vector = measurement.state_vector - eval_measurement_prediction.state_vector
    kalman_gain = np.linalg.pinv(measurement_model.matrix()) \
        @ np.diag(np.clip(np.abs(innovation_vector) / layer_width, -1, 1).ravel())
    I_KH = np.identity(prediction.ndim) - kalman_gain @ measurement_model.matrix()
    posterior_covariance = \
        I_KH @ prediction.covar @ I_KH.T + kalman_gain @ measurement_model.covar() @ kalman_gain.T
    eval_posterior = GaussianState(
        prediction.mean + kalman_gain@(measurement.state_vector-eval_measurement_prediction.mean),
        posterior_covariance)

    # Initialise a updater
    updater = UpdaterClass(measurement_model=measurement_model, layer_width=layer_width)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert np.allclose(measurement_prediction.mean,
                       eval_measurement_prediction.mean,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.covar,
                       eval_measurement_prediction.covar,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.cross_covar,
                       eval_measurement_prediction.cross_covar,
                       0, atol=1.e-14)

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14)
    assert np.array_equal(posterior.hypothesis.prediction, prediction)
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14)
    assert np.allclose(posterior.hypothesis.measurement_prediction.covar,
                       measurement_prediction.covar, 0, atol=1.e-14)
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14)
    assert np.array_equal(posterior.hypothesis.prediction, prediction)
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14)
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp
