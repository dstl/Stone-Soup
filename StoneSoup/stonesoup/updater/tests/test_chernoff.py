"""Test for updater.chernoff module"""
import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import GaussianDetection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction)
from stonesoup.types.state import GaussianState
from stonesoup.updater.chernoff import ChernoffUpdater


@pytest.mark.parametrize(
    "UpdaterClass, measurement_model, prediction, measurement, omega",
    [
        (   # Chernoff Updater
            ChernoffUpdater,
            LinearGaussian(ndim_state=2, mapping=[0, 1],
                           noise_covar=np.array([[0.04, 0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            GaussianDetection(state_vector=np.array([[-6.23, 0.83]]),
                              covar=np.diag([0.75, 1.2])),
            0.5
        )
    ],
    ids=["standard"]
)
def test_chernoff(UpdaterClass, measurement_model, prediction, measurement, omega):

    # Calculate evaluation variables
    innov_cov = 1/(1-omega)*measurement_model.noise_covar + 1/omega*prediction.covar
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        innov_cov,
        cross_covar=prediction.covar @ measurement_model.matrix().T)

    posterior_cov = np.linalg.inv(omega*np.linalg.inv(measurement.covar) +
                                  (1-omega)*np.linalg.inv(prediction.covar))
    posterior_mean = posterior_cov@(omega*np.linalg.inv(measurement.covar) @
                                    measurement.state_vector + (1-omega) *
                                    np.linalg.inv(prediction.covar)@prediction.state_vector)
    eval_posterior = GaussianState(
        posterior_mean,
        posterior_cov)

    # Initialise a Chernoff updater
    updater = UpdaterClass(measurement_model=measurement_model, omega=omega)

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
    assert np.allclose(posterior.hypothesis.measurement_prediction.covar,
                       measurement_prediction.covar, 0, atol=1.e-14)
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp
