"""Test for updater.chernoff module"""
import pytest
import numpy as np
from scipy.stats import multivariate_normal

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
        ),
        (   # Chernoff Updater, asymmetric omega
            ChernoffUpdater,
            LinearGaussian(ndim_state=2, mapping=[0, 1],
                           noise_covar=np.array([[0.04, 0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            GaussianDetection(state_vector=np.array([[-6.23, 0.83]]),
                              covar=np.diag([0.75, 1.2])),
            0.3
        ),
    ],
    ids=["standard", "asymmetric_omega"]
)
def test_chernoff(UpdaterClass, measurement_model, prediction, measurement, omega):

    # Calculate evaluation variables
    innov_cov = measurement_model.noise_covar/omega + prediction.covar/(1-omega)
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

    # Get and assert measurement prediction without measurement noise
    measurement_prediction = updater.predict_measurement(prediction, measurement_noise=False)
    assert np.allclose(measurement_prediction.mean,
                       eval_measurement_prediction.mean,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.covar,
                       prediction.covar,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.cross_covar,
                       eval_measurement_prediction.cross_covar,
                       0, atol=1.e-14)

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


@pytest.mark.parametrize("property_value, argument, symmetric", [
    (False, None, False),
    (True, None, True),
    (False, True, True),
    (True, False, False),
])
def test_chernoff_force_symmetric_covariance(property_value, argument, symmetric):
    """Symmetric covariance forced by the property, unless the update argument says otherwise"""
    # Asymmetric prediction covariance, as from accumulated rounding
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0019, 0.0365]]))
    covar = np.diag([0.75, 1.2])
    measurement = GaussianDetection(
        state_vector=np.array([[-6.23], [0.83]]), covar=covar,
        measurement_model=LinearGaussian(ndim_state=2, mapping=[0, 1], noise_covar=covar))
    updater = ChernoffUpdater(None, force_symmetric_covariance=property_value)

    posterior = updater.update(
            SingleHypothesis(prediction, measurement),
            force_symmetric_covariance=argument)

    assert np.array_equal(posterior.covar, posterior.covar.T) == symmetric


@pytest.mark.parametrize("omega", [0.2, 0.5, 0.7])
def test_chernoff_innovation_covariance_normalises(omega):
    """Innovation covariance is the normalising constant of the Chernoff fusion rule

    For the Gaussian case, the identity
    N(x; a, A/omega) N(x; b, B/(1-omega)) = N(x; d, D) N(a; b, V)
    holds for all x, where (d, D) is the updated state.
    """
    prediction = GaussianStatePrediction(
        np.array([[-6.45], [0.7]]), np.array([[4.1123, 0.0013], [0.0013, 0.0365]]))
    covar = np.array([[0.75, 0.1], [0.1, 1.2]])
    measurement = GaussianDetection(
        state_vector=np.array([[-6.23], [0.83]]), covar=covar,
        measurement_model=LinearGaussian(ndim_state=2, mapping=[0, 1], noise_covar=covar))
    updater = ChernoffUpdater(None, omega=omega)

    posterior = updater.update(SingleHypothesis(prediction, measurement))
    innov_covar = posterior.hypothesis.measurement_prediction.covar

    a, b = measurement.state_vector.ravel(), prediction.state_vector.ravel()
    for x in ([-6.3, 0.75], [-5., 1.], [-8., 0.2]):
        lhs = (multivariate_normal.pdf(x, a, covar/omega)
               * multivariate_normal.pdf(x, b, prediction.covar/(1-omega)))
        rhs = (multivariate_normal.pdf(x, posterior.state_vector.ravel(), posterior.covar)
               * multivariate_normal.pdf(a, b, innov_covar))
        assert np.isclose(lhs, rhs)
