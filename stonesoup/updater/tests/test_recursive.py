"""Test for updater.recursive module"""
import datetime
import math

import numpy as np
import pytest

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianMeasurementPrediction, EnsembleStatePrediction
from stonesoup.types.state import GaussianState, EnsembleState
from stonesoup.updater.recursive import BayesianRecursiveUpdater, RecursiveEnsembleUpdater, \
    RecursiveLinearisedEnsembleUpdater, VariableStepBayesianRecursiveUpdater, \
    ErrorControllerBayesianRecursiveUpdater


def test_bruf_single_step(measurement_model, prediction, measurement, timestamp):

    n_steps = 1

    updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=n_steps,
                                       use_joseph_cov=False)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)


def test_bruf_multi_step(measurement_model, prediction, measurement, timestamp):

    n_steps = 10

    updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=n_steps,
                                       use_joseph_cov=False)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))


def test_force_symmetric_bruf(measurement_model, prediction, measurement, timestamp):

    n_steps = 1

    updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=n_steps,
                                       use_joseph_cov=False, force_symmetric_covariance=True)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)


def test_bruf_errors(measurement_model, prediction, measurement):

    # Initialise a kalman updater
    updater = BayesianRecursiveUpdater(measurement_model=measurement_model,
                                       number_steps=0,
                                       force_symmetric_covariance=True)

    # Run updater
    with pytest.raises(ValueError):
        _ = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))


def test_jcru_single_step(measurement_model, prediction, measurement, timestamp):

    n_steps = 1

    updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=n_steps,
                                       use_joseph_cov=True)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)


def test_jcru_multi_step(measurement_model, prediction, measurement, timestamp):

    n_steps = 10

    updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=n_steps,
                                       use_joseph_cov=True)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))


def test_force_symmetric_jcru(measurement_model, prediction, measurement, timestamp):

    n_steps = 1

    updater = BayesianRecursiveUpdater(measurement_model=measurement_model, number_steps=n_steps,
                                       use_joseph_cov=True, force_symmetric_covariance=True)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)


def test_jcru_errors(measurement_model, prediction, measurement):

    # Initialise a kalman updater
    updater = BayesianRecursiveUpdater(measurement_model=measurement_model,
                                       number_steps=0, use_joseph_cov=True)

    # Run updater
    with pytest.raises(ValueError):
        _ = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))


def test_recursive_linearised_ensemble_updater():
    # Initialize variables
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime(2021, 3, 5, 22, 3, 17)
    num_vectors = 100

    test_ensemble = EnsembleState.generate_ensemble(
                        np.array([[-6.45], [0.7]]),
                        np.array([[4.1123, 0.0013],
                                  [0.0013, 0.0365]]), num_vectors)

    # Create Prediction, Measurement, and Updater
    prediction = EnsembleStatePrediction(test_ensemble,
                                         timestamp=timestamp)

    measurement = Detection(np.array([[-6.23]]), timestamp, measurement_model=measurement_model)
    updater = RecursiveLinearisedEnsembleUpdater(measurement_model=measurement_model,
                                                 number_steps=5)

    # Construct hypothesis

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    assert updated_state.num_vectors == \
           updated_state.hypothesis.prediction.num_vectors
    assert np.allclose(updated_state.sqrt_covar @ updated_state.sqrt_covar.T,
                       updated_state.covar)

    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    assert updated_state.num_vectors == \
           updated_state.hypothesis.prediction.num_vectors
    assert np.allclose(updated_state.sqrt_covar @ updated_state.sqrt_covar.T,
                       updated_state.covar)


def test_recursive_linearised_ensemble_errors(measurement_model, prediction, measurement):

    # Initialise a kalman updater
    updater = RecursiveLinearisedEnsembleUpdater(measurement_model=measurement_model,
                                                 number_steps=0,
                                                 force_symmetric_covariance=True)

    # Run updater
    with pytest.raises(ValueError):
        _ = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))


def test_recursive_ensemble_multi_step():

    # Initialize variables
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime(2021, 3, 5, 22, 3, 17)
    num_vectors = 100

    test_ensemble = EnsembleState.generate_ensemble(
                        np.array([[-6.45], [0.7]]),
                        np.array([[4.1123, 0.0013],
                                  [0.0013, 0.0365]]), num_vectors)

    # Create Prediction, Measurement, and Updater
    prediction = EnsembleStatePrediction(test_ensemble,
                                         timestamp=timestamp)
    measurement = Detection(np.array([[-6.23]]), timestamp, measurement_model=measurement_model)
    updater = RecursiveEnsembleUpdater(measurement_model=measurement_model, number_steps=5)

    # Construct hypothesis

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    assert updated_state.num_vectors == \
           updated_state.hypothesis.prediction.num_vectors
    assert np.allclose(updated_state.sqrt_covar @ updated_state.sqrt_covar.T,
                       updated_state.covar)

    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    assert updated_state.num_vectors == \
           updated_state.hypothesis.prediction.num_vectors
    assert np.allclose(updated_state.sqrt_covar @ updated_state.sqrt_covar.T,
                       updated_state.covar)


def test_recursive_ensemble_single_step():

    # Initialize variables
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime(2021, 3, 5, 22, 3, 17)
    num_vectors = 100

    test_ensemble = EnsembleState.generate_ensemble(
                        np.array([[-6.45], [0.7]]),
                        np.array([[4.1123, 0.0013],
                                  [0.0013, 0.0365]]), num_vectors)

    # Create Prediction, Measurement, and Updater
    prediction = EnsembleStatePrediction(test_ensemble,
                                         timestamp=timestamp)
    measurement = Detection(np.array([[-6.23]]), timestamp, measurement_model=measurement_model)
    updater = RecursiveEnsembleUpdater(measurement_model=measurement_model, number_steps=1)

    # Construct hypothesis

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    assert updated_state.num_vectors == \
           updated_state.hypothesis.prediction.num_vectors
    assert np.allclose(updated_state.sqrt_covar @ updated_state.sqrt_covar.T,
                       updated_state.covar)

    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    assert updated_state.num_vectors == \
           updated_state.hypothesis.prediction.num_vectors
    assert np.allclose(updated_state.sqrt_covar @ updated_state.sqrt_covar.T,
                       updated_state.covar)


def test_recursive_ensemble_errors():
    # Initialize variables
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime(2021, 3, 5, 22, 3, 17)
    num_vectors = 100

    test_ensemble = EnsembleState.generate_ensemble(
        np.array([[-6.45], [0.7]]),
        np.array([[4.1123, 0.0013],
                  [0.0013, 0.0365]]), num_vectors)

    # Create Prediction, Measurement, and Updater
    prediction = EnsembleStatePrediction(test_ensemble,
                                         timestamp=timestamp)
    measurement = Detection(np.array([[-6.23]]), timestamp, measurement_model=measurement_model)
    updater = RecursiveEnsembleUpdater(measurement_model=measurement_model, number_steps=0)

    # Construct hypothesis

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater
    with pytest.raises(ValueError):
        _ = updater.update(hypothesis)


@pytest.mark.parametrize("n_steps, force_sym_cov, use_joseph_cov",
                         [(1, False, False), (1, True, False), (1, False, True)])
def test_vsbruf_single_step(n_steps, force_sym_cov, use_joseph_cov, measurement_model,
                            prediction, measurement, timestamp):

    updater = VariableStepBayesianRecursiveUpdater(measurement_model=measurement_model,
                                                   number_steps=n_steps,
                                                   use_joseph_cov=False,
                                                   force_symmetric_covariance=force_sym_cov)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)


@pytest.mark.parametrize("n_steps, force_sym_cov, use_joseph_cov",
                         [(10, False, False), (10, True, False), (10, False, True)])
def test_vsbruf_multi_step(n_steps, force_sym_cov, use_joseph_cov, measurement_model,
                           prediction, measurement, timestamp):

    updater = VariableStepBayesianRecursiveUpdater(measurement_model=measurement_model,
                                                   number_steps=n_steps,
                                                   use_joseph_cov=use_joseph_cov,
                                                   force_symmetric_covariance=force_sym_cov)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))


def test_vsbruf_errors(measurement_model, prediction, measurement):

    # Initialise a kalman updater
    updater = VariableStepBayesianRecursiveUpdater(measurement_model=measurement_model,
                                                   number_steps=0,
                                                   force_symmetric_covariance=True)

    # Run updater
    with pytest.raises(ValueError):
        _ = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))


@pytest.mark.parametrize("n_steps, force_sym_cov, use_joseph_cov",
                         [(1, False, False), (1, True, False), (1, False, True)])
def test_ec_bruf_single_step(n_steps, force_sym_cov, use_joseph_cov,
                             measurement_model, prediction, measurement, timestamp):

    updater = ErrorControllerBayesianRecursiveUpdater(measurement_model=measurement_model,
                                                      number_steps=n_steps,
                                                      use_joseph_cov=use_joseph_cov,
                                                      force_symmetric_covariance=force_sym_cov,
                                                      atol=10e-3,
                                                      rtol=10e-3,
                                                      f=math.sqrt(0.38),
                                                      fmin=0.2,
                                                      fmax=6)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain @ eval_measurement_prediction.covar @ kalman_gain.T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.covar,
                        eval_measurement_prediction.covar,
                        0, atol=1.e-14))
    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert (np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert (np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert (np.array_equal(posterior.hypothesis.measurement, measurement))
    assert (posterior.timestamp == prediction.timestamp)


@pytest.mark.parametrize("n_steps, force_sym_cov, use_joseph_cov",
                         [(10, False, False), (10, True, False), (10, False, True)])
def test_ecbruf_multi_step(n_steps, force_sym_cov, use_joseph_cov,
                           measurement_model, prediction, measurement, timestamp):

    updater = ErrorControllerBayesianRecursiveUpdater(measurement_model=measurement_model,
                                                      number_steps=n_steps,
                                                      use_joseph_cov=use_joseph_cov,
                                                      force_symmetric_covariance=force_sym_cov,
                                                      atol=10e-3,
                                                      rtol=10e-3,
                                                      f=math.sqrt(0.38),
                                                      fmin=0.2,
                                                      fmax=6)

    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement)

    # Run updater

    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim
    # Test updater runs with measurement prediction already in hypothesis.
    test_measurement_prediction = updater.predict_measurement(prediction)
    hypothesis = SingleHypothesis(prediction=prediction,
                                  measurement=measurement,
                                  measurement_prediction=test_measurement_prediction)
    updated_state = updater.update(hypothesis)

    assert updated_state.timestamp == timestamp
    assert updated_state.hypothesis.prediction == prediction
    assert updated_state.hypothesis.measurement == measurement
    assert updated_state.ndim == updated_state.hypothesis.prediction.ndim

    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        measurement_model.matrix() @ prediction.covar
        @ measurement_model.matrix().T
        + n_steps * measurement_model.covar(),
        cross_covar=prediction.covar @ measurement_model.matrix().T)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert (np.allclose(measurement_prediction.mean,
                        eval_measurement_prediction.mean,
                        0, atol=1.e-14))

    assert (np.allclose(measurement_prediction.cross_covar,
                        eval_measurement_prediction.cross_covar,
                        0, atol=1.e-14))


def test_ecbruf_errors(measurement_model, prediction, measurement):

    # Initialise a kalman updater
    updater = ErrorControllerBayesianRecursiveUpdater(measurement_model=measurement_model,
                                                      number_steps=0,
                                                      use_joseph_cov=False,
                                                      force_symmetric_covariance=True,
                                                      atol=10e-3,
                                                      rtol=10e-3,
                                                      f=math.sqrt(0.38),
                                                      fmin=0.2,
                                                      fmax=6)

    # Run updater
    with pytest.raises(ValueError):
        _ = updater.update(SingleHypothesis(prediction=prediction, measurement=measurement))
