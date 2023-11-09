"""Test for updater.ensemble module"""
import numpy as np
import datetime

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import EnsembleStatePrediction
from stonesoup.types.state import EnsembleState
from stonesoup.updater.ensemble import EnsembleUpdater, EnsembleSqrtUpdater, \
    LinearisedEnsembleUpdater


def test_ensemble():

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
    measurement = Detection(np.array([[-6.23]]), timestamp)
    updater = EnsembleUpdater(measurement_model)

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


def test_sqrt_ensemble():

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
    measurement = Detection(np.array([[-6.23]]), timestamp)
    updater = EnsembleSqrtUpdater(measurement_model)

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


def test_linearised_ensemble_updater():
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

    measurement = Detection(np.array([[-6.23]]), timestamp)
    updater = LinearisedEnsembleUpdater(measurement_model)

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
