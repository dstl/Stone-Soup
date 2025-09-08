"""Test for updater.ensemble module"""
import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import EnsembleStatePrediction
from stonesoup.types.state import EnsembleState
from stonesoup.updater.ensemble import EnsembleUpdater, EnsembleSqrtUpdater, \
    LinearisedEnsembleUpdater


@pytest.fixture(params=[True, False])
def model_on_detection(request):
    return request.param


@pytest.mark.parametrize(
        'ensemble_updater', [EnsembleUpdater, EnsembleSqrtUpdater, LinearisedEnsembleUpdater])
def test_ensemble(ensemble_updater, model_on_detection):

    # Initialize variables
    measurement_model = LinearGaussian(ndim_state=2, mapping=[0],
                                       noise_covar=np.array([[0.04]]))
    timestamp = datetime.datetime(2021, 3, 5, 22, 3, 17)
    num_vectors = 200

    test_ensemble = EnsembleState.generate_ensemble(
                        np.array([[-6.45], [0.7]]),
                        np.array([[4.1123, 0.0013],
                                  [0.0013, 0.0365]]), num_vectors)

    # Create Prediction, Measurement, and Updater
    prediction = EnsembleStatePrediction(test_ensemble,
                                         timestamp=timestamp)
    if model_on_detection:
        measurement = Detection(
            np.array([[-6.23]]), timestamp, measurement_model=measurement_model)
        updater = ensemble_updater(None)
    else:
        measurement = Detection(np.array([[-6.23]]), timestamp)
        updater = ensemble_updater(measurement_model)

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
    test_measurement_prediction = updater.predict_measurement(
        prediction, measurement_model=measurement.measurement_model)
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

    measurement_model.noise_covar += 1.  # Increase noise to confirm test
    updater = EnsembleUpdater(measurement_model)
    noise_measurement_prediction = updater.predict_measurement(
        prediction, measurement_model=measurement.measurement_model)
    no_noise_measurement_prediction = updater.predict_measurement(
        prediction, measurement_model=measurement.measurement_model, measurement_noise=False)
    assert np.sum(np.trace(no_noise_measurement_prediction.covar)) \
        < np.sum(np.trace(noise_measurement_prediction.covar))
