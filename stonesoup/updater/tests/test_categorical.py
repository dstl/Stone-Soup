import datetime

import numpy as np
import pytest

from ...models.measurement.categorical import MarkovianMeasurementModel
from ...models.measurement.linear import LinearGaussian
from ...types.array import StateVector
from ...types.detection import CategoricalDetection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import CategoricalMeasurementPrediction, CategoricalStatePrediction
from ...types.update import CategoricalStateUpdate
from ...updater.categorical import HMMUpdater


def test_hmm_updater():
    E = np.array([[30, 25, 5],
                  [20, 25, 10],
                  [10, 25, 80],
                  [40, 25, 5]])

    measurement_categories = ['red', 'green', 'blue', 'yellow']

    measurement_model = MarkovianMeasurementModel(E, measurement_categories=measurement_categories)

    updater = HMMUpdater(measurement_model)

    now = datetime.datetime.now()

    prediction = CategoricalStatePrediction([80, 10, 10], timestamp=now)

    # Test check measurement model
    assert updater._check_measurement_model(measurement_model) == measurement_model
    assert updater._check_measurement_model(None) == measurement_model
    updater.measurement_model = None
    with pytest.raises(ValueError, match="No measurement model specified"):
        updater._check_measurement_model(None)
    updater.measurement_model = measurement_model
    with pytest.raises(ValueError, match="HMMUpdater must be used in conjuction with "
                                         "HiddenMarkovianMeasurementModel types"):
        updater._check_measurement_model(LinearGaussian(np.eye(3),
                                                        mapping=(0, 1, 2),
                                                        noise_covar=np.eye(3)))

    measurement = CategoricalDetection(StateVector([10, 20, 30, 40]),
                                       timestamp=now,
                                       measurement_model=measurement_model,
                                       categories=measurement_categories)

    # Test measurement prediction
    measurement_prediction = updater.predict_measurement(prediction,
                                                         measurement_model)
    exp_measurement_prediction_vector = measurement_model.function(prediction, noise=False)
    assert isinstance(measurement_prediction, CategoricalMeasurementPrediction)
    assert np.allclose(measurement_prediction.state_vector, exp_measurement_prediction_vector)
    assert measurement_prediction.categories == measurement_categories

    # Test update (with/without measurement prediction)

    for hypothesis in [SingleHypothesis(prediction=prediction,
                                        measurement=measurement),
                       SingleHypothesis(prediction=prediction,
                                        measurement=measurement,
                                        measurement_prediction=measurement_prediction)]:
        update = updater.update(hypothesis)

        measurement_prediction = hypothesis.measurement_prediction
        assert isinstance(measurement_prediction, CategoricalMeasurementPrediction)
        assert np.allclose(measurement_prediction.state_vector, exp_measurement_prediction_vector)

        assert isinstance(update, CategoricalStateUpdate)
        assert update.hypothesis == hypothesis
        assert update.state_vector.shape[0] == 3
