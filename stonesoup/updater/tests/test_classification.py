# coding: utf-8
import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.models.measurement.observation import BasicTimeInvariantObservationModel
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.update import StateUpdate
from stonesoup.updater.classification import ClassificationUpdater
from ...types.state import State


def create_random_multinomial(length):
    total = 0
    sv = list()
    for i in range(length - 1):
        x = np.random.uniform(0, 1 - total)
        sv.append(x)
        total += x
    sv.append(1 - total)
    sv = StateVector(sv)
    return State(sv)


def get_measurement_models(num_models, ndim_state):
    models = set()
    for _ in range(num_models):
        rows = list()
        for _ in range(np.random.randint(1, 10)):
            rows.append(create_random_multinomial(ndim_state).state_vector.flatten())
        ET = np.array(rows)
        models.add(BasicTimeInvariantObservationModel(ET.T))
    return models


@pytest.mark.parametrize('ndim_state', (1, 2, 3, 4))
def test_classification_updater(ndim_state):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)

    measurement_models = get_measurement_models(10, ndim_state)

    # test measurement model error
    with pytest.raises(ValueError, match="Measurement model must be observation-based with an "
                                         "Emission matrix property for ClassificationUpdater"):
        ClassificationUpdater(CartesianToElevationBearingRange(ndim_state,
                                                               np.arange(ndim_state),
                                                               np.eye(1)))

    for measurement_model in measurement_models:
        prediction = create_random_multinomial(ndim_state)
        prediction.timestamp = future

        updater = ClassificationUpdater(measurement_model)

        # test measurement prediction
        eval_measurement_prediction = measurement_model.function(prediction)
        measurement_prediction = updater.predict_measurement(prediction)

        assert (np.allclose(measurement_prediction.state_vector,
                            eval_measurement_prediction,
                            0,
                            atol=1.e-14))

        E = measurement_model.emission_matrix
        measurement = measurement_model.function(create_random_multinomial(E.shape[0]))
        hypothesis = SingleHypothesis(prediction=prediction,
                                      measurement=Detection(measurement,
                                                            timestamp=future,
                                                            measurement_model=measurement_model))

        # test get emission matrix
        assert (updater.get_emission_matrix(hypothesis) == E).all()

        # test update (without measurement prediction)
        eval_posterior = np.multiply(prediction.state_vector, E@measurement)
        eval_posterior = eval_posterior / np.sum(eval_posterior)
        posterior = updater.update(hypothesis)

        assert isinstance(posterior, StateUpdate)
        assert (np.allclose(posterior.state_vector, eval_posterior, 0, atol=1.e-14))
        assert posterior.timestamp == prediction.timestamp

        # Assert presence of measurement
        assert hasattr(posterior, 'hypothesis')
        assert posterior.hypothesis == hypothesis
