# coding: utf-8
import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.linear import LinearGaussian
from ...models.measurement.nonlinear import CartesianToElevationBearingRange
from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.transition.tests.test_categorical import create_random_categorical
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.update import StateUpdate, CategoricalStateUpdate
from ...updater.categorical import HMMUpdater


def get_measurement_models(num_models, ndim_state):
    models = set()
    for _ in range(num_models):
        rows = list()
        ndim_meas = np.random.randint(1, 10)
        for _ in range(ndim_state):
            rows.append(create_random_categorical(ndim_meas).state_vector.flatten())
        E = np.array(rows)
        models.add(CategoricalMeasurementModel(ndim_state=ndim_state, emission_matrix=E))
    return models


@pytest.mark.parametrize('ndim_state', (1, 2, 3, 4))
def test_classification_updater(ndim_state):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)

    measurement_models = get_measurement_models(10, ndim_state)

    for measurement_model in measurement_models:
        prediction = create_random_categorical(ndim_state)
        prediction.timestamp = future

        updater = HMMUpdater(measurement_model)
        empty_updater = HMMUpdater()
        wrong_updater = HMMUpdater(LinearGaussian(ndim_state=ndim_state,
                                                  mapping=[0, 1],
                                                  noise_covar=np.eye(ndim_state)))

        # test measurement prediction
        eval_measurement_prediction = measurement_model.function(prediction)
        measurement_prediction = updater.predict_measurement(prediction)

        assert (np.allclose(measurement_prediction.state_vector,
                            eval_measurement_prediction,
                            0,
                            atol=1.e-14))

        E = measurement_model.emission_matrix
        measurement = measurement_model.function(create_random_categorical(E.shape[0]))
        hypothesis = SingleHypothesis(prediction=prediction,
                                      measurement=Detection(measurement,
                                                            timestamp=future,
                                                            measurement_model=measurement_model))

        # test get emission matrix
        assert (updater._get_emission_matrix(hypothesis) == E).all()

        # test update (without measurement prediction)
        eval_posterior = np.multiply(prediction.state_vector, E@measurement)
        eval_posterior = eval_posterior / np.sum(eval_posterior)
        posterior = updater.update(hypothesis)

        assert isinstance(posterior, CategoricalStateUpdate)
        assert posterior.num_categories == prediction.num_categories
        assert posterior.category_names == prediction.category_names
        assert (np.allclose(posterior.state_vector, eval_posterior, 0, atol=1.e-14))
        assert posterior.timestamp == prediction.timestamp

        # Assert presence of measurement
        assert hasattr(posterior, 'hypothesis')
        assert posterior.hypothesis == hypothesis

        empty_hypothesis = SingleHypothesis(prediction=prediction,
                                            measurement=Detection(measurement,
                                                                  timestamp=future))

        # test measurement model error
        with pytest.raises(ValueError,
                           match="No measurement model specified"):
            empty_updater.update(empty_hypothesis)
        # test measurement model error
        with pytest.raises(ValueError,
                           match="Measurement model must be categorical. I.E. it must have an "
                                 "Emission matrix property for the HMMUpdater"):
            wrong_updater.update(empty_hypothesis)

