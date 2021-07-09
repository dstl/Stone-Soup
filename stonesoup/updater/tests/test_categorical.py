# coding: utf-8
import datetime

import numpy as np
import pytest

from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.measurement.linear import LinearGaussian
from ...models.transition.tests.test_categorical import create_categorical, \
    create_categorical_matrix
from ...types.detection import CategoricalDetection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import CategoricalMeasurementPrediction
from ...types.state import CategoricalState, GaussianState
from ...types.update import CategoricalStateUpdate
from ...updater.categorical import HMMUpdater


@pytest.mark.parametrize('ndim_state', (2, 3, 4))
@pytest.mark.parametrize('ndim_meas', (2, 3, 4))
def test_classification_updater(ndim_state, ndim_meas):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)

    E = create_categorical_matrix(ndim_state, ndim_meas)
    Ecov = 0.1 * np.eye(ndim_meas)
    measurement_model = CategoricalMeasurementModel(ndim_state=ndim_state,
                                                    emission_matrix=E,
                                                    emission_covariance=Ecov)

    prediction = CategoricalState(create_categorical(ndim_state))
    prediction.timestamp = future

    updater = HMMUpdater(measurement_model)
    empty_updater = HMMUpdater()
    wrong_updater = HMMUpdater(LinearGaussian(ndim_state=ndim_state,
                                              mapping=[0, 1],
                                              noise_covar=np.eye(ndim_state)))

    # Test measurement prediction
    eval_measurement_prediction = measurement_model.function(prediction)
    measurement_prediction = updater.predict_measurement(prediction)
    assert isinstance(measurement_prediction, CategoricalMeasurementPrediction)
    assert (np.allclose(measurement_prediction.state_vector,
                        eval_measurement_prediction,
                        0,
                        atol=1.e-14))

    E = measurement_model.emission_matrix
    measurement = measurement_model.function(CategoricalState(create_categorical(E.shape[0])))
    hypothesis = SingleHypothesis(
        prediction=prediction,
        measurement=CategoricalDetection(measurement, timestamp=future,
                                         measurement_model=measurement_model)
    )

    # Test get emission matrix
    assert (updater._get_emission_matrix(hypothesis) == E).all()

    # Test update with and without measurement prediction
    for meas_pred in [None, measurement_prediction]:
        hypothesis.measurement_prediction = meas_pred
        eval_posterior = np.multiply(prediction.state_vector, E @ measurement)
        eval_posterior = eval_posterior / np.sum(eval_posterior)
        posterior = updater.update(hypothesis)

        assert isinstance(posterior, CategoricalStateUpdate)
        assert posterior.category_names == prediction.category_names
        assert (np.allclose(posterior.state_vector, eval_posterior, 0, atol=1.e-14))
        assert posterior.timestamp == prediction.timestamp

        # Assert presence of measurement
        assert hasattr(posterior, 'hypothesis')
        assert posterior.hypothesis == hypothesis

    # Test measurement model error
    empty_hypothesis = SingleHypothesis(
        prediction=prediction,
        measurement=CategoricalDetection(measurement, timestamp=future)
    )

    with pytest.raises(ValueError,
                       match="No measurement model specified"):
        empty_updater.update(empty_hypothesis)
    with pytest.raises(ValueError,
                       match="Measurement model must be categorical. I.E. it must have an "
                             "Emission matrix property for the HMMUpdater"):
        wrong_updater.update(empty_hypothesis)

    # Test non-categorical prediction

    bad_hypothesis = SingleHypothesis(
        prediction=GaussianState(create_categorical(ndim_state), np.eye(ndim_state)),
        measurement=CategoricalDetection(measurement, timestamp=future)
    )

    with pytest.raises(ValueError, match="Prediction must be a categorical state type"):
        updater.update(bad_hypothesis)
