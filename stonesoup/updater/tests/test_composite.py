# coding: utf-8
import datetime

import numpy as np
import pytest

from ...models.measurement.categorical import CategoricalMeasurementModel
from ...models.measurement.linear import LinearGaussian
from ...models.transition.tests.test_categorical import create_categorical_matrix
from ...predictor.tests.test_composite import create_state
from ...types.detection import Detection, CompositeDetection, CategoricalDetection
from ...types.hypothesis import SingleHypothesis, CompositeHypothesis
from ...types.prediction import CompositePrediction
from ...types.state import State
from ...types.update import CompositeUpdate
from ...updater.categorical import HMMUpdater
from ...updater.composite import CompositeUpdater
from ...updater.kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater
from ...updater.particle import ParticleUpdater


def create_measurement_model(gaussian: bool, ndim_state: int):
    """Generate appropriate measurement models of particular dimensions"""
    if gaussian:
        return LinearGaussian(ndim_state=ndim_state,
                              noise_covar=np.eye(ndim_state),
                              mapping=np.arange(ndim_state))
    else:
        return CategoricalMeasurementModel(create_categorical_matrix(ndim_state, ndim_state),
                                           0.1 * np.eye(2))


def random_updater_prediction_and_measurement(num_updaters, timestamp, future_timestamp):
    ndim_states = np.random.randint(2, 5, 5)

    sub_updaters = [KalmanUpdater(create_measurement_model(True, ndim_states[0])),
                    ExtendedKalmanUpdater(create_measurement_model(True, ndim_states[1])),
                    UnscentedKalmanUpdater(create_measurement_model(True, ndim_states[2])),
                    ParticleUpdater(create_measurement_model(True, ndim_states[3])),
                    HMMUpdater(create_measurement_model(False, ndim_states[4]))]

    sub_predictions = [
        create_state(True, False, ndim_states[0], timestamp),
        create_state(True, False, ndim_states[1], timestamp),
        create_state(True, False, ndim_states[2], timestamp),
        create_state(True, True, ndim_states[3], timestamp),
        create_state(False, False, ndim_states[4], timestamp)
    ]

    sub_measurements = [
        Detection(
            state_vector=create_state(True, False, ndim_states[0], future_timestamp).state_vector,
            timestamp=future_timestamp,
            measurement_model=sub_updaters[0].measurement_model),
        Detection(
            state_vector=create_state(True, False, ndim_states[1], future_timestamp).state_vector,
            timestamp=future_timestamp,
            measurement_model=sub_updaters[1].measurement_model),
        Detection(
            state_vector=create_state(True, False, ndim_states[2], future_timestamp).state_vector,
            timestamp=future_timestamp,
            measurement_model=sub_updaters[2].measurement_model),
        Detection(
            state_vector=create_state(True, True, ndim_states[3], future_timestamp).state_vector,
            timestamp=future_timestamp,
            measurement_model=sub_updaters[3].measurement_model),
        CategoricalDetection(
            state_vector=create_state(False, False, ndim_states[4], future_timestamp).state_vector,
            timestamp=future_timestamp,
            measurement_model=sub_updaters[4].measurement_model)
    ]

    updater = CompositeUpdater(sub_updaters[:num_updaters])
    prediction = CompositePrediction(sub_predictions[:num_updaters])
    measurement = CompositeDetection(sub_measurements[:num_updaters])

    return updater, prediction, measurement


@pytest.mark.parametrize('num_updaters', [1, 2, 3, 4, 5])
def test_composite_updater(num_updaters):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)

    updater, prediction, measurement = random_updater_prediction_and_measurement(num_updaters,
                                                                                 now,
                                                                                 future)

    sub_updaters = updater.sub_updaters

    # Test instantiation errors
    with pytest.raises(ValueError, match="sub-updaters must be defined as an ordered list"):
        CompositeUpdater(set(sub_updaters))

    with pytest.raises(ValueError, match="all sub-updaters must be an Updater type"):
        CompositeUpdater(sub_updaters + [1, 2, 3])

    updater = CompositeUpdater(sub_updaters)

    # Test measurement model error
    with pytest.raises(NotImplementedError,
                       match="A composition of updaters have no defined measurement model"):
        updater.measurement_model

    # Test predict measurement error
    with pytest.raises(NotImplementedError,
                       match="A composite updater has no method to predict a measurement"):
        updater.predict_measurement()

    # Test update
    sub_hypotheses = list()
    eval_sub_meas_preds = list()
    eval_sub_updates = list()
    for sub_prediction, sub_measurement, sub_updater in zip(prediction, measurement, sub_updaters):
        sub_hypothesis = SingleHypothesis(prediction=sub_prediction, measurement=sub_measurement)
        sub_hypotheses.append(sub_hypothesis)

        sub_meas_model = sub_measurement.measurement_model

        eval_sub_meas_preds.append(sub_updater.predict_measurement(sub_prediction, sub_meas_model))
        eval_sub_updates.append(sub_updater.update(sub_hypothesis))

    hypothesis = CompositeHypothesis(prediction, measurement, sub_hypotheses=sub_hypotheses)

    update = updater.update(hypothesis)
    eval_update = CompositeUpdate(sub_states=eval_sub_updates, hypothesis=hypothesis)

    # Test update
    assert isinstance(update, CompositeUpdate)
    assert len(update) == len(prediction)
    assert update.hypothesis == hypothesis
    for sub_update, eval_sub_update in zip(update, eval_update):
        assert (np.allclose(sub_update.state_vector,
                            eval_sub_update.state_vector,
                            0,
                            atol=1.e-14))
        assert sub_update.hypothesis == eval_sub_update.hypothesis
    assert (np.allclose(update.state_vector,
                        eval_update.state_vector,
                        0,
                        atol=1.e-14))

    # Test update error
    with pytest.raises(ValueError,
                       match="CompositeUpdater can only be used with CompositeHypothesis types"):
        updater.update(State([0]))
    with pytest.raises(ValueError,
                       match="CompositeHypothesis must be composed of same number of "
                             "sub-hypotheses as sub-updaters"):
        del hypothesis[0]
        updater.update(hypothesis)

    # Test iter
    for i, exp_sub_updater in enumerate(updater):
        assert exp_sub_updater == sub_updaters[i]

    # Test len
    assert len(updater) == num_updaters

    # Test get
    for i in range(num_updaters):
        assert updater[i] == sub_updaters[i]

    updater_slice = updater[:num_updaters - 1]
    assert isinstance(updater_slice, CompositeUpdater)
    assert len(updater_slice) == num_updaters - 1
    for i, expected_updater in enumerate(sub_updaters[:num_updaters - 1]):
        assert updater_slice[i] == expected_updater

    # Test contains
    for sub_updater in sub_updaters:
        assert sub_updater in updater

    # Test append
    new_sub_updater = KalmanUpdater(LinearGaussian(ndim_state=4,
                                                   noise_covar=np.eye(4),
                                                   mapping=np.arange(4)))
    updater.append(new_sub_updater)
    assert new_sub_updater in updater
    assert len(updater) == num_updaters + 1
