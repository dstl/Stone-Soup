# coding: utf-8
import datetime
from copy import copy

import numpy as np
import pytest

from ...models.measurement.categorical import MarkovianMeasurementModel
from ...models.measurement.linear import LinearGaussian
from ...predictor.tests.test_composite import create_state
from ...types.detection import Detection, CompositeDetection, CategoricalDetection, MissedDetection
from ...types.hypothesis import SingleHypothesis, CompositeHypothesis
from ...types.prediction import CompositePrediction
from ...types.state import State
from ...types.update import CompositeUpdate
from ...updater.categorical import HMMUpdater
from ...updater.composite import CompositeUpdater
from ...updater.kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater
from ...updater.particle import ParticleUpdater


def create_measurement_model(gaussian: bool, ndim_state: int):
    """Generate measurement models of particular dimensions"""
    if gaussian:
        return LinearGaussian(ndim_state=ndim_state,
                              noise_covar=np.eye(ndim_state),
                              mapping=np.arange(ndim_state))
    else:
        return MarkovianMeasurementModel(
            emission_matrix=np.random.rand(ndim_state, ndim_state)
        )


def random_updater_prediction_and_measurement(num_updaters, timestamp):
    """Create a random composite updater, prediction and measurement"""

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
            state_vector=create_state(True, False, ndim_states[0], timestamp).state_vector,
            timestamp=timestamp,
            measurement_model=sub_updaters[0].measurement_model),
        Detection(
            state_vector=create_state(True, False, ndim_states[1], timestamp).state_vector,
            timestamp=timestamp,
            measurement_model=sub_updaters[1].measurement_model),
        Detection(
            state_vector=create_state(True, False, ndim_states[2], timestamp).state_vector,
            timestamp=timestamp,
            measurement_model=sub_updaters[2].measurement_model),
        Detection(
            state_vector=create_state(True, True, ndim_states[3], timestamp).state_vector,
            timestamp=timestamp,
            measurement_model=sub_updaters[3].measurement_model),
        CategoricalDetection(
            state_vector=create_state(False, False, ndim_states[4], timestamp).state_vector,
            timestamp=timestamp,
            measurement_model=sub_updaters[4].measurement_model)
    ]

    updater = CompositeUpdater(sub_updaters[:num_updaters])
    prediction = CompositePrediction(sub_predictions[:num_updaters])
    measurement = CompositeDetection(sub_measurements[:num_updaters])

    return updater, prediction, measurement


@pytest.mark.parametrize('num_updaters', [1, 2, 3, 4, 5])
def test_composite_updater(num_updaters):
    now = datetime.datetime.now()

    updater, prediction, measurement = random_updater_prediction_and_measurement(num_updaters, now)

    sub_updaters = updater.sub_updaters

    # Test instantiation errors
    with pytest.raises(
            ValueError,
            match="Sub-updaters must be defined as an ordered list, not <class 'set'>"):
        CompositeUpdater(set(sub_updaters))

    with pytest.raises(ValueError, match="Cannot create an empty composite updater"):
        CompositeUpdater(list())

    with pytest.raises(ValueError, match="All sub-updaters must be a Updater type"):
        CompositeUpdater(sub_updaters + [1, 2, 3])

    updater = CompositeUpdater(sub_updaters)

    # Test measurement model error
    with pytest.raises(NotImplementedError,
                       match="A composition of updaters has no defined measurement model"):
        updater.measurement_model

    # Test predict measurement error
    with pytest.raises(NotImplementedError,
                       match="A composite updater has no method to predict a measurement"):
        updater.predict_measurement()

    # Test update

    sub_hypotheses = list()
    eval_sub_meas_preds = list()
    eval_sub_updates = list()

    # calculate expected update
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

    # hypothesis with null sub-hypothesis
    alt_sub_measurements = copy(measurement.sub_states)
    alt_sub_measurements[0] = MissedDetection(timestamp=measurement.timestamp)
    alt_measurement = CompositeDetection(alt_sub_measurements)
    alt_sub_hypotheses = copy(sub_hypotheses)
    alt_sub_hypotheses[0] = SingleHypothesis(prediction=prediction[0],
                                             measurement=alt_sub_measurements[0])
    alt_hypothesis = CompositeHypothesis(prediction, alt_measurement,
                                         sub_hypotheses=alt_sub_hypotheses)
    alt_update = updater.update(alt_hypothesis)
    alt_eval_sub_updates = copy(eval_sub_updates)
    alt_eval_sub_updates[0] = prediction[0]
    alt_eval_update = CompositeUpdate(sub_states=alt_eval_sub_updates, hypothesis=alt_hypothesis)

    assert isinstance(alt_update, CompositeUpdate)
    assert len(alt_update) == len(prediction)
    assert alt_update.hypothesis == alt_hypothesis
    for i, (sub_update, eval_sub_update) in enumerate(zip(alt_update, alt_eval_update)):
        assert (np.allclose(sub_update.state_vector,
                            eval_sub_update.state_vector,
                            0,
                            atol=1.e-14))
        if i > 0:
            assert sub_update.hypothesis == eval_sub_update.hypothesis
    assert (np.allclose(alt_update.state_vector,
                        alt_eval_update.state_vector,
                        0,
                        atol=1.e-14))

    # Test update error
    with pytest.raises(ValueError,
                       match="CompositeUpdater can only update with CompositeHypothesis types"):
        updater.update(State([0]))
    if num_updaters > 1:
        with pytest.raises(ValueError,
                           match=f"Mismatch in number of sub-hypotheses {num_updaters - 1} and "
                           f"number of sub-updaters {num_updaters}"):
            updater.update(hypothesis[1:])

    # Test contains
    for sub_updater in sub_updaters:
        assert sub_updater in updater
    assert KalmanUpdater(create_measurement_model(True, 5)) not in updater
    assert 'a' not in updater

    # Test get
    for i in range(num_updaters):
        assert updater[i] == sub_updaters[i]
    # Test get slice
    if num_updaters > 1:
        updater_slice = updater[1:]
        assert isinstance(updater_slice, CompositeUpdater)
        assert updater_slice.sub_updaters == sub_updaters[1:]

    # Test iter
    for i, exp_sub_updater in enumerate(updater):
        assert exp_sub_updater == sub_updaters[i]

    # Test len
    assert len(updater) == num_updaters
