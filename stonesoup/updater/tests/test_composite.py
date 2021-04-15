# coding: utf-8
import datetime

import numpy as np
import pytest

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.models.measurement.classification import BasicTimeInvariantObservationModel
from stonesoup.predictor.tests.test_classification import create_random_multinomial
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection, CompositeDetection
from stonesoup.types.hypothesis import SingleHypothesis, CompositeHypothesis
from stonesoup.types.update import StateUpdate, CompositeUpdate
from stonesoup.updater.classification import ClassificationUpdater
from stonesoup.updater.composite import CompositeUpdater
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater
from stonesoup.updater.particle import ParticleUpdater, GromovFlowParticleUpdater
from stonesoup.updater.pointprocess import PointProcessUpdater
from ...types.state import State

# coding: utf-8
import datetime

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ...models.transition.classification import \
    BasicTimeInvariantClassificationTransitionModel
from ...models.transition.linear import RandomWalk, \
    CombinedLinearGaussianTransitionModel
from ...predictor.composite import CompositePredictor
from ...types.array import StateVector
from ...types.array import StateVectors
from ...types.numeric import Probability  # Similar to a float type
from ...types.particle import Particles
from ...types.prediction import CompositePrediction, CompositeMeasurementPrediction
from ...types.state import ParticleState
from ...predictor.classification import ClassificationPredictor
from ...predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor, \
    UnscentedKalmanPredictor
from ...predictor.particle import ParticlePredictor, ParticleFlowKalmanPredictor
from ...types.state import State, GaussianState, CompositeState
from ...predictor.tests.test_composite import create_state


def create_measurement_model(gaussian: bool, ndim_state: int):
    """Generate appropriate measurement models of particular dimensions"""
    if gaussian:
        return LinearGaussian(ndim_state=ndim_state,
                              noise_covar=np.eye(ndim_state),
                              mapping=np.arange(ndim_state))
    else:
        rows = list()
        for _ in range(np.random.randint(1, 10)):
            rows.append(create_random_multinomial(ndim_state).state_vector.flatten())
        ET = np.array(rows)
        return BasicTimeInvariantObservationModel(ET.T)


def get_sub_updaters(num_updaters):
    possible_updaters = [KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater,
                         ParticleUpdater, ClassificationUpdater]
    sub_updater_types = possible_updaters[:num_updaters]

    sub_updaters = list()
    for sub_updater_type in sub_updater_types:
        ndim_state = np.random.randint(2, 10)

        if sub_updater_type == ClassificationUpdater:
            gaussian = False
        else:
            gaussian = True

        measurement_model = create_measurement_model(gaussian, ndim_state)

        sub_updaters.append(sub_updater_type(measurement_model))

    return sub_updaters


@pytest.mark.parametrize('num_updaters', [1, 2, 3, 4, 5])
def test_composite_updater(num_updaters):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)

    # get random sub-updaters
    sub_updaters = get_sub_updaters(num_updaters)

    # create appropriate predictions and measurements
    sub_predictions = list()
    sub_measurements = list()
    for sub_updater in sub_updaters:
        if isinstance(sub_updater, ClassificationUpdater):
            gaussian = False
            particles = False
        else:
            gaussian = True
            if isinstance(sub_updater, (ParticleUpdater, GromovFlowParticleUpdater)):
                particles = True
            else:
                particles = False

        sub_measurement_model = sub_updater.measurement_model

        # generate sub-predictions
        state = create_state(gaussian, particles, sub_measurement_model.ndim_state)
        state.timestamp = future
        sub_predictions.append(state)

        # generate sub-measurements (not particle)
        sub_measurement = create_state(gaussian, False, sub_measurement_model.ndim_meas)
        sub_measurements.append(Detection(state_vector=sub_measurement.state_vector,
                                          timestamp=future,
                                          measurement_model=sub_measurement_model))

    prediction = CompositePrediction(sub_predictions)
    measurement = CompositeDetection(sub_measurements)

    # test instantiation errors
    with pytest.raises(ValueError, match="sub-updaters must be defined as an ordered list"):
        CompositeUpdater(set(sub_updaters))

    with pytest.raises(ValueError, match="all sub-updaters must be an Updater type"):
        CompositeUpdater(sub_updaters + [1, 2, 3])

    updater = CompositeUpdater(sub_updaters)

    # test measurement model error
    with pytest.raises(NotImplementedError,
                       match="A composition of updaters have no defined measurement model"):
        updater.measurement_model

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
    eval_measurement_prediction = CompositeMeasurementPrediction(eval_sub_meas_preds)
    eval_update = CompositeUpdate(sub_states=eval_sub_updates, hypothesis=hypothesis)

    # test update (without measurement prediction)
    assert isinstance(update, CompositeUpdate)
    assert len(update) == len(prediction)
    assert update.hypothesis.prediction == eval_update.hypothesis.prediction
    assert update.hypothesis.measurement == eval_update.hypothesis.measurement
    assert update.hypothesis.measurement_prediction == eval_update.hypothesis.measurement_prediction
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
    assert update.hypothesis == eval_update.hypothesis
    assert len(update) == len(eval_update)

    # test update error
    with pytest.raises(ValueError,
                       match="CompositeUpdater can only be used with CompositeHypothesis types"):
        updater.update(State([0]))
    with pytest.raises(ValueError,
                       match="CompositeHypothesis must be composed of same number of "
                             "sub-hypotheses as sub-updaters"):
        del hypothesis[0]
        updater.update(hypothesis)
