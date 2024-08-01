"""Test for updater.particle module"""
import datetime

import numpy as np
import pytest

from ...resampler.particle import SystematicResampler
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import ConstantVelocity, ConstantAcceleration, KnownTurnRate
from ...models.transition.linear import CombinedLinearGaussianTransitionModel
from ...predictor.particle import RaoBlackwellisedMultiModelPredictor, MultiModelPredictor
from ...updater.particle import RaoBlackwellisedParticleUpdater, MultiModelParticleUpdater
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.particle import RaoBlackwellisedParticle, MultiModelParticle
from ...types.prediction import (
    RaoBlackwellisedParticleStatePrediction, MultiModelParticleStatePrediction)
from ...types.state import RaoBlackwellisedParticleState, MultiModelParticleState
from ...types.update import RaoBlackwellisedParticleStateUpdate, MultiModelParticleStateUpdate


@pytest.fixture()
def dynamic_model_list():
    return [CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                   ConstantVelocity(0.01))),
            CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.1),
                                                   ConstantAcceleration(0.1))),
            KnownTurnRate([0.1, 0.1], np.radians(20))]


@pytest.fixture()
def position_mappings():
    return [[0, 1, 3, 4],
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4]]


@pytest.fixture()
def transition_matrix():
    return [[0.40, 0.40, 0.2],
            [0.40, 0.40, 0.2],
            [0.40, 0.40, 0.2]]


@pytest.fixture(params=[None, SystematicResampler])
def resampler(request):
    if request.param is None:
        return None
    else:
        return request.param()


def test_multi_model(dynamic_model_list, position_mappings, transition_matrix, resampler):
    # Initialise particles
    particle1 = MultiModelParticle(
        state_vector=[1, 1, -0.5, 1, 1, -0.5],
        weight=1/3000,
        dynamic_model=0)
    particle2 = MultiModelParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        dynamic_model=1)
    particle3 = MultiModelParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        dynamic_model=2)

    particles = [particle1, particle2, particle3] * 1000
    timestamp = datetime.datetime.now()

    particle_state = MultiModelParticleState(
        None, particle_list=particles, timestamp=timestamp)

    predictor = MultiModelPredictor(
        dynamic_model_list, transition_matrix, position_mappings
    )

    timestamp += datetime.timedelta(seconds=5)
    prediction = predictor.predict(particle_state, timestamp)

    assert isinstance(prediction, MultiModelParticleStatePrediction)

    measurement_model = LinearGaussian(6, [0, 3], np.diag([1, 1]))
    updater = MultiModelParticleUpdater(measurement_model, predictor, resampler=resampler)

    # Detection close to where known turn rate model would place particles
    detection = Detection([[0.5, 7.]], timestamp)

    update = updater.update(hypothesis=SingleHypothesis(prediction, detection))

    assert isinstance(update, MultiModelParticleStateUpdate)
    # NOTE: This uses the parent particles' model, to see which model was applied
    # rather, than the new model that has been changed to.
    model_weights = [
        np.sum(update.weight[update.parent.dynamic_model == model_index])
        for model_index in range(len(dynamic_model_list))]
    assert float(np.sum(model_weights)) == pytest.approx(1)
    assert isinstance(dynamic_model_list[np.argmax(model_weights)], KnownTurnRate)


def test_rao_blackwellised(dynamic_model_list, position_mappings, transition_matrix, resampler):
    # Initialise particles
    particle1 = RaoBlackwellisedParticle(
        state_vector=[1, 1, -0.5, 1, 1, -0.5],
        weight=1/3000,
        model_probabilities=[0.01, 0.98, 0.01])
    particle2 = RaoBlackwellisedParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        model_probabilities=[0.98, 0.01, 0.01])
    particle3 = RaoBlackwellisedParticle(
        state_vector=[1, 1, 0.5, 1, 1, 0.5],
        weight=1/3000,
        model_probabilities=[0.2, 0.2, 0.6])

    particles = [particle1, particle2, particle3] * 1000
    timestamp = datetime.datetime.now()

    particle_state = RaoBlackwellisedParticleState(
        None, particle_list=particles, timestamp=timestamp)

    predictor = RaoBlackwellisedMultiModelPredictor(
        dynamic_model_list, transition_matrix, position_mappings
    )

    timestamp += datetime.timedelta(seconds=5)
    prediction = predictor.predict(particle_state, timestamp)

    assert isinstance(prediction, RaoBlackwellisedParticleStatePrediction)

    measurement_model = LinearGaussian(6, [0, 3], np.diag([1, 1]))
    updater = RaoBlackwellisedParticleUpdater(measurement_model, predictor, resampler=resampler)

    # Detection close to where known turn rate model would place particles
    detection = Detection([[0.5, 7.]], timestamp)

    update = updater.update(hypothesis=SingleHypothesis(prediction, detection))

    assert isinstance(update, RaoBlackwellisedParticleStateUpdate)

    average_model_proabilities = np.average(
        update.model_probabilities, weights=update.weight, axis=1)
    assert len(average_model_proabilities) == update.model_probabilities.shape[0]
    assert isinstance(dynamic_model_list[np.argmax(average_model_proabilities)], KnownTurnRate)
