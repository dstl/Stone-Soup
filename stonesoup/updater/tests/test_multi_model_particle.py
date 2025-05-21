"""Test for updater.particle module"""
import datetime
from functools import partial

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import CombinedLinearGaussianTransitionModel
from ...models.transition.linear import ConstantVelocity, ConstantAcceleration, KnownTurnRate
from ...predictor.particle import RaoBlackwellisedMultiModelPredictor, MultiModelPredictor
from ...regulariser.particle import MultiModelMCMCRegulariser
from ...resampler.particle import SystematicResampler, ESSResampler
from ...types.array import StateVectors
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.particle import RaoBlackwellisedParticle
from ...types.prediction import (
    RaoBlackwellisedParticleStatePrediction, MultiModelParticleStatePrediction)
from ...types.state import RaoBlackwellisedParticleState, MultiModelParticleState
from ...types.update import RaoBlackwellisedParticleStateUpdate, MultiModelParticleStateUpdate
from ...updater.particle import RaoBlackwellisedParticleUpdater, MultiModelParticleUpdater


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


@pytest.fixture(params=[
    None, SystematicResampler, ESSResampler,
    pytest.param(partial(ESSResampler, threshold=0), id='avoid_resample')])
def resampler(request):
    if request.param is None:
        return None
    else:
        return request.param()


@pytest.fixture(params=[None, MultiModelMCMCRegulariser])
def regulariser(request, dynamic_model_list, position_mappings, resampler):
    if request.param is None:
        return None
    else:
        return request.param(dynamic_model_list, position_mappings)


@pytest.fixture(params=[True, False])
def constraint_func(request):
    if request.param:
        def func(particles):
            part_indx = particles.state_vector[0, :] < 0
            return part_indx
        return func


def test_multi_model(
        dynamic_model_list, position_mappings, transition_matrix, resampler, regulariser,
        constraint_func):

    state_vector = StateVectors(multivariate_normal.rvs(
        [1, 1, 0.5, 1, 1, 0.5],
        np.diag([0.2, 0.07, 0.02]*2),
        3000).T)

    timestamp = datetime.datetime.now()

    particle_state = MultiModelParticleState(
        state_vector,
        log_weight=np.full((3000, ), np.log(1/3000)),
        dynamic_model=np.array([0, 1, 2]*1000),
        timestamp=timestamp)

    predictor = MultiModelPredictor(
        dynamic_model_list, transition_matrix, position_mappings
    )

    timestamp += datetime.timedelta(seconds=5)
    prediction = predictor.predict(particle_state, timestamp)

    assert isinstance(prediction, MultiModelParticleStatePrediction)

    measurement_model = LinearGaussian(6, [0, 3], np.diag([2, 2]))
    updater = MultiModelParticleUpdater(
        measurement_model, predictor, resampler=resampler, regulariser=regulariser,
        constraint_func=constraint_func)

    # Detection close to where known turn rate model would place particles
    detection = Detection([[0.5, 7.]], timestamp, measurement_model=measurement_model)

    update = updater.update(hypothesis=SingleHypothesis(prediction, detection))

    assert isinstance(update, MultiModelParticleStateUpdate)
    # NOTE: This uses the parent particles' model, to see which model was applied
    # rather, than the new model that has been changed to.
    model_weights = [
        np.sum(update.weight[update.parent.dynamic_model == model_index])
        for model_index in range(len(dynamic_model_list))]
    if not constraint_func:
        assert float(np.sum(model_weights)) == pytest.approx(1)
    assert isinstance(dynamic_model_list[np.argmax(model_weights)], KnownTurnRate)


def test_rao_blackwellised(
        dynamic_model_list, position_mappings, transition_matrix, resampler, constraint_func):
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
    updater = RaoBlackwellisedParticleUpdater(
        measurement_model, predictor, resampler=resampler, constraint_func=constraint_func)

    # Detection close to where known turn rate model would place particles
    detection = Detection([[0.5, 7.]], timestamp)

    update = updater.update(hypothesis=SingleHypothesis(prediction, detection))

    assert isinstance(update, RaoBlackwellisedParticleStateUpdate)

    average_model_proabilities = np.average(
        update.model_probabilities, weights=update.weight, axis=1)
    assert len(average_model_proabilities) == update.model_probabilities.shape[0]
    assert isinstance(dynamic_model_list[np.argmax(average_model_proabilities)], KnownTurnRate)
