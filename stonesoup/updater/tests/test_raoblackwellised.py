"""Test for updater.particle module"""
import datetime

import numpy as np

from stonesoup.types.detection import Detection
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import ConstantVelocity, ConstantAcceleration, KnownTurnRate
from ...models.transition.linear import CombinedLinearGaussianTransitionModel
from ...predictor.particle import RaoBlackwellisedMultiModelPredictor
from ...updater.particle import RaoBlackwellisedParticleUpdater
from ...types.hypothesis import SingleHypothesis
from ...types.particle import RaoBlackwellisedParticle
from ...types.prediction import RaoBlackwellisedParticleStatePrediction
from ...types.state import RaoBlackwellisedParticleState
from ...types.update import RaoBlackwellisedParticleStateUpdate


def test_rao_blackwellised():
    # Initialise two particles
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

    dynamic_model_list = [CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                                 ConstantVelocity(0.01))),
                          CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.1),
                                                                 ConstantAcceleration(0.1))),
                          KnownTurnRate([0.1, 0.1], np.radians(20))]

    transition_matrix = [[0.40, 0.40, 0.2],
                         [0.40, 0.40, 0.2],
                         [0.40, 0.40, 0.2]]

    position_mappings = [[0, 1, 3, 4],
                         [0, 1, 2, 3, 4, 5],
                         [1, 2, 3, 4]]

    predictor = RaoBlackwellisedMultiModelPredictor(
        dynamic_model_list, transition_matrix, position_mappings
    )

    timestamp += datetime.timedelta(seconds=5)
    prediction = predictor.predict(particle_state, timestamp)

    assert isinstance(prediction, RaoBlackwellisedParticleStatePrediction)

    measurement_model = LinearGaussian(6, [0, 3], np.diag([1, 1]))
    updater = RaoBlackwellisedParticleUpdater(measurement_model, predictor)

    # Detection close to where known turn rate model would place particles
    detection = Detection([[0.5, 7.]], timestamp)

    update = updater.update(hypothesis=SingleHypothesis(prediction, detection))

    assert isinstance(update, RaoBlackwellisedParticleStateUpdate)

    average_model_proabilities = np.average(
        update.model_probabilities, weights=update.weight, axis=1)
    assert len(average_model_proabilities) == update.model_probabilities.shape[0]
    assert isinstance(dynamic_model_list[np.argmax(average_model_proabilities)], KnownTurnRate)
