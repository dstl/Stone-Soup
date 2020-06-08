# -*- coding: utf-8 -*-
"""Test for updater.particle module"""
import numpy as np
import datetime

from ...resampler.particle import RaoBlackwellisedSystematicResampler
from ...models.measurement.linear import LinearGaussian

from ...updater.particle import RaoBlackwellisedParticleUpdater
from ...types.particle import RaoBlackwellisedParticle
from ...models.transition.linear import ConstantVelocity, ConstantAcceleration
from ...models.transition.linear import CombinedLinearGaussianTransitionModel
from ...predictor.particle import RaoBlackwellisedMultiModelPredictor
from ...types.prediction import ParticleState
from ...types.hypothesis import SingleHypothesis
from ...types.numeric import Probability


def test_rao_blackwellised_updater():
    measurement_model = LinearGaussian(
        ndim_state=9,
        mapping=(0, 3, 6),
        noise_covar=np.diag([0.75, 0.75, 0.75]))

    start_time = datetime.datetime.now()

    # Initialise two particles
    particle1 = RaoBlackwellisedParticle(state_vector=np.reshape([1, 1, 1, 1, 1, 1, 1, 1, 1], (-1, 1)), weight=0.5,
                                          model_probabilities=[0.5, 0.5])
    particle2 = RaoBlackwellisedParticle(state_vector=np.reshape([1, 1, 1, 1, 1, 1, 1, 1, 1], (-1, 1)), weight=0.5,
                                          model_probabilities=[0.5, 0.5])
    particle1.parent = particle2
    particle2.parent = particle1

    particles = [particle1, particle2]

    prior_state = ParticleState(particles, timestamp=datetime.timedelta(seconds=1))

    dynamic_model_list = [CombinedLinearGaussianTransitionModel((ConstantVelocity(0.01),
                                                                 ConstantVelocity(0.01),
                                                                 ConstantVelocity(0.01))),
                          CombinedLinearGaussianTransitionModel((ConstantAcceleration(0.1),
                                                                 ConstantAcceleration(0.1),
                                                                 ConstantAcceleration(0.1)))]

    transition = [[0.50, 0.50],
                  [0.50, 0.50]]

    position_mapping = [[0, 1, 3, 4, 6, 7],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8]]

    measurement_model = LinearGaussian(
        ndim_state=9,
        mapping=(0, 3, 6),
        noise_covar=np.diag([0.75, 0.75, 0.75]))
    resampler = RaoBlackwellisedSystematicResampler()

    updater = RaoBlackwellisedParticleUpdater(measurement_model, resampler)

    for particle in particles:
        rv = updater.calculate_model_probabilities(
            particle, position_mapping, transition, dynamic_model_list, prior_state.timestamp
        )
        assert type(rv) == list
