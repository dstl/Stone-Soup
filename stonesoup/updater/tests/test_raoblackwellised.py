"""Test for updater.particle module"""
import numpy as np
import datetime

from ...models.measurement.linear import LinearGaussian

from ...updater.particle import RaoBlackwellisedParticleUpdater
from ...types.particle import RaoBlackwellisedParticle
from ...models.transition.linear import ConstantVelocity, ConstantAcceleration
from ...models.transition.linear import CombinedLinearGaussianTransitionModel


def test_rao_blackwellised_updater():
    # Initialise two particles
    particle1 = RaoBlackwellisedParticle(
        state_vector=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        weight=0.5,
        model_probabilities=[0.5, 0.5])
    particle2 = RaoBlackwellisedParticle(
        state_vector=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        weight=0.5,
        model_probabilities=[0.5, 0.5])
    particle1.parent = particle2
    particle2.parent = particle1

    particles = [particle1, particle2]

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

    for particle in particles:
        rv = RaoBlackwellisedParticleUpdater.calculate_model_probabilities(
            particle, measurement_model, position_mapping, transition, dynamic_model_list,
            datetime.timedelta(seconds=1)
        )
        assert type(rv) == list
