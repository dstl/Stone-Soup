# -*- coding: utf-8 -*-
from functools import lru_cache

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import ParticleMeasurementPrediction
from ..types.update import ParticleStateUpdate


class ParticleUpdater(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, prediction, measurement,
               measurement_prediction=None, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        prediction : :class:`~.ParticleStatePrediction`
            The state prediction
        measurement : :class:`~.Detection`
            The measurement
        measurement_prediction : None
            Not required and ignored if passed.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """

        for particle in prediction.particles:
            particle.weight *= self.measurement_model.pdf(
                measurement.state_vector, particle.state_vector, **kwargs)

        # Normalise the weights
        sum_w = Probability.sum(i.weight for i in prediction.particles)
        for particle in prediction.particles:
            particle.weight /= sum_w

        # Resample
        new_particles = self.resampler.resample(prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   prediction,
                                   measurement_prediction,
                                   measurement,
                                   prediction.timestamp)

    @lru_cache()
    def get_measurement_prediction(self, state_prediction, **kwargs):
        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = self.measurement_model.function(
                particle.state_vector, noise=0, **kwargs)
            new_particles.append(
                Particle(new_state_vector,
                         weight=particle.weight,
                         parent=particle.parent))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)
