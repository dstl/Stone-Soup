from functools import lru_cache

import numpy as np

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types import Particle,ParticleState

class ParticleUpdater(Updater):
    """Simple Kalman Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(
        Resampler, doc = 'Resampler to prevent particle degeneracy')


    def update(self, state_pred, meas, innovation, **kwargs):
        """Kalman Filter update step

        Parameters
        ----------
        state_pred : :class:`ParticleState`
            The state prediction
        meas_pred : :UNUSED BUT KEPT TO MAINTAIN PARITY WITH KALMAN UPDATER

        meas : :class:`Detection`
            The measurement

        Returns
        -------
        :class:`ParticleState`
            The state posterior
        :UNUSED BUT KEPT TO MAINTAIN PARITY WITH KALMAN UPDATER
        """

        if meas:
            for particle in state_pred.particles:
                particle.weight = self.measurement_model.pdf(
                    meas.state_vector, particle.state_vector)

            # Normalise the weights
            sum_w = sum(i.weight for i in state_pred.particles)
            if sum_w == 0:
                raise RuntimeError(
                    'Sum of weights is equal to zero; track lost')
            for particle in state_pred.particles:
                particle.weight /= sum_w

            new_particles = self.resampler.resample(state_pred.particles)

        return ParticleState(new_particles,timestamp = state_pred.timestamp), None

    @lru_cache()
    def get_measurement_prediction(self, state_pred):

        new_particles=[]
        for particle in state_pred.particles:
            new_state_vector = self.measurement_model.function(particle.state_vector, noise = 0)
            new_particles.append(
                Particle(new_state_vector, weight=particle.weight, timestamp=state_pred.timestamp, parent=particle.parent))

        return ParticleState(new_particles,timestamp = state_pred.timestamp), None
