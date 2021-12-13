# -*- coding: utf-8 -*-
import numpy as np

from .base import Resampler
from ..base import Property
from ..types.numeric import Probability
from ..types.particle import Particles


class SystematicResampler(Resampler):

    def resample(self, particles):
        """Resample the particles

        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        if not isinstance(particles, Particles):
            particles = Particles(particle_list=particles)
        n_particles = len(particles)
        weight = Probability(1 / n_particles)

        log_weights = np.array([weight.log_value for weight in particles.weight])
        weight_order = np.argsort(log_weights, kind='stable')
        max_log_value = log_weights[weight_order[-1]]
        with np.errstate(divide='ignore'):
            cdf = np.log(np.cumsum(np.exp(log_weights[weight_order] - max_log_value)))
        cdf += max_log_value

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        u_j = u_i + (1 / n_particles) * np.arange(n_particles)
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]
        new_particles = Particles(state_vector=particles.state_vector[:, index],
                                  weight=[weight] * n_particles,
                                  parent=Particles(state_vector=particles.state_vector[:, index],
                                                   weight=particles.weight[index]))
        return new_particles


class ESSResampler(SystematicResampler):

    threshold: float = Property(default=None,
                                doc='Threshold compared with ESS to decide whether to resample')

    def resample(self, particles):
        if not isinstance(particles, Particles):
            particles = Particles(particle_list=particles)
        if self.threshold is None:
            self.threshold = len(particles) / 2
        if 1 / np.sum(np.square(particles.weight)) < self.threshold:  # If ESS too small, resample
            return super().resample(particles)
        else:
            return particles
