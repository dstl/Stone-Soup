# -*- coding: utf-8 -*-
import numpy as np

from .base import Resampler
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
        weight = Probability(1/n_particles)
        log_weights = np.array([weight.log_value for weight in particles.weight])
        weight_order = np.argsort(log_weights, kind='stable')
        cdf = [v.log_value for v in np.cumsum(particles.weight[weight_order])]

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        u_j = u_i + (1 / n_particles) * np.arange(n_particles)
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]
        new_particles = Particles(state_vector=particles.state_vector[:, index],
                                  weight=[weight]*n_particles,
                                  parent=Particles(state_vector=particles.state_vector[:, index],
                                                   weight=particles.weight[index]))
        return new_particles
