# -*- coding: utf-8 -*-
import numpy as np

from .base import Resampler
from ..types.particle import Particle


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

        n_particles = len(particles)
        cdf = np.cumsum([p.weight for p in particles])
        particles_listed = list(particles)
        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)
        new_particles = []

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j

            particle = particles_listed[np.argmax(u_j < cdf)]
            new_particles.append(
                Particle(particle.state_vector,
                         weight=1 / n_particles,
                         parent=particle))

        return new_particles
