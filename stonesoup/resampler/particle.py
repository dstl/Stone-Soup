# -*- coding: utf-8 -*-
import numpy as np
from operator import attrgetter

from .base import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle, RaoBlackwellisedParticle, MultiModelParticle


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
        weight = Probability(1/n_particles)

        particles_sorted = sorted(particles, key=attrgetter('weight'), reverse=False)
        cdf = np.cumsum([p.weight for p in particles_sorted])

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)
        new_particles = []

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j

            particle = particles_sorted[np.argmax(u_j < cdf)]
            new_particles.append(
                Particle(particle.state_vector,
                         weight=weight,
                         parent=particle))

        return new_particles


class MultiModelSystematicResampler(Resampler):

    def resample(self, particles):
        """Resample the particles

        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.MultiModelParticle`
            The resampled particles
        """

        n_particles = len(particles)
        weight = Probability(1/n_particles)
        particles_sorted = sorted(particles, key=attrgetter('weight'), reverse=False)
        cdf = np.cumsum([p.weight for p in particles_sorted])

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)
        new_particles = []

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j
            particle = particles_sorted[np.argmax(u_j < cdf)]
            new_particles.append(
                MultiModelParticle(particle.state_vector,
                                   weight=weight,
                                   parent=particle,
                                   dynamic_model=particle.dynamic_model))

        return new_particles


class RaoBlackwellisedSystematicResampler(Resampler):

    def resample(self, particles):
        """Resample the particles

        Parameters
        ----------
        particles : list of :class:`~.RaoBlackwellisedParticle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        n_particles = len(particles)
        weight = Probability(1/n_particles)
        particles_sorted = sorted(particles, key=attrgetter('weight'), reverse=False)
        cdf = np.cumsum([p.weight for p in particles_sorted])

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / n_particles)
        new_particles = []

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        for j in range(n_particles):

            u_j = u_i + (1 / n_particles) * j
            particle = particles_sorted[np.argmax(u_j < cdf)]

            new_particles.append(
                RaoBlackwellisedParticle(particle.state_vector,
                                         weight=weight,
                                         parent=particle,
                                         model_probabilities=particle.model_probabilities))

        return new_particles
