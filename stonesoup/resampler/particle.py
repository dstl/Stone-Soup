# -*- coding: utf-8 -*-
import numpy as np

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
                         weight=weight,
                         parent=particle,
                         dynamic_model=particle.dynamic_model))

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
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        n_particles = len(particles)
        weight = Probability(1/n_particles)
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
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The resampled particles
        """

        n_particles = len(particles)
        weight = Probability(1/n_particles)
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
                RaoBlackwellisedParticle(particle.state_vector,
                                         weight=weight,
                                         parent=particle,
                                         model_probabilities=particle.model_probabilities))

        return new_particles


class MultiResampler(Resampler):

    def __init__(self, detection_matrix_split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detection_matrix_split = detection_matrix_split

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

        craft_sum = np.cumsum(self.detection_matrix_split)

        particles_per_craft = []
        for i in range(len(self.detection_matrix_split)):
            if i == 0:
                particles_per_craft.append([particle for particle in particles if particle.dynamic_model
                                            in range(self.detection_matrix_split[i])])
            else:
                particles_per_craft.append([particle for particle in particles if particle.dynamic_model
                                            in range(craft_sum[i - 1], craft_sum[i])])

        n_particles = [len(particle) for particle in particles_per_craft]
        weight = Probability(1/sum(n_particles))
        cdf = [list(np.cumsum([p.weight for p in particle])) for particle in particles_per_craft]
        particles_listed = [list(particle) for particle in particles_per_craft]

        # Pick random starting point
        u_i = [np.random.uniform(0, 1 / number_of_particles) for number_of_particles in n_particles]

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        new_particles = []
        for i, craft_particles in enumerate(n_particles):
            for j in range(craft_particles):
                u_j = u_i[i] + (1 / craft_particles) * j
                particle = particles_listed[i][np.argmax(u_j < cdf[i])]
                new_particles.append(
                    RaoBlackwellisedParticle(particle.state_vector,
                                             weight=weight,
                                             parent=particle,
                                             dynamic_model=particle.dynamic_model,
                                             model_probabilities=particle.model_probabilities))
        return new_particles
