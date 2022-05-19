# -*- coding: utf-8 -*-
import numpy as np

from .base import Resampler
from ..base import Property
from ..types.numeric import Probability
from ..types.state import ParticleState


class SystematicResampler(Resampler):

    def resample(self, particles):
        """Resample the particles

        Parameters
        ----------
        particles : :class:`~.ParticleState` or list of :class:`~.Particle`
            The particles or particle state to be resampled according to their weights

        Returns
        -------
        particle state: :class:`~.ParticleState`
            The particle state after resampling
        """

        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)
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
        new_particles = ParticleState(state_vector=particles.state_vector[:, index],
                                      weight=[weight]*n_particles,
                                      parent=ParticleState(
                                          state_vector=particles.state_vector[:, index],
                                          weight=particles.weight[index],
                                          timestamp=particles.timestamp),
                                      timestamp=particles.timestamp)
        return new_particles


class ESSResampler(Resampler):
    """ This wrapper uses a :class:`~.Resampler` to resample the particles inside
        an instant of :class:`~.Particles`, but only after checking if this is necessary
        by comparing Effective Sample Size (ESS) with a supplied threshold (numeric).
        Kish's ESS is used, as recommended in Section 3.5 of this tutorial [1]_.

        References
        ----------
        .. [1] Doucet A., Johansen A.M., 2009, Tutorial on Particle Filtering \
        and Smoothing: Fifteen years later, Handbook of Nonlinear Filtering, Vol. 12.

        """

    threshold: float = Property(default=None,
                                doc='Threshold compared with ESS to decide whether to resample. \
                                    Default is number of particles divided by 2, \
                                        set in resample method')
    resampler: Resampler = Property(default=SystematicResampler,
                                    doc='Resampler to wrap, which is called \
                                        when ESS below threshold')

    def resample(self, particles):
        """
        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight

        Returns
        -------
        particles : list of :class:`~.Particle`
            The particles, either unchanged or resampled, depending on weight degeneracy
        """
        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)
        if self.threshold is None:
            self.threshold = len(particles) / 2
        if 1 / np.sum(np.square(particles.weight)) < self.threshold:  # If ESS too small, resample
            return self.resampler.resample(self.resampler, particles)
        else:
            return particles
