import numpy as np

from .base import Resampler
from ..base import Property
from ..types.state import ParticleState


class SystematicResampler(Resampler):
    """
    Traditional style resampler for particle filter. Calculates first random point in (0, 1/nparts],
    then calculates |nparts| points that are equidistantly distributed across the cdf. Complexity
    of order O(N) where N is the number of resampled particles.

    """

    def resample(self, particles, nparts=None):
        """Resample the particles

        Parameters
        ----------
        particles : :class:`~.ParticleState` or list of :class:`~.Particle`
            The particles or particle state to be resampled according to their weights
        nparts : int
            The number of particles to be returned from resampling

        Returns
        -------
        particle state: :class:`~.ParticleState`
            The particle state after resampling
        """

        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)
        if nparts is None:
            nparts = len(particles)

        log_weights = particles.log_weight
        weight_order = np.argsort(log_weights, kind='stable')
        max_log_value = log_weights[weight_order[-1]]
        with np.errstate(divide='ignore'):
            cdf = np.log(np.cumsum(np.exp(log_weights[weight_order] - max_log_value)))
        cdf += max_log_value

        # Pick random starting point
        u_i = np.random.uniform(0, 1 / nparts)

        # Cycle through the cumulative distribution and copy the particle
        # that pushed the score over the current value
        u_j = u_i + (1 / nparts) * np.arange(nparts)
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]

        new_particles = particles[index]
        new_particles.log_weight = np.full((nparts, ), np.log(1/nparts))
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
        # If ESS too small, resample
        if 1 / np.sum(np.exp(2*particles.log_weight)) < self.threshold:
            return self.resampler.resample(self.resampler, particles)
        else:
            return particles


class MultinomialResampler(Resampler):
    """
    Traditional style resampler for particle filter. Calculates a random point in (0, 1]
    individually for each particle, and picks the corresponding particle from the cdf calculated
    from particle weights. Complexity is of order O(NM) where N and M are the number of resampled
    and existing particles respectively.
    """

    def resample(self, particles, nparts=None):
        """Resample the particles

        Parameters
        ----------
        particles : :class:`~.ParticleState` or list of :class:`~.Particle`
            The particles or particle state to be resampled according to their weights
        nparts : int
            The number of particles to be returned from resampling

        Returns
        -------
        particle state: :class:`~.ParticleState`
            The particle state after resampling
        """

        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)
        if nparts is None:
            nparts = len(particles)

        log_weights = particles.log_weight
        weight_order = np.argsort(log_weights, kind='stable')
        max_log_value = log_weights[weight_order[-1]]
        with np.errstate(divide='ignore'):
            cdf = np.log(np.cumsum(np.exp(log_weights[weight_order] - max_log_value)))
        cdf += max_log_value

        # Pick random points for each of the particles
        u_j = np.random.rand(nparts)

        # Pick particles that represent the chosen point from the cdf
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]

        new_particles = particles[index]
        new_particles.log_weight = np.full((nparts, ), np.log(1/nparts))
        return new_particles


class StratifiedResampler(Resampler):
    """
    Traditional style resampler for particle filter. Splits the cdf into N evenly sized
    subpopulations ('strata'), then independently picks one value from each stratum. Complexity of
    order O(N).

    """

    def resample(self, particles, nparts=None):
        """Resample the particles

        Parameters
        ----------
        particles : :class:`~.ParticleState` or list of :class:`~.Particle`
            The particles or particle state to be resampled according to their weights
        nparts : int
            The number of particles to be returned from resampling

        Returns
        -------
        particle state: :class:`~.ParticleState`
            The particle state after resampling
        """

        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)
        if nparts is None:
            nparts = len(particles)

        log_weights = particles.log_weight
        weight_order = np.argsort(log_weights, kind='stable')
        max_log_value = log_weights[weight_order[-1]]
        with np.errstate(divide='ignore'):
            cdf = np.log(np.cumsum(np.exp(log_weights[weight_order] - max_log_value)))
        cdf += max_log_value

        # Strata lower bounds:
        s_l = np.arange(nparts) * (1 / nparts)

        # Independently pick a point in each strata
        u_j = np.random.uniform(s_l, s_l + 1 / nparts)

        # Pick particles that represent the chosen point from the cdf
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]

        new_particles = particles[index]
        new_particles.log_weight = np.full((nparts, ), np.log(1/nparts))
        return new_particles
