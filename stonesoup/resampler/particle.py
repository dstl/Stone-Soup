import numpy as np
from enum import Enum

from .base import Resampler
from ..base import Property
from ..types.state import ParticleState


class SystematicResampler(Resampler):
    """
    Traditional style resampler for particle filter. Calculates first random point in
    (0, 1/nparts], then calculates `nparts` points that are equidistantly distributed across the
    CDF. Complexity of order O(N) where N is the number of resampled particles.

    """

    def resample(self, particles, nparts=None):
        """
        Resample the particles

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
    """
    This wrapper uses a :class:`~.Resampler` to resample the particles inside
    an instance of :class:`~.Particles`, but only after checking if this is necessary
    by comparing the Effective Sample Size (ESS) with a supplied threshold (numeric).
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
    resampler: Resampler = Property(default=SystematicResampler(),
                                    doc='Resampler to wrap, which is called \
                                        when ESS below threshold')

    def resample(self, particles, nparts=None):
        """
        Parameters
        ----------
        particles : list of :class:`~.Particle`
            The particles to be resampled according to their weight
        nparts : int
            The number of particles to be returned from resampling

        Returns
        -------
        particles : list of :class:`~.Particle`
            The particles, either unchanged or resampled, depending on weight degeneracy
        """
        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)
        if nparts is None:
            nparts = len(particles)
        if self.threshold is None:
            self.threshold = len(particles) / 2
        # If ESS too small, resample
        if 1 / np.sum(np.exp(2*particles.log_weight)) < self.threshold:
            return self.resampler.resample(particles, nparts)
        else:
            return particles


class MultinomialResampler(Resampler):
    """
    Traditional style resampler for particle filter. Calculates a random point in (0, 1]
    individually for each particle, and picks the corresponding particle from the CDF calculated
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

        # Pick particles that represent the chosen point from the CDF
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]

        new_particles = particles[index]
        new_particles.log_weight = np.full((nparts, ), np.log(1/nparts))
        return new_particles


class StratifiedResampler(Resampler):
    """
    Traditional style resampler for particle filter. Splits the CDF into N evenly sized
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

        # Independently pick a point in each stratum
        u_j = np.random.uniform(s_l, s_l + (1 / nparts))

        # Pick particles that represent the chosen point from the CDF
        index = weight_order[np.searchsorted(cdf, np.log(u_j))]

        new_particles = particles[index]
        new_particles.log_weight = np.full((nparts, ), np.log(1/nparts))
        return new_particles


class ResidualMethod(Enum):
    MULTINOMIAL = 'multinomial'
    SYSTEMATIC = 'systematic'
    STRATIFIED = 'stratified'


class ResidualResampler(Resampler):
    """
    Wrapper around a traditional resampler.
    Any particle, p with weight W >= 1/N, will be resampled floor(W_p)
    times, providing N_stage_1 = sum(floor(W_p)) resampled particles.

    The residual weights of each particle are carried over and passed into another resampler, where
    the remaining N - N_stage_1 particles are resampled from.

    Should be a more computationally efficient method than resampling all particles from a CDF.
    Cannot be used to upsample or downsample.
    """
    residual_method: ResidualMethod = Property(default=ResidualMethod.MULTINOMIAL,
                                    doc="Method used to resample particles from the residuals.")

    def resample(self, particles, nparts=None):
        """Resample the particles.
        For ResidualResampler, `nparts` must equal len(`particles`)

        Parameters
        ----------
        particles : :class:`~.ParticleState` or list of :class:`~.Particle`
            The particles or particle state to be resampled according to their weights

        nparts : int
            The number of particles to be returned from resampling - must equal number of particles
            from previous step

        Returns
        -------
        particle state: :class:`~.ParticleState`
            The particle state after resampling
        """
        if not isinstance(particles, ParticleState):
            particles = ParticleState(None, particle_list=particles)

        if nparts is None:
            nparts = len(particles)
        elif nparts != len(particles):
            raise NotImplementedError("This resampler does not currently support up- or down-"
                                      "sampling")

        log_weights = particles.log_weight

        # Get the true weight of each particle
        weights = np.exp(log_weights)

        # **** Stage 1 ****

        # Calculate the floor
        floors = np.floor(weights * nparts)

        # Generate particle index from stage 1 resampling
        # There might be a better way to do this, seems a complicated way to do a simple task
        stage_1_index = np.array([index for sublist in
                                  [[index]*int(floor) for index, floor in enumerate(floors)
                                   if int(floor) != 0] for index in sublist])

        # **** Stage 2 ****

        # Calculate number of particles to be resampled from residuals
        n_stage_2_parts = nparts - int(sum(floors))

        # Check stage 2 is necessary (Necessary in all cases except where all weights = 1/N)

        if n_stage_2_parts > 0:
            # Calculate the residuals
            r_weights = weights - floors/nparts

            # Normalise residual weights
            normalised_r_weights = r_weights * 1/sum(r_weights)

            r_log_weights = np.log(normalised_r_weights)
            weight_order = np.argsort(r_log_weights, kind='stable')
            max_log_value = r_log_weights[weight_order[-1]]
            with np.errstate(divide='ignore'):
                cdf = np.log(np.cumsum(np.exp(r_log_weights[weight_order] - max_log_value)))
            cdf += max_log_value

            if self.residual_method == ResidualMethod.MULTINOMIAL:
                # Pick random points for each of the particles
                u_j = np.random.rand(n_stage_2_parts)

            elif self.residual_method == ResidualMethod.SYSTEMATIC:
                # Pick random starting point
                u_i = np.random.uniform(0, 1 / n_stage_2_parts)

                # Cycle through the cumulative distribution and copy the particle
                # that pushed the score over the current value
                u_j = u_i + (1 / n_stage_2_parts) * np.arange(n_stage_2_parts)

            elif self.residual_method == ResidualMethod.STRATIFIED:
                # Strata lower bounds:
                s_l = np.arange(n_stage_2_parts) * (1 / n_stage_2_parts)

                # Independently pick a point in each stratum
                u_j = np.random.uniform(s_l, s_l + (1 / n_stage_2_parts))
            else:
                raise ValueError("Invalid string variable given for stage 2 residual_method")

            # Pick particles that represent the chosen point from the CDF
            stage_2_index = weight_order[np.searchsorted(cdf, np.log(u_j))]
            # Combine the indexes from both stages
            index = np.concatenate([stage_1_index, stage_2_index])

        else:
            index = stage_1_index

        new_particles = particles[index]
        new_particles.log_weight = np.full((nparts, ), np.log(1/nparts))

        return new_particles
