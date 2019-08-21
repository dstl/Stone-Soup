# -*- coding: utf-8 -*-
import copy
from functools import lru_cache

import numpy as np
from scipy.linalg import inv

from .base import Updater
from ..base import Property
from ..functions import cholesky_eps, sde_euler_maruyama_integration
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import MeasurementPrediction
from ..types.update import Update, ParticleStateUpdate


class ParticleUpdater(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler: Resampler = Property(doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """
        particles = hypothesis.prediction.particles

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        particles.weight *= measurement_model.pdf(
            hypothesis.measurement, particles, num_samples=len(particles),
            **kwargs)

        # Normalise the weights
        sum_w = np.array(Probability.sum(particles.weight))
        particles.weight = particles.weight / sum_w

        # Resample
        new_particles = self.resampler.resample(particles)

        return Update.from_state(
            hypothesis.prediction,
            particles=new_particles, hypothesis=hypothesis,
            timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = measurement_model.function(particle, **kwargs)
            new_particles.append(
                Particle(new_state_vector,
                         weight=particle.weight,
                         parent=particle.parent))

        return MeasurementPrediction.from_state(
            state_prediction, particles=new_particles, timestamp=state_prediction.timestamp)


class GromovFlowParticleUpdater(Updater):
    """Gromov Flow Particle Updater

    This is implementation of Gromov method for stochastic particle flow
    filters [1]_.

    Parameters
    ----------

    References
    ----------
    .. [1] Daum, Fred & Huang, Jim & Noushin, Arjang. "Generalized Gromov
       method for stochastic particle flow filters." 2017
    """

    def update(self, hypothesis, **kwargs):

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        num_steps = 20
        b = 2
        s0 = (b-1) / (b**num_steps - 1)
        steps = [s0*b**n for n in range(num_steps)]

        time_steps = np.zeros((len(steps) + 1, ))
        time_steps[1:] = np.cumsum(steps)

        P = hypothesis.prediction.covar
        R = measurement_model.covar()
        inv_R = inv(R)

        # Start by making our own copy of the particle before we move them...
        particles = [copy.copy(particle) for particle in hypothesis.prediction.particles]

        def function(state, lambda_):
            try:
                H = measurement_model.matrix()
            except AttributeError:
                H = measurement_model.jacobian(state)

            # Eq. (12) Ref [1]
            a = P - lambda_*P@H.T@inv(R + lambda_*H@P@H.T)@H@P
            b = a @ H.T @ inv_R

            measurement_particle_state_vector = measurement_model.function(state, **kwargs)
            f = -b @ (measurement_particle_state_vector - hypothesis.measurement.state_vector)

            Q = b @ H @ a
            B = cholesky_eps((Q+Q.T)/2)

            return f, B

        for particle in particles:
            particle.state_vector = sde_euler_maruyama_integration(function, time_steps, particle)

        return ParticleStateUpdate(
            particles,
            hypothesis,
            timestamp=hypothesis.measurement.timestamp)

    predict_measurement = ParticleUpdater.predict_measurement
