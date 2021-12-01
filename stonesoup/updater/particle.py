# -*- coding: utf-8 -*-
import copy
from functools import lru_cache

import numpy as np
from scipy.linalg import inv

from .base import Updater
from .kalman import KalmanUpdater, ExtendedKalmanUpdater
from ..base import Property
from ..functions import cholesky_eps, sde_euler_maruyama_integration
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import (
    Prediction, ParticleMeasurementPrediction, GaussianStatePrediction, MeasurementPrediction)
from ..types.update import ParticleStateUpdate, Update


class ParticleUpdater(Updater):
    """Particle Updater

    Perform an update by multiplying particle weights by PDF of measurement
    model (either :attr:`~.Detection.measurement_model` or
    :attr:`measurement_model`), and normalising the weights. If provided, a
    :attr:`resampler` will be used to take a new sample of particles (this is
    called every time, and it's up to the resampler to decide if resampling is
    required).
    """

    resampler: Resampler = Property(default=None, doc='Resampler to prevent particle degeneracy')

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
        particles = copy.copy(hypothesis.prediction.particles)

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        particles.weight = particles.weight * measurement_model.pdf(
            hypothesis.measurement, particles, num_samples=len(particles),
            **kwargs)

        # Normalise the weights
        sum_w = np.array(Probability.sum(particles.weight))
        particles.weight = particles.weight / sum_w

        # Resample
        if self.resampler is not None:
            particles = self.resampler.resample(particles)

        return Update.from_state(
            hypothesis.prediction,
            particles=particles, hypothesis=hypothesis,
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
    filters [#]_. The Euler Maruyama method is used for integration, over 20
    steps using an exponentially increase step size.

    Parameters
    ----------

    References
    ----------
    .. [#] Daum, Fred & Huang, Jim & Noushin, Arjang. "Generalized Gromov
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


class GromovFlowKalmanParticleUpdater(GromovFlowParticleUpdater):
    """Gromov Flow Parallel Kalman Particle Updater

    This is a wrapper around the :class:`~.GromovFlowParticleUpdater` which
    can use a :class:`~.ExtendedKalmanUpdater` or
    :class:`~.UnscentedKalmanUpdater` in parallel in order to maintain a state
    covariance, as proposed in [#]_. In this implementation, the mean of the
    :class:`~.ParticleState` is used the EKF/UKF update.

    This should be used in conjunction with the
    :class:`~.ParticleFlowKalmanPredictor`.

    Parameters
    ----------

    References
    ----------
    .. [#] Ding, Tao & Coates, Mark J., "Implementation of the Daum-Huang
       Exact-Flow Particle Filter" 2012
    """
    kalman_updater: KalmanUpdater = Property(
        default=None,
        doc="Kalman updater to use. Default `None` where a new instance of"
            ":class:`~.ExtendedKalmanUpdater` will be created utilising the"
            "same measurement model.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.kalman_updater is None:
            self.kalman_updater = ExtendedKalmanUpdater(self.measurement_model)

    def update(self, hypothesis, **kwargs):
        particle_update = super().update(hypothesis, **kwargs)

        particle_pred = hypothesis.prediction

        kalman_hypothesis = copy.copy(hypothesis)
        # Convert to GaussianState
        kalman_hypothesis.prediction = GaussianStatePrediction(
            particle_pred.state_vector, particle_pred.covar, particle_pred.timestamp)
        # Needed for cross covar
        kalman_hypothesis.measurement_prediction = None
        kalman_update = self.kalman_updater.update(kalman_hypothesis, **kwargs)

        return ParticleStateUpdate(
            particle_update.particles,
            hypothesis,
            kalman_update.covar,
            timestamp=particle_update.timestamp)

    def predict_measurement(
            self, state_prediction, *args, **kwargs):
        particle_prediction = super().predict_measurement(
            state_prediction, *args, **kwargs)

        kalman_prediction = self.kalman_updater.predict_measurement(
            Prediction.from_state(
                state_prediction, state_prediction.state_vector, state_prediction.covar,
                target_type=GaussianStatePrediction),
            *args, **kwargs)

        return ParticleMeasurementPrediction(
            particle_prediction.particles,
            kalman_prediction.covar,
            timestamp=particle_prediction.timestamp)
