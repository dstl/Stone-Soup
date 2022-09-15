import copy
from functools import lru_cache

import numpy as np
from scipy.linalg import inv, block_diag
from scipy.stats import multivariate_normal

from .base import Updater
from .kalman import KalmanUpdater, ExtendedKalmanUpdater
from ..base import Property
from ..functions import cholesky_eps, sde_euler_maruyama_integration
from ..predictor.particle import MultiModelPredictor, RaoBlackwellisedMultiModelPredictor
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import RaoBlackwellisedParticle, MultiModelParticle
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
        predicted_state = copy.copy(hypothesis.prediction)

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        predicted_state.weight = predicted_state.weight * measurement_model.pdf(
            hypothesis.measurement, predicted_state, **kwargs)

        # Normalise the weights
        sum_w = np.array(Probability.sum(predicted_state.weight))
        predicted_state.weight = predicted_state.weight / sum_w

        # Resample
        if self.resampler is not None:
            predicted_state = self.resampler.resample(predicted_state)

        return Update.from_state(
            state=hypothesis.prediction,
            state_vector=predicted_state.state_vector,
            weight=predicted_state.weight,
            hypothesis=hypothesis,
            timestamp=hypothesis.measurement.timestamp,
            particle_list=None
            )

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_state_vector = measurement_model.function(state_prediction, **kwargs)
        new_weight = state_prediction.weight

        return MeasurementPrediction.from_state(
            state_prediction, state_vector=new_state_vector, timestamp=state_prediction.timestamp,
            particle_list=None, weight=new_weight)


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
            None,
            hypothesis,
            particle_list=particles,
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
            particle_pred.mean, particle_pred.covar, particle_pred.timestamp)
        # Needed for cross covar
        kalman_hypothesis.measurement_prediction = None
        kalman_update = self.kalman_updater.update(kalman_hypothesis, **kwargs)

        return ParticleStateUpdate(
            particle_update.state_vector,
            hypothesis,
            weight=particle_update.weight,
            fixed_covar=kalman_update.covar,
            particle_list=None,
            timestamp=particle_update.timestamp)

    def predict_measurement(self, state_prediction, *args, **kwargs):
        particle_prediction = super().predict_measurement(
            state_prediction, *args, **kwargs)

        kalman_prediction = self.kalman_updater.predict_measurement(
            Prediction.from_state(
                state_prediction, state_prediction.state_vector, state_prediction.covar,
                target_type=GaussianStatePrediction),
            *args, **kwargs)

        return ParticleMeasurementPrediction(
            state_vector=particle_prediction.state_vector,
            weight=state_prediction.weight,
            fixed_covar=kalman_prediction.covar,
            particle_list=None,
            timestamp=particle_prediction.timestamp)


class MultiModelParticleUpdater(Updater):
    """Particle Updater for the Multi Model system"""

    resampler: Resampler = Property(doc='Resampler to prevent particle degeneracy')
    predictor: MultiModelPredictor = Property()

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
        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        new_particles = [copy.copy(particle) for particle in hypothesis.prediction.particles]

        for particle in new_particles:
            particle.weight = particle.weight \
                * measurement_model.pdf(hypothesis.measurement, particle, **kwargs) \
                * self.predictor.transition_matrix[particle.parent.dynamic_model][particle.dynamic_model]  # noqa: E501

        # Normalise the weights
        sum_w = Probability.sum(i.weight for i in hypothesis.prediction.particles)
        for particle in new_particles:
            particle.weight /= sum_w

        new_particles = self.resampler.resample(new_particles)

        return ParticleStateUpdate(None,
                                   particle_list=new_particles,
                                   hypothesis=hypothesis,
                                   timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = measurement_model.function(
                particle.state_vector, noise=False, **kwargs)
            new_particles.append(
                MultiModelParticle(new_state_vector,
                                   weight=particle.weight,
                                   parent=particle.parent,
                                   dynamic_model=particle.dynamicmodel))

        return ParticleMeasurementPrediction(
            None, particle_list=new_particles, timestamp=state_prediction.timestamp)


class RaoBlackwellisedParticleUpdater(Updater):
    """Particle Updater for the Raoblackwellised scheme"""

    resampler: Resampler = Property(doc='Resampler to prevent particle degeneracy')
    predictor: RaoBlackwellisedMultiModelPredictor = Property(
        doc="Predictor which hold holds transition matrix, models and mappings")

    def update(self, hypothesis, prior_timestamp, **kwargs):  # TODO: Handle prior timestamp
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

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model
        time_interval = hypothesis.prediction.timestamp - prior_timestamp

        for particle in hypothesis.prediction.particles:

            new_model_probabilities, prob_position_given_previous_position = \
                self.calculate_model_probabilities(
                    particle, measurement_model, self.predictor.position_mapping,
                    self.transition_matrix, self.predictor.transition_models, time_interval)

            particle.model_probabilities = new_model_probabilities

            prob_y_given_x = measurement_model.pdf(
                hypothesis.measurement, particle,
                **kwargs)

            particle.weight *= prob_y_given_x

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        new_particles = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(None,
                                   particle_list=new_particles,
                                   hypothesis=hypothesis,
                                   timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = measurement_model.function(
                particle.state_vector, noise=False, **kwargs)
            new_particles.append(
                RaoBlackwellisedParticle(new_state_vector,
                                         weight=particle.weight,
                                         parent=particle.parent,
                                         model_probabilities=particle.model_probabilities))

        return ParticleMeasurementPrediction(
            None, particle_list=new_particles, timestamp=state_prediction.timestamp)

    @staticmethod
    def calculate_model_probabilities(particle, measurement_model, position_mapping,
                                      transition_matrix, transition_models, time_interval):
        """Calculates the new model probabilities based
            on the ones calculated in the previous time step"""

        previous_probabilities = particle.model_probabilities

        denominator_components = []
        # Loop over the current models m_k
        for l in range(len(previous_probabilities)):  # noqa: E741
            selected_model_sum = 0
            # Loop over the previous models m_k-1
            for i, model in enumerate(transition_models):
                # Looks up p(m_k|m_k-1)
                # Note: if p(m_k|m_k-1) = 0 then p(m_k|x_1:k) = 0
                transition_probability = transition_matrix[i][l]
                if transition_probability == 0:
                    break
                # Getting required states to apply the model to that state vector
                parent_required_state_space = copy.copy(particle.parent)
                parent_required_state_space.state_vector = \
                    particle.parent.state_vector[position_mapping[i], :]

                # The noiseless application of m_k onto x_k-1
                mean = model.function(
                    parent_required_state_space, time_interval=time_interval, noise=False)

                # Input the indices that were removed previously
                for j in range(len(particle.state_vector)):
                    if j not in position_mapping[i]:
                        mean = np.insert(mean, j, 0, axis=0)

                cov_matrices = [measurement_model.covar()
                                for _ in range(len(model.model_list))]

                prob_position_given_model_and_old_position = multivariate_normal.pdf(
                    particle.state_vector.ravel(),
                    mean=mean.ravel(),
                    cov=block_diag(*cov_matrices)
                )
                # p(m_k-1|x_1:k-1)
                prob_previous_iteration_given_model = previous_probabilities[l]

                product_of_probs = Probability(prob_position_given_model_and_old_position *
                                               transition_probability *
                                               prob_previous_iteration_given_model)
                selected_model_sum = selected_model_sum + product_of_probs
            denominator_components.append(selected_model_sum)

        # Calculate the denominator
        denominator = sum(denominator_components)

        # Calculate the new probabilities
        new_probabilities = []
        for i in range(len(previous_probabilities)):
            new_probabilities.append(
                Probability(denominator_components[i] / denominator))

        return [new_probabilities, denominator]
