# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle, RaoBlackwellisedParticle, MultiModelParticle
from ..types.prediction import ParticleMeasurementPrediction
from ..types.update import ParticleStateUpdate


class ParticleUpdater(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

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

        for particle in hypothesis.prediction.particles:
            particle.weight *= measurement_model.pdf(
                hypothesis.measurement, particle,
                **kwargs)

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        new_particles = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
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
                         parent=particle.parent,
                         dynamic_model=particle.dynamicmodel))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)


class MultiModelParticleUpdater(Updater):
    """Particle Updater for the Multi Model system

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, predictor=None, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.
        predictor: :class:`~.MultiModelParticlePredictor`
            Predictor which holds the transition matrix, dynamic models and the
            mapping rules.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """
        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        for particle in hypothesis.prediction.particles:

            hypothesis.measurement.state_vector = np.reshape(
                hypothesis.measurement.state_vector, (-1, 1)
            )

            particle.weight *= measurement_model.pdf(
                hypothesis.measurement, particle,
                **kwargs) * predictor.transition_matrix[particle.parent.dynamic_model][particle.dynamic_model]

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        new_particles = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
                                   timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = measurement_model.function(
                particle.state_vector, noise=0, **kwargs)
            new_particles.append(
                MultiModelParticle(new_state_vector,
                                   weight=particle.weight,
                                   parent=particle.parent,
                                   dynamic_model=particle.dynamicmodel))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)


class RaoBlackwellisedParticleUpdater(Updater):
    """Particle Updater for the Raoblackwellised scheme

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, predictor=None, prior_timestamp=None,
               transition=None, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.
        predictor: :class:`~.Predictor`
            Predictor which holds the transition matrix, dynamic models and the
            mapping rules.
        prior_timestamp: :class: `~.datetime.datetime`
            the timestamp associated with the prior measurement state.
        transition: :np.array:
            a non block diagonal transition_matrix example:
                [[0.97 0.01 0.01 0.01]
                 [0.01 0.97 0.01 0.01]
                 [0.01 0.01 0.97 0.01]
                 [0.01 0.01 0.01 0.97]]
            which would represent using four models.
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

            new_model_probabilities, prob_position_given_previous_position = self.calculate_model_probabilities(
                particle, predictor.position_mapping, transition, predictor.transition_model, time_interval)

            particle.model_probabilities = new_model_probabilities

            hypothesis.measurement.state_vector = np.reshape(
                hypothesis.measurement.state_vector, (-1, 1)
            )

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

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
                                   timestamp=hypothesis.measurement.timestamp)

    @lru_cache()
    def predict_measurement(self, state_prediction, measurement_model=None,
                            **kwargs):

        if measurement_model is None:
            measurement_model = self.measurement_model

        new_particles = []
        for particle in state_prediction.particles:
            new_state_vector = measurement_model.function(
                particle.state_vector, noise=0, **kwargs)
            new_particles.append(
                RaoBlackwellisedParticle(new_state_vector,
                                         weight=particle.weight,
                                         parent=particle.parent,
                                         model_probabilities=particle.model_probabilities))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)

    def calculate_model_probabilities(self, particle, position_mapping,
                                      transition_matrix, model_list, time_interval):

        """Calculates the new model probabilities based
            on the ones calculated in the previous time step"""

        previous_probabilities = particle.model_probabilities

        denominator_components = []
        # Loop over the current models m_k
        for l in range(len(previous_probabilities)):
            selected_model_sum = 0
            # Loop over the previous models m_k-1
            for i, model in enumerate(model_list):
                # Looks up p(m_k|m_k-1)
                # Note: if p(m_k|m_k-1) = 0 then p(m_k|x_1:k) = 0
                transition_probability = transition_matrix[i][l]
                if transition_probability == 0:
                    break
                # Getting required states to apply the model to that state vector
                parent_required_state_space = particle.parent.state_vector[
                    np.array(position_mapping[i])]

                # The noiseless application of m_k onto x_k-1
                mean = model.function(
                    parent_required_state_space,
                    time_interval=time_interval, noise=False)

                # Input the indices that were removed previously
                for j in range(len(particle.state_vector)):
                    if j not in position_mapping[i]:
                        mean = np.insert(mean, j, 0)
                mean = mean.reshape((1, -1))[0]

                cov_matrices = [self.measurement_model.noise_covar for i in range(len(model.model_list))]

                prob_position_given_model_and_old_position = multivariate_normal.pdf(
                    np.transpose(particle.state_vector),
                    mean=mean,
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
