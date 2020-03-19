# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np
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
        always_resample: :Boolean:
            if True, then the particle filter will resample every time step.
            Otherwise, will only resample when 25% or less of the particles
            are deemed effective.
            Calculated by 1 / sum(particle.weight^2) for all particles

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
                hypothesis.measurement.state_vector, particle.state_vector,
                **kwargs)

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        # Only resamples if less than a quarter of the particles are effective.
        # Resample
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
                Particle(new_state_vector,
                         weight=particle.weight,
                         parent=particle.parent,
                         dynamic_model=particle.dynamicmodel))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)


class MultiModelParticleUpdater(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, predictor=None, always_resample=True, **kwargs):
        """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.
        predictor: :class:`~.Predictor`
            Predictor which holds the transition matrix, dynamic models and the
            mapping rules.
        always_resample: :Boolean:
            if True, then the particle filter will resample every time step.
            Otherwise, will only resample when 25% or less of the particles
            are deemed effective.
            Calculated by 1 / sum(particle.weight^2) for all particles

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
                hypothesis.measurement.state_vector, particle.state_vector,
                **kwargs) * predictor.transition_matrix[particle.parent.dynamic_model][particle.dynamicmodel]

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        n_eff = 1 / sum([p.weight * p.weight for p in hypothesis.prediction.particles])

        # Only resamples if less than a quarter of the particles are effective.
        if n_eff < len(hypothesis.prediction.particles) / 2 or always_resample:
            # Resample
            new_particles = self.resampler.resample(
                hypothesis.prediction.particles)
        else:
            new_particles = [particle for particle in hypothesis.prediction.particles]

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
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, predictor=None, prior_timestamp=None,
               transition=None, always_resample=True, **kwargs):
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
        always_resample: :Boolean:
            if True, then the particle filter will resample every time step.
            Otherwise, will only resample when 25% or less of the particles
            are deemed effective.
            Calculated by 1 / sum(particle.weight^2) for all particles
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

            prob_y_given_x = measurement_model.pdf(
                hypothesis.measurement.state_vector, particle.state_vector,
                **kwargs)

            particle.weight *= prob_y_given_x

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        n_eff = 1 / sum([p.weight * p.weight for p in hypothesis.prediction.particles])

        # Only resamples if less than a quarter of the particles are effective.
        if n_eff < len(hypothesis.prediction.particles) / 2 or always_resample:
            # Resample
            new_particles = self.resampler.resample(
                hypothesis.prediction.particles)
        else:
            new_particles = [particle for particle in hypothesis.prediction.particles]

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
                                         dynamic_model=particle.dynamicmodel,
                                         model_probabilities=particle.model_probabilities))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)

    def calculate_model_probabilities(self, particle, position_mapping,
                                      transition_matrix, model_list, time_interval):

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
                    time_interval=time_interval, noise=True)

                # Input the indices that were removed previously
                for j in range(len(particle.state_vector)):
                    if j not in position_mapping[i]:
                        mean = np.insert(mean, j, 0)
                mean = mean.reshape((1, -1))[0]

                prob_position_given_model_and_old_position = multivariate_normal.pdf(
                    np.transpose(particle.state_vector),
                    mean=mean,
                    cov=np.diag([0.75 for i in mean])
                )
                # p(m_k-1|x_1:k-1)
                prob_previous_iteration_given_model = previous_probabilities[l]

                product_of_probs = (prob_position_given_model_and_old_position *
                                               transition_probability *
                                               prob_previous_iteration_given_model)
                selected_model_sum = selected_model_sum + product_of_probs
            denominator_components.append(selected_model_sum)

        # Calculate the denominator
        denominator = sum(denominator_components)

        # Calculate the new probabilities
        new_probabilities = []
        for i in range(len(previous_probabilities)):
            new_probabilities.append(denominator_components[i] / denominator)

        return [new_probabilities, denominator]
