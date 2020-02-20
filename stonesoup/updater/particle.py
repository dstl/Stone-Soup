# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np
import sys
from collections import namedtuple

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle, RaoBlackwellisedParticle
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
                hypothesis.measurement.state_vector, particle.state_vector,
                **kwargs)

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        # Resample
        new_particles, n_eff = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
                                   timestamp=hypothesis.measurement.timestamp), n_eff

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

    def update(self, hypothesis, predictor, **kwargs):
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

            particle.weight *= self.weight_calculator()

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        # Resample
        new_particles, n_eff = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
                                   timestamp=hypothesis.measurement.timestamp), n_eff

    def weight_calculator(self):

        prob_y_given_x = measurement_model.pdf(
            hypothesis.measurement.state_vector, particle.state_vector,
            **kwargs)
        transition_probability = predictor.transition_matrix[particle.parent.dynamic_model][particle.dynamic_model]

        parent_required_state_space = particle.parent.state_vector[
            np.array(predictor.position_mapping[particle.parent.dynamic_model])]

        mean = predictor.model_list[particle.parent.dynamic_model].function(
            parent_required_state_space, time_interval=time_interval, noise=False)

        for j in range(len(particle.state_vector)):
            if j not in predictor.position_mapping[particle.parent.dynamic_model]:
                mean = np.insert(mean, j, 0)

        particle_position = self.measurement_model.matrix() @ particle.state_vector

        prob_position_given_model_and_old_position = Probability(self.measurement_model.pdf(particle_position,
                                                                                            mean))
        return prob_y_given_x * transition_probability / prob_position_given_model_and_old_position

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


class RaoBlackwellisedParticleUpdater(Updater):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, predictor, prior_timestamp, transition, **kwargs):
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
            particle.model_probabilities = self.calculate_model_probabilities(particle, predictor, time_interval)

        for particle in hypothesis.prediction.particles:

            predictor.transition_matrix = transition

            prob_y_given_x = measurement_model.pdf(
                hypothesis.measurement.state_vector, particle.state_vector,
                **kwargs)

            prob_position_given_previous_position = sum(
                self.calculate_model_probabilities(particle, predictor, time_interval))

            prob_proposal = self.calculate_model_probabilities(
                particle, predictor, time_interval)[particle.dynamic_model] / prob_position_given_previous_position

            particle.weight *= prob_y_given_x * prob_position_given_previous_position / prob_proposal
            # particle.weight *= prob_y_given_x

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        # Resample
        new_particles, n_eff = self.resampler.resample(
            hypothesis.prediction.particles)

        return ParticleStateUpdate(new_particles,
                                   hypothesis,
                                   timestamp=hypothesis.measurement.timestamp), n_eff

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

    def calculate_model_probabilities(self, particle, predictor, time_interval):

        previous_probabilities = particle.model_probabilities

        denominator = []
        for i, model in enumerate(predictor.model_list):

            # if p(m_k|m_k-1) = 0 then p(m_k|x_1:k) = 0
            transition_probability = predictor.transition_matrix[particle.parent.dynamic_model][i]
            # Getting required states to apply the model to that state vector
            parent_required_state_space = particle.parent.state_vector[np.array(predictor.position_mapping[i])]

            # The noiseless application of m_k onto x_k-1

            mean = model.function(parent_required_state_space, time_interval=time_interval, noise=False)

            # Input the indices that were removed previously
            for j in range(len(particle.state_vector)):
                if j not in predictor.position_mapping[i]:
                    mean = np.insert(mean, j, particle.state_vector[j])

            # Extracting x, y, z from the particle
            particle_position = self.measurement_model.matrix() @ particle.state_vector

            prob_position_given_model_and_old_position = Probability(self.measurement_model.pdf(particle_position,
                                                                                                mean))
            # p(m_k-1|x_1:k-1)
            prob_previous_iteration_with_old_model = Probability(previous_probabilities[i])

            product_of_probs = Probability(prob_position_given_model_and_old_position *
                                           transition_probability *
                                           prob_previous_iteration_with_old_model)
            denominator.append(product_of_probs)

        new_probabilities = []
        for i in range(len(previous_probabilities)):
            new_probabilities.append(denominator[i] / sum(denominator))

        return new_probabilities
