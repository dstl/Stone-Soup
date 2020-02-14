# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np
import sys
from scipy.stats import multivariate_normal

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle, RaoBlackwellisedParticle
from ..types.prediction import ParticleMeasurementPrediction
from ..types.update import ParticleStateUpdate
from ..predictor.multi_model import MultiModelPredictor
from ..predictor.multi_model import RaoBlackwellisedMultiModelPredictor


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


class MultiModelParticleUpdater(Updater, RaoBlackwellisedMultiModelPredictor):
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
                **kwargs) * self.transition_matrix[particle.parent.dynamic_model][particle.dynamic_model]

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


class RaoBlackwellisedParticleUpdater(Updater, MultiModelPredictor):
    """Simple Particle Updater

        Perform measurement update step in the standard Kalman Filter.
        """

    resampler = Property(Resampler,
                         doc='Resampler to prevent particle degeneracy')

    def update(self, hypothesis, iteration, **kwargs):
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
                **kwargs) * self.transition_matrix[particle.parent.dynamic_model][particle.dynamic_model]

            model_probabilities = self.calculate_model_probabilities(particle)
            particle.model_probabilities = model_probabilities

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
                                         model_probabilities=particle.model_probabilities,
                                         time_interval=particle.time_interval))

        return ParticleMeasurementPrediction(
            new_particles, timestamp=state_prediction.timestamp)

    def calculate_model_probabilities(self, particle):

        previous_probabilities = particle.model_probabilities

        denominator = []
        for i, model in enumerate(self.model_list):

            # if p(m_k|m_k-1) = 0 then p(m_k|x_1:k) = 0
            transition_probability = self.transition_matrix[particle.parent.dynamic_model][i]
            if transition_probability == 0:
                previous_probabilities[i] = 0
                denominator.append(0)
            else:
                # Getting required states to apply the model to that state vector
                parent_required_state_space = particle.parent.state_vector[np.array(self.position_mapping[i])]

                # The noiseless application of m_k onto x_k-1
                mean = model.function(parent_required_state_space, time_interval=particle.time_interval, noise=False)

                # Input the indices that were removed previously
                for j in range(len(particle.state_vector)):
                    if j not in self.position_mapping[i]:
                        mean = np.insert(mean, j, particle.state_vector[j])

                # Extracting x, y, z from the particle
                particle_position = self.measurement_model.matrix() @ particle.state_vector

                # p(x_k|m_k, x_k-1)
                prob_position_given_model_and_old_position = self.measurement_model.pdf(particle_position, mean)
                # p(m_k-1|x_1:k-1)
                prob_previous_iteration_with_old_model = previous_probabilities[i]

                product_of_probs = prob_position_given_model_and_old_position * \
                                   transition_probability * \
                                   prob_previous_iteration_with_old_model

                denominator.append(product_of_probs)

        if denominator.count(0) == 4:
            print(particle)
            print(denominator)
            sys.exit()

        for i in range(len(previous_probabilities)):

            previous_probabilities[i] = denominator[i] / sum(denominator)

        return previous_probabilities

