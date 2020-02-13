# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np
from scipy.stats import multivariate_normal

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
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

        # Normalise the weights
        sum_w = Probability.sum(
            i.weight for i in hypothesis.prediction.particles)
        for particle in hypothesis.prediction.particles:
            particle.weight /= sum_w

        if iteration > 3:
            for particle in hypothesis.prediction.particles:
                self.calculate_model_probabilities(particle)

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

    def calculate_model_probabilities(self, particle):

        previous_probabilities = particle.parent.model_probabilities

        for i, model in enumerate(self.model_list):

            transition_probability = self.transition_matrix[particle.parent.dynamic_model][i]

            parent_required_state_space = particle.parent.state_vector[np.array(self.position_mapping[i])]
            particle_required_state_space = particle.state_vector[np.array(self.position_mapping[i])]

            mean = model.function(parent_required_state_space, time_interval=particle.time_interval, noise=False)
            mean = np.squeeze(mean)

            covar = model.covar(particle.time_interval)
            
            print(mean)
            print(particle_required_state_space)

            print(self.measurement_model.pdf(particle_required_state_space, mean))

            print(model)

            # print(multivariate_normal.pdf(particle_required_state_space, mean=mean, cov=covar))
            # print("Test")
