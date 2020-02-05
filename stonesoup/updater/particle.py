# -*- coding: utf-8 -*-
from functools import lru_cache

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import ParticleMeasurementPrediction
from ..types.update import ParticleStateUpdate
from ..predictor.multi_model import MultiModelPredictor


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


class MultiModelParticleUpdater(Updater, MultiModelPredictor):
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
                **kwargs) * self.transition_matrix[particle.dynamic_model[0]][particle.dynamic_model[1]]

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
