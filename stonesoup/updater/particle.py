# -*- coding: utf-8 -*-
from functools import lru_cache

from .base import Updater
from ..base import Property
from ..resampler import Resampler
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import MeasurementPrediction
from ..types.update import Update


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
        particles = hypothesis.prediction.particles.__deepcopy__()

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model

        for i, particle in enumerate(particles):
            particles.weight[i] *= measurement_model.pdf(
                hypothesis.measurement, particle,
                **kwargs)

        # Normalise the weights
        sum_w = Probability.sum(
            i for i in particles.weight)
        for i, _ in enumerate(particles):
            particles.weight[i] /= sum_w

        # Resample
        new_particles = self.resampler.resample(
            particles)

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
