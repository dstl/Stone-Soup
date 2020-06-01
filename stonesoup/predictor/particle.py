# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np

from random import random
from .base import Predictor
from ..types.particle import Particle
from ..types.prediction import ParticleStatePrediction
from ..models.transition.linear import ConstantVelocity, ConstantAcceleration, ConstantTurn, CombinedLinearGaussianTransitionModel


class ParticlePredictor(Predictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.
    """

    @lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.ParticleState`
            A prior state object
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)

        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state
        """
        noise_coeff = [0.1]
        turn = [0.1]

        """When converting between models, a different number of state dimensions are needed ie ConstantVelocity 
        requires position and velocity, constant acceleration requires position, velocity and acceleration,
        this is causing a problem with the matrix multiplication"""

        transition_matrix = np.array([[0.9, 0.1],
                                      [0.1, 0.9]])

        sum_cv = np.cumsum(transition_matrix[0])
        sum_ca = np.cumsum(transition_matrix[1])

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        new_particles = []
        for particle in prior.particles:
            new_state_vector = self.transition_model.function(
                particle,
                noise=True,
                time_interval=time_interval,
                **kwargs)
            new_particles.append(
                Particle(new_state_vector,
                         weight=particle.weight,
                         parent=particle.parent))

        return ParticleStatePrediction(new_particles, timestamp=timestamp)
