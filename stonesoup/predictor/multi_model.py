# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np

from random import random
from .base import Predictor
from ..types.particle import Particle
from ..types.prediction import ParticleStatePrediction


class MultiModelPredictor(Predictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.
    """

    def __init__(self, transition_matrix, model_list, position_mapping, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.position_mapping = position_mapping
        self.transition_matrix = transition_matrix
        self.model_list = model_list

        self.probabilities = []
        if type(self.transition_matrix) == float:
            self.probabilities.append(self.transition_matrix)
        else:
            for rows in self.transition_matrix:
                self.probabilities.append(np.cumsum(rows))

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

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        new_particles = []
        for particle in prior.particles:
            for model_index in range(len(self.transition_matrix)):
                if particle.dynamic_model == model_index:

                    self.transition_model = self.model_list[particle.dynamic_model]
                    # Based on given position mapping create a new state vector that contains only the required states
                    required_state_space = particle.state_vector[np.array(self.position_mapping[particle.dynamic_model])]

                    new_state_vector = self.transition_model.function(
                        required_state_space,
                        time_interval=time_interval,
                        **kwargs)

                    # Setting the new dynamic model based on the given accepted models
                    self.transition_model = self.model_list[model_index]
                    # Change the value of the dynamic value randomly according to the defined transition matrix
                    dynamic_model = np.searchsorted(self.probabilities[model_index], random())

                    # Calculate the indices removed from the state vector to become compatible with the dynamic model
                    missed_indices = []
                    for i, position in enumerate(particle.state_vector):
                        if position not in required_state_space:
                            missed_indices.append(i)

                    if len(missed_indices) != 0:
                        for i in missed_indices:
                            new_state_vector = np.insert(new_state_vector, i, particle.state_vector[i])
                    new_state_vector = np.reshape(new_state_vector, (-1, 1))

                    new_particles.append(
                        Particle(new_state_vector,
                                 weight=particle.weight,
                                 parent=particle,
                                 dynamic_model=dynamic_model))

        return ParticleStatePrediction(new_particles, timestamp=timestamp)
