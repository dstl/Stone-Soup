from abc import abstractmethod

import numpy as np
from scipy.stats import multivariate_normal as mvn
from typing import Sequence, Optional

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..types.matrix import TransitionMatrix
from ..types.mixture import GaussianMixture
from ..types.prediction import Prediction
from ..types.state import WeightedGaussianState, ExpandedModelAugmentedWeightedGaussianState
from ..types.update import Update


class Reducer(Base):
    transition_probabilities: TransitionMatrix = Property(
        doc="Transition Probability Matrix")
    transition_model_list: Optional[Sequence[TransitionModel]] = Property(
        default=None, doc="List of models to be used.")
    model_history_length: Optional[int] = Property(
        default=None, doc="number of previous models to store in history.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_likelihood(self, states, timestamp):
        if isinstance(states, list):
            try:
                (isinstance(states[0][0], WeightedGaussianState))
            except TypeError or IndexError:
                print(len(states))

            if isinstance(states[0], list):
                if isinstance(states[0][0], WeightedGaussianState):
                    states = GaussianMixture(states[0])
            else:
                states = GaussianMixture(states)
        if len(states.components) > 1:
            if states.timestamp == timestamp:
                if isinstance(states[0], (Prediction, Update)):
                    if isinstance(states[0], ExpandedModelAugmentedWeightedGaussianState):
                        pass
                    else:
                        Likelihood_j = []
                        for i in range(len(states)):
                            Likelihood_j.append(mvn.pdf(
                                states[i].hypothesis.measurement.state_vector.T,
                                states[i].hypothesis.measurement_prediction.mean.ravel(),
                                states[i].hypothesis.measurement_prediction.covar))
                        c_j = self.transition_probabilities[states[0]].T @ states.weights
                        weights = Likelihood_j * c_j.ravel()
                        weights = weights / np.sum(weights)
                        for i in range(len(states)):
                            states[i].weight = weights[i]
            else:
                pass

        else:
            if isinstance(states, list):
                states = GaussianMixture(states)
        return states

    @abstractmethod
    def reduce(self, states, timestamp, **kwargs):
        raise NotImplementedError
