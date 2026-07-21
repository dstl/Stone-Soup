from abc import abstractmethod

import numpy as np
from scipy.stats import multivariate_normal as mvn
from typing import Sequence, Optional

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..types.matrix import TransitionMatrix
from ..types.mixture import GaussianMixture
from ..types.prediction import Prediction
from ..types.state import ExpandedModelAugmentedWeightedGaussianState
from ..types.update import Update


class Reducer(Base):
    """Base reducer for multiple model framework Gaussian mixture states.

    Reducers are responsible for computing likelihoods and reducing the number of
    hypotheses in a mixture while preserving the most probable model-conditioned
    states.
    """
    transition_probabilities: TransitionMatrix = Property(
        doc="Transition probability matrix used when combining model hypotheses.")
    transition_model_list: Optional[Sequence[TransitionModel]] = Property(
        default=None, doc="List of transition models available for reduction.")
    model_history_length: Optional[int] = Property(
        default=None, doc="Number of previous models to store in history.")
    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = (np.random.RandomState(self.seed)
                             if self.seed is not None
                             else np.random.mtrand._rand)

    def calculate_likelihood(self, states, timestamp):
        """Calculate state likelihoods for a Gaussian mixture.

        Parameters
        ----------
        states : :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            A :class:`~.GaussianMixture` object containing the current hypotheses.
        timestamp : datetime.datetime
            The timestamp for which likelihoods are being evaluated.

        Returns
        -------
        states : :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            Updated mixture with likelihood-weighted component weights.
        """

        if not isinstance(states, GaussianMixture):
            raise TypeError("States needs to be a GaussianMixture")

        if len(states.components) > 1:
            if states.timestamp == timestamp:
                if isinstance(states[0], (Prediction, Update)):
                    if not isinstance(states[0], ExpandedModelAugmentedWeightedGaussianState):
                        Likelihood_j = []
                        for state in states:
                            Likelihood_j.append(mvn.pdf(
                                state.hypothesis.measurement.state_vector.T,
                                state.hypothesis.measurement_prediction.mean.ravel(),
                                state.hypothesis.measurement_prediction.covar))
                        c_j = self.transition_probabilities[states[0]].T @ states.weights
                        weights = Likelihood_j * c_j.ravel()
                        weights = weights / np.sum(weights)
                        for i in range(len(states)):
                            states[i].weight = weights[i]
        else:
            if isinstance(states, list):
                states = GaussianMixture(states)
        return states

    @abstractmethod
    def reduce(self, states, timestamp, random_state=None, **kwargs):
        raise NotImplementedError
