import numpy as np
from itertools import product

from .base import Reducer

from ..functions import gm_reduce_single
from ..types.array import StateVectors
from ..types.mixture import GaussianMixture
from ..types.state import ModelAugmentedWeightedGaussianState


class ModelReducer(Reducer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction_histories = [list(x) for x in product(self.transition_model_list,
                                                             repeat=self.model_history_length)]
        self.reduced_output = []

    def reduce(self, states, timestamp):
        temp = []
        if len(self.transition_probabilities[states[0]].ravel()) == len(states.weights):
            m_ij_weights = (
                self.transition_probabilities[states[0]].ravel() *
                (states.weights/np.sum(states.weights)))
        elif len(self.transition_probabilities[states[0]].ravel())**2 == len(states.weights):
            m_ij_weights = (
                self.transition_probabilities.transition_matrices[1].ravel() *
                (states.weights/np.sum(states.weights)))*np.random.rand(len(states.weights))
        else:
            m_ij_weights = (
                self.transition_probabilities[states[0]] @
                (states.weights/np.sum(states.weights)))
        for history in self.reduction_histories:
            hist_state = []
            for state in states:
                if state.model_histories[-self.model_history_length:] == history:
                    hist_state.append(state)
            if len(hist_state) == 0:
                hist_state = states
            else:
                hist_state = GaussianMixture(hist_state)
            means = StateVectors([state.state_vector for state in hist_state])
            covars = np.stack([state.covar for state in hist_state], axis=2)
            weights = np.asarray(hist_state.weights * np.random.rand(len(hist_state.weights)))

            mean, covar = gm_reduce_single(means, covars, weights)
            temp.append(ModelAugmentedWeightedGaussianState.from_state(
                hist_state[0],
                state_vector=mean,
                covar=covar,
                weight=np.sum(hist_state.weights)/np.sum(m_ij_weights)))

        states = self.calculate_likelihood(temp, timestamp)
        return states
