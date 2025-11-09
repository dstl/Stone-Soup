import numpy as np
import datetime

from ...augmentor import ModelAugmentor
from ...models.transition.linear import ConstantVelocity
from ...types.matrix import TransitionMatrix
from ...types.mixture import GaussianMixture
from ...types.state import ModelAugmentedWeightedGaussianState

from ..model import ModelReducer


def test_model_reducer():
    cv1 = ConstantVelocity(1)
    cv2 = ConstantVelocity(2)
    transitions = [cv1, cv2]
    timestamp = datetime.datetime.now()
    state = ModelAugmentedWeightedGaussianState(
        state_vector=[0, 1],
        covar=np.diag([1, 1]),
        weight=1,
        timestamp=timestamp)
    transition_probabilities = TransitionMatrix(
        transition_matrix=np.array([[1, 1], [1, 1]]))

    augmentor = ModelAugmentor(transition_probabilities=transition_probabilities,
                               transition_models=transitions,
                               histories=1)
    prior = GaussianMixture(components=[state, state])
    augmented_states = augmentor.augment(prior)
    reducer = ModelReducer(transition_probabilities=transition_probabilities,
                           transition_model_list=transitions,
                           model_history_length=1)
    reduced_states = reducer.reduce(augmented_states, timestamp)
    # print(reduced_states)
    print(prior, reduced_states)
    # assert all(getattr(prior, name) == getattr(reduced_states, name)
    #            for name in type(prior).properties)
    # assert transition_probabilities == reducer.transition_probabilities
    assert len(prior) == len(reduced_states)
