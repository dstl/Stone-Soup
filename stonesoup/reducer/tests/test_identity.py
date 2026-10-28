import numpy as np
import datetime

from ...types.matrix import TransitionMatrix
from ...types.mixture import GaussianMixture
from ...types.state import ModelAugmentedWeightedGaussianState

from ..identity import IdentityReducer


def test_identity_reducer():
    timestamp = datetime.datetime.now()
    state = ModelAugmentedWeightedGaussianState(
        state_vector=[0, 1],
        covar=np.diag([1, 1]),
        weight=1,
        timestamp=timestamp)
    transition_probabilities = TransitionMatrix(
        transition_matrix=np.array([[1, 1], [1, 1]]))

    prior = GaussianMixture(components=[state, state])
    reducer = IdentityReducer(transition_probabilities=transition_probabilities)
    reduced_states = reducer.reduce(prior, timestamp)

    assert all(getattr(prior, name) == getattr(reduced_states, name)
               for name in type(prior).properties)
    assert transition_probabilities == reducer.transition_probabilities
