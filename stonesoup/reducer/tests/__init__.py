import pytest
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Sequence

from ...base import Property
from ...models.transition import TransitionModel
from ...types.matrix import TransitionMatrix
from ...types.mixture import GaussianMixture
from ...types.numeric import Probability
from ...types.state import ModelAugmentedWeightedGaussianState

from ..base import Reducer


prior = ModelAugmentedWeightedGaussianState(
    state_vector=[[0], [1], [0], [1]],
    covar=np.diag([1.5, 0.5, 1.5, 0.5]),
    timestamp=datetime.now().replace(second=0, microsecond=0),
    weight=Probability(1),
    model_histories=[],
    model_history_length=0)
states = GaussianMixture([prior])


def test_base():
    Reducer.__abstractmethods__ = set()

    @dataclass
    class Dummy(Reducer):
        transition_probabilities: TransitionMatrix = Property(default=None, doc="TPM")
        transition_models: Sequence[TransitionModel] = Property(default=None,
                                                                doc="List of transition models")
        histories: int = Property(default=None, doc="Depth of history to be stored")
    d = Dummy()
    with pytest.raises(NotImplementedError):
        d.reduce(states, states[0].timestamp)


@pytest.mark.parametrize(
    "states, full_output",
    [(GaussianMixture(components=[prior, prior]),
      0),
     ([prior, prior],
      0),
     ([[prior], [prior]],
      0)],
    ids=["GM", "List", "List(List)"]
)
def test_calculate_likelihood(states, full_output):
    Reducer.__abstractmethods__ = set()

    @dataclass
    class Dummy(Reducer):
        transition_probabilities: TransitionMatrix = Property(default=None, doc="TPM")
        transition_models: Sequence[TransitionModel] = Property(default=None,
                                                                doc="List of transition models")
        histories: int = Property(default=None, doc="Depth of history to be stored")
    reducer = Dummy()
    reduced_states = reducer.calculate_likelihood(states, prior.timestamp)
    print(reduced_states)
    assert full_output == 0
