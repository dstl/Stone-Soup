import pytest
import numpy as np

from datetime import datetime

from ..matrix import TransitionMatrix
from ..numeric import Probability
from ..state import ModelAugmentedWeightedGaussianState

from ...models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                         ConstantVelocity, KnownTurnRate)

cv = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5), ConstantVelocity(0.5)])
ct = CombinedLinearGaussianTransitionModel([KnownTurnRate(np.array([0.5, 0.5]), 0.1)])

transitioning_probabilities_0 = TransitionMatrix(np.atleast_2d(1))
transitioning_probabilities_1 = TransitionMatrix([0.75, 0.25])
transitioning_probabilities_2 = TransitionMatrix([[0.95, 0.05], [0.25, 0.75]])

prior = ModelAugmentedWeightedGaussianState(
    state_vector=[[0], [1], [0], [1]],
    covar=np.diag([1.5, 0.5, 1.5, 0.5]),
    timestamp=datetime.now().replace(second=0, microsecond=0),
    weight=Probability(1),
    model_histories=[],
    model_history_length=0)


@pytest.mark.parametrize(
    "transitioning_probabilities, full_output",
    [(TransitionMatrix(np.atleast_2d(1)),
      {0: np.array([[1]])}),
     (TransitionMatrix([0.75, 0.25]),
      {0: np.array([[0.75, 0.25]])}),
     (TransitionMatrix([[0.95, 0.05], [0.25, 0.75]]),
      {1: np.array([[0.95, 0.05], [0.25, 0.75]]),
       0: np.array([[0.6, 0.4]])})],
    ids=["KF", "GPB1", "GPB2"]
)
def test_transition_matrix(transitioning_probabilities, full_output):
    print(transitioning_probabilities.get_all_transition_matrices)
    for (k1, v1), (k2, v2) in zip(
            transitioning_probabilities.get_all_transition_matrices.items(), full_output.items()):
        assert k1 == k2
        assert np.allclose(v1, v2)


@pytest.mark.parametrize(
    "model_length, num_components, output",
    [(0,
      0,
      None,
      ),
     (0,
      2,
      np.array([[0.75, 0.25]])),
     (1,
      2,
      np.array([[0.6, 0.4]])),
     (2,
      2,
      np.array([[0.95, 0.05], [0.25, 0.75]]),
      )],
    ids=["KF", "GPB1", "GPB2:1/2", "GPB2:2/2"]
)
def test_transition_state(model_length, num_components, output, request):
    print(request.node.callspec.id)
    tpm = TransitionMatrix([[0.95, 0.05], [0.25, 0.75]], num_components)
    prior = ModelAugmentedWeightedGaussianState(
        state_vector=[[0], [1], [0], [1]],
        covar=np.diag([1.5, 0.5, 1.5, 0.5]),
        timestamp=datetime.now().replace(second=0, microsecond=0),
        weight=Probability(1),
        model_histories=[cv]*(model_length-1),
        model_history_length=model_length)
    print(tpm[prior], output)
    if model_length == 0:
        with pytest.raises(TypeError):
            assert tpm(prior)
    else:
        assert np.allclose(tpm[prior], output)
