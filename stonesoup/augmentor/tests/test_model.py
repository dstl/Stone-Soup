import pytest

import numpy as np
from datetime import datetime

from ...augmentor import ModelAugmentor
from ...models.transition.linear import ConstantVelocity
from ...types.matrix import TransitionMatrix
from ...types.mixture import GaussianMixture
from ...types.numeric import Probability
from ...types.prediction import ExpandedModelAugmentedWeightedGaussianStatePrediction, Prediction
from ...types.state import ModelAugmentedWeightedGaussianState
from ...types.update import ExpandedModelAugmentedWeightedGaussianStateUpdate, Update


prior = ModelAugmentedWeightedGaussianState(
    state_vector=[1, 1],
    covar=np.diag([1, 1]),
    timestamp=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
    weight=Probability(1),
    model_histories=[],
    model_history_length=0)
priors = GaussianMixture([prior])

model_augmentor = ModelAugmentor(
    transition_probabilities=TransitionMatrix(np.atleast_2d(1)),
    transition_models=[ConstantVelocity(1e-5)],
    histories=0)


@pytest.mark.parametrize(
    "priors, priors_class",
    [(priors,
      ModelAugmentedWeightedGaussianState),
     (GaussianMixture([Prediction.from_state(prior, prior=None)]),
      ExpandedModelAugmentedWeightedGaussianStatePrediction),
     (GaussianMixture([Update.from_state(prior, hypothesis=None)]),
      ExpandedModelAugmentedWeightedGaussianStateUpdate),
     (GaussianMixture([Update.from_state(prior, hypothesis=None),
                       Update.from_state(prior, hypothesis=None)]),
      ExpandedModelAugmentedWeightedGaussianStateUpdate)],
    ids=["state", "prediction", "update", "2updates"]
)
def test_model_augmentor(priors, priors_class):
    augmented_states = model_augmentor.augment(priors)
    assert len(augmented_states) == len(priors)
    assert isinstance(augmented_states[0], priors_class)
