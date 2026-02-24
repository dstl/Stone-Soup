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
    transition_probabilities=TransitionMatrix(np.atleast_2d(1), 0),
    transition_models=[ConstantVelocity(1e-5)],
    histories=0)


@pytest.mark.parametrize(
    "priors, priors_class",
    [(priors,
      ModelAugmentedWeightedGaussianState),
     (GaussianMixture([Prediction.from_state(prior, prior=None)]),
      ExpandedModelAugmentedWeightedGaussianStatePrediction),
     (GaussianMixture([Update.from_state(prior, hypothesis=None)]),
      ExpandedModelAugmentedWeightedGaussianStateUpdate)],
    ids=["state", "prediction", "update"]
)
def test_model_augmentor(priors, priors_class):
    augmented_states = model_augmentor.augment(priors)
    assert len(augmented_states) == len(priors)*len(model_augmentor.transition_models)
    assert isinstance(augmented_states[0], priors_class)


prior1 = ModelAugmentedWeightedGaussianState(
        state_vector=[1, 1],
        covar=np.diag([1, 1]),
        timestamp=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        weight=Probability(1),
        model_histories=[ModelAugmentedWeightedGaussianState(
            state_vector=[1, 1],
            covar=np.diag([1, 1]),
            timestamp=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            weight=Probability(1),
            model_histories=[],
            model_history_length=2)],
        model_history_length=2)


@pytest.mark.parametrize(
    "priors, model_augmentor, priors_class",
    [(GaussianMixture([Update.from_state(prior, hypothesis=None)]),
      ModelAugmentor(
        transition_probabilities=TransitionMatrix([[0.7, 0.3]], 1),
        transition_models=[ConstantVelocity(1e-5), ConstantVelocity(1e-1)],
        histories=0),
      ExpandedModelAugmentedWeightedGaussianStateUpdate),
     (GaussianMixture([Update.from_state(prior1, hypothesis=None),
                       Update.from_state(prior1, hypothesis=None)]),
      ModelAugmentor(
        transition_probabilities=TransitionMatrix([[0.7, 0.3],
                                                   [0.4, 0.6]], 2),
        transition_models=[ConstantVelocity(1e-5), ConstantVelocity(1e-1)],
        histories=0),
      ExpandedModelAugmentedWeightedGaussianStateUpdate)],
    ids=["2updates", "4updates"]
)
def test_model_augmentor2(priors, model_augmentor, priors_class, request):
    print(request.node.callspec.id)
    for state in priors:
        state.model_histories = [prior]
    print(model_augmentor.transition_probabilities.get_all_transition_matrices)
    augmented_states = model_augmentor.augment(priors)
    print(len(priors), len(model_augmentor.transition_models))
    print("augmented states: ", len(augmented_states))
    print("priors: ", len(priors))
    print(type(augmented_states[0]))
    print(type(priors_class))
    assert len(augmented_states) == len(priors)*len(model_augmentor.transition_models)
    assert isinstance(augmented_states[0], priors_class)
