from ..types.mixture import GaussianMixture
from ..types.prediction import Prediction, ExpandedModelAugmentedWeightedGaussianStatePrediction
from ..types.state import State, ExpandedModelAugmentedWeightedGaussianState
from ..types.update import Update, ExpandedModelAugmentedWeightedGaussianStateUpdate

from .base import Augmentor


class ModelAugmentor(Augmentor):
    """Model Augmentor"""

    def augment(self, states, *args, **kwargs):
        """Augments the prior states and the transition models (combinatorically)."""
        new_states = []
        for state in states:
            for model in self.transition_models:
                if isinstance(state, Prediction):
                    target_type = ExpandedModelAugmentedWeightedGaussianStatePrediction
                elif isinstance(state, Update):
                    target_type = ExpandedModelAugmentedWeightedGaussianStateUpdate
                else:
                    target_type = ExpandedModelAugmentedWeightedGaussianState
                new_state = State.from_state(
                    state,
                    model=model,
                    target_type=target_type)
                new_states.append(new_state)
        return GaussianMixture(new_states)
