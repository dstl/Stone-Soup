from ..types.mixture import GaussianMixture
from ..types.prediction import Prediction
from ..types.state import ExpandedModelAugmentedWeightedGaussianState
from ..types.update import Update

from .base import Augmentor


class ModelAugmentor(Augmentor):
    """Model Augmentor"""

    def augment(self, states, *args, **kwargs):
        """Augments the prior states and the transition models (combinatorically)."""
        new_states = []
        for state in states:
            state_vector = state.state_vector
            covar = state.covar
            timestamp = state.timestamp
            weight = state.weight
            model_histories = state.model_histories
            measurement_histories = state.measurement_histories
            model_history_length = state.model_history_length
            measurement_history_length = state.measurement_history_length
            existence = state.existence

            for model in self.transition_models:
                # Should be from_state to inherit the Prediction or Update info
                temp_state = ExpandedModelAugmentedWeightedGaussianState(
                    state_vector=state_vector,
                    covar=covar,
                    timestamp=timestamp,
                    weight=weight,
                    model_histories=model_histories,
                    measurement_histories=measurement_histories,
                    model_history_length=model_history_length,
                    measurement_history_length=measurement_history_length,
                    existence=existence,
                    model=model
                    )
                if isinstance(state, Prediction):
                    temp_state = Prediction.from_state(temp_state, prior=state.prior)
                elif isinstance(state, Update):
                    temp_state = Update.from_state(temp_state, hypothesis=state.hypothesis)
                new_states.append(temp_state)
        return GaussianMixture(new_states)
