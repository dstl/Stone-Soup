from ..types.mixture import GaussianMixture
from ..types.prediction import Prediction, ExpandedModelAugmentedWeightedGaussianStatePrediction
from ..types.state import State, ExpandedModelAugmentedWeightedGaussianState
from ..types.update import Update, ExpandedModelAugmentedWeightedGaussianStateUpdate

from .base import Augmentor


class ModelAugmentor(Augmentor):
    """Augmentor that expands states by transition model hypotheses.

    The ModelAugmentor creates an augmented hypothesis set by combining each
    input state with every transition model in :attr:`transition_models`.
    Each augmented state carries the model-specific weight and history required
    by model-augmented prediction and update routines.
    """

    def augment(self, states, *args, **kwargs):
        """Augment a sequence of states with all available transition models.

        Parameters
        ----------
        states : :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            Input states to expand into model-conditioned hypotheses.

        Returns
        -------
        :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            A mixture containing an augmented state for each input state and
            each transition model.
        """
        new_states = []
        for i, state in enumerate(states):
            for j, model in enumerate(self.transition_models):
                if isinstance(state, Prediction):
                    target_type = ExpandedModelAugmentedWeightedGaussianStatePrediction
                elif isinstance(state, Update):
                    target_type = ExpandedModelAugmentedWeightedGaussianStateUpdate
                else:
                    target_type = ExpandedModelAugmentedWeightedGaussianState
                new_state = State.from_state(
                    state,
                    weight=self.transition_probabilities[state][i, j]*state.weight,
                    model=model,
                    target_type=target_type)
                new_states.append(new_state)
        return GaussianMixture(new_states)
