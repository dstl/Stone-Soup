from abc import abstractmethod

from typing import Sequence

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..types.matrix import TransitionMatrix


class Augmentor(Base):
    """Augmentor Base class.

    The Augmentor extends the components of a GaussianMixture with the most recent transition
    model and updates the transition model history information required by multiple model
    algorithms.
    """
    transition_probabilities: TransitionMatrix = Property(
        doc="Transition probability matrix used to weight augmented states based on"
            "model transitions.")
    transition_models: Sequence[TransitionModel] = Property(
        doc="List of all transition models for state-model combinatorics.")
    histories: int = Property(
        doc="Number of past model transitions to retain.")

    @abstractmethod
    def augment(self, states):
        """Augment a collection of states with model history information.

        Parameters
        ----------
        states : :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            Input states to augment with transition model history.

        Returns
        -------
        :class:`~.GaussianMixture` of :class:`~.ModelAugmentedWeightedGaussianState`
            Augmented state objects containing updated history fields.
        """
        raise NotImplementedError
