from abc import abstractmethod

from typing import Sequence

from ..base import Base, Property
from ..models.transition import TransitionModel
from ..types.matrix import TransitionMatrix


class Augmentor(Base):
    transition_probabilities: TransitionMatrix = Property(doc="TPM")
    transition_models: Sequence[TransitionModel] = Property(doc="List of transition models")
    histories: int = Property(doc="Depth of history to be stored")

    @abstractmethod
    def augment(self, states):
        """Augment with the models"""
        raise NotImplementedError
