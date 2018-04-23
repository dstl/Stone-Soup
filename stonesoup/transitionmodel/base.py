# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class TransitionModel(Base):
    """Transition Model base class"""

    @abstractmethod
    def transition(self, state):
        """Transitions *state* to a new state

        Parameters
        ----------
        state : State

        Returns
        -------
        State
            Transitioned state
        """
        raise NotImplemented
