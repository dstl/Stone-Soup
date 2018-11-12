# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import TransitionModel


class OrbitalModel(TransitionModel):

    """Transition Model base class"""

    @property
    @abstractmethod
    def ndim_state(self):
        """Number of state dimensions"""
        pass
