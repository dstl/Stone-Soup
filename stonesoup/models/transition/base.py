# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Model


class TransitionModel(Model):
    """Transition Model base class"""

    @property
    @abstractmethod
    def ndim_state(self):
        """Number of state dimensions"""
        pass
