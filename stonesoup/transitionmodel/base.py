# -*- coding: utf-8 -*-
from abc import abstractproperty
from ..types.model import Model


class TransitionModel(Model):
    """Transition Model base class

    Paramaters
    ----------
    ndim_state: int
        The number of state dimensions

        - Constant for each model
    """

    @abstractproperty
    def ndim_state(self):
        """ Number of state dimensions"""
        pass
