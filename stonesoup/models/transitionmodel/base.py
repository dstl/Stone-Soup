# -*- coding: utf-8 -*-
from abc import abstractproperty

from ..base import Model


class TransitionModel(Model):
    """Transition Model base class"""

    @abstractproperty
    def ndim_state(self):
        """ Number of state dimensions"""
        pass
