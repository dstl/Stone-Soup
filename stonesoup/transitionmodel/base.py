# -*- coding: utf-8 -*-
from ..base import Base
from abc import abstractmethod, abstractproperty


class TransitionModel(Base):
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

    @abstractmethod
    def eval(self):
        """ Model transition function """
        pass

    @abstractmethod
    def random(self):
        """ Model noise/sample generation function """
        pass

    @abstractmethod
    def pdf(self):
        """ Model pdf/likelihood evaluation function """
        pass
