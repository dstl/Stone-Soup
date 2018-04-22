# -*- coding: utf-8 -*-
from ..base import Base
from abc import abstractmethod, abstractproperty


class ControlModel(Base):
    """Control Model base class"""

    @abstractproperty
    def ndim_state(self):
        """ Number of state dimesions """
        pass

    @abstractproperty
    def ndim_ctrl(self):
        """ Number of control input dimesions """
        pass

    @abstractproperty
    def mapping(self):
        """ Mapping between control input and state dims """
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
