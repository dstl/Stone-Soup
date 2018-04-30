# -*- coding: utf-8 -*-
from ..types.model import Model
from abc import abstractproperty


class ControlModel(Model):
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
