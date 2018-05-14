# -*- coding: utf-8 -*-
from abc import abstractproperty

import scipy as sp

from ..base import Model
from ...base import Property


class ControlModel(Model):
    """Control Model base class"""

    ndim_state = Property(int, doc="Number of state dimensions")
    mapping = Property(
        sp.ndarray, doc="Mapping between control and state dims")

    @abstractproperty
    def ndim_ctrl(self):
        """ Number of control input dimesions """
        pass
