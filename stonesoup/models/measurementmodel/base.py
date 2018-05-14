# -*- coding: utf-8 -*-
from abc import abstractproperty

import scipy as sp

from ..base import Model
from ...base import Property


class MeasurementModel(Model):
    """Measurement Model base class"""

    ndim_state = Property(int, doc="Number of state dimensions")
    mapping = Property(
        sp.ndarray, doc="Mapping between measurement and state dims")

    @abstractproperty
    def ndim_meas(self):
        """ Number of measurement dimensions"""
        pass
