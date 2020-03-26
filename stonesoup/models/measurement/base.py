# -*- coding: utf-8 -*-
from abc import abstractmethod

import scipy as sp

from ..base import Model
from ...base import Property


class MeasurementModel(Model):
    """Measurement Model base class"""

    ndim_state = Property(int, doc="Number of state dimensions")
    mapping = Property(
        sp.ndarray, doc="Mapping between measurement and state dims")

    @property
    def ndim(self):
        return self.ndim_meas

    @property
    @abstractmethod
    def ndim_meas(self):
        """Number of measurement dimensions"""
        pass
