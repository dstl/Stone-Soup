# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC

import numpy as np

from ..base import Model
from ...base import Property


class MeasurementModel(Model, ABC):
    """Measurement Model base class"""

    ndim_state = Property(int, doc="Number of state dimensions")
    mapping = Property(np.ndarray, doc="Mapping between measurement and state dims")

    @property
    def ndim(self):
        return self.ndim_meas

    @property
    @abstractmethod
    def ndim_meas(self):
        """Number of measurement dimensions"""
        pass
