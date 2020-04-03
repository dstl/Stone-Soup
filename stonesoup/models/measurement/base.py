# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC

import numpy as np

from stonesoup.functions import coerce_to_valid_mapping
from ..base import Model
from ...base import Property


class MeasurementModel(Model, ABC):
    """Measurement Model base class"""

    ndim_state = Property(int, doc="Number of state dimensions")
    mapping = Property(np.ndarray, doc="Mapping between measurement and state dims")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = coerce_to_valid_mapping(self.mapping)

    @property
    def ndim(self):
        return self.ndim_meas

    @property
    @abstractmethod
    def ndim_meas(self):
        """Number of measurement dimensions"""
        pass
