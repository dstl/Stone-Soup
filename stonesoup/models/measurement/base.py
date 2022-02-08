# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC
from typing import Sequence

from ..base import Model
from ...base import Property


class MeasurementModel(Model, ABC):
    """Measurement Model base class"""

    ndim_state: int = Property(doc="Number of state dimensions")
    mapping: Sequence[int] = Property(doc="Mapping between measurement and state dims")

    @property
    def ndim(self) -> int:
        return self.ndim_meas

    @property
    @abstractmethod
    def ndim_meas(self) -> int:
        """Number of measurement dimensions"""
        pass
