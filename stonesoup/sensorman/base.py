# -*- coding: utf-8 -*-
from abc import abstractmethod
from ..base import Base
import scipy as sp

from stonesoup.models.base import Model
from stonesoup.base import Property


class SensorManModel(Base):
    """Sensor management Model base class"""

    #ndim_state = Property(int, doc="Number of state dimensions")
    #mapping = Property(
    #    sp.ndarray, doc="Mapping between measurement and state dims")

    @property
    def ndim(self):
        raise NotImplementedError


    #@property
    #@abstractmethod
    #def ndim_meas(self):
    #    """Number of measurement dimensions"""
    #    raise NotImplementedError

