# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.measurementmodel import MeasurementModel




class Initiator(Base):
    """Initiator base class"""

    @abstractmethod
    def initiate(self):
        """ Track Initiation method """
        pass
