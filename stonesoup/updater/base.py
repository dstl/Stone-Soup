# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.measurementmodel import MeasurementModel


class Updater(Base):
    """Updater base class"""

    measurement_model = Property(MeasurementModel, doc="measurement model")

    @abstractmethod
    def update(self, prediction, measurement, **kwargs):
        raise NotImplemented
