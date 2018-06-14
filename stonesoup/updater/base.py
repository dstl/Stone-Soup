# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.measurementmodel import MeasurementModel


class Updater(Base):
    """Updater base class"""

    measurement_model = Property(MeasurementModel, doc="measurement model")

    @abstractmethod
    def get_measurement_prediction(self, state_prediction, **kwargs):
        raise NotImplemented

    @abstractmethod
    def update(self, prediction, measurement,
               measurement_prediction=None, **kwargs):
        raise NotImplemented
