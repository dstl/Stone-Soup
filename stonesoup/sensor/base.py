# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..types.array import StateVector
from ..models import MeasurementModel


class Sensor(Base):
    """Sensor base class

    A sensor object that opperates according to a given
    :class:`~.MeasurementModel`.
    """

    measurement_model = Property(
        MeasurementModel, default=None, doc="Measurement model")

    @abstractmethod
    def gen_measurement(**kwargs):
        """Generate a measurement"""
        raise NotImplementedError


class MountableSensor(Sensor):
    """MountableSensor base class

    A sensor that can be mounted on a platform.
    """

    platform_offset = Property(
        StateVector, default=None,
        doc="A state vector describing the mounting offset of the sensor,\
            relative to the platform on which it is mounted")
