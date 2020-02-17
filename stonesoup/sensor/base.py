# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..types.state import StateVector


class Sensor(Base):
    """Sensor base class

    A sensor object that operates according to a given
    :class:`~.MeasurementModel`.
    """

    @abstractmethod
    def measure(self, **kwargs):
        """Generate a measurement"""
        raise NotImplementedError


class Sensor3DCartesian(Sensor):
    """Sensor base class extended to include 3D cartesian motion


    """
    position = Property(StateVector,
                        doc="The sensor position on a 3D Cartesian plane,\
                            expressed as a 3x1 array of Cartesian \
                            coordinates in the order :math:`x,y,z`")
    orientation = Property(
        StateVector,
        doc="A 3x1 array of angles (rad), specifying the sensor \
            orientation in terms of the counter-clockwise rotation \
            around each Cartesian axis in the order :math:`x,y,z`. The \
            rotation angles are positive if the rotation is in the \
            counter-clockwise direction when viewed by an observer \
            looking along the respective rotation axis, towards the \
            origin")

    @abstractmethod
    def measure(self, **kwargs):
        """Generate a measurement"""
        raise NotImplementedError
