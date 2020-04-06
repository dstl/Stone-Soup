# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC

from ..platform import Platform

from ..base import Base, Property


class SensorBase(Base, ABC):
    """Sensor base class

        A sensor object that operates according to a given
        :class:`~.MeasurementModel`.
    """
    platform_system = Property(Platform, default=None,
                               doc='`weakref` to the platform on which the '
                                   'sensor is mounted')

    # noinspection PyPropertyDefinition
    @platform_system.setter
    def set_platform_system(self, value):
        # this slightly od construction is to allow overriding by subclasses
        self._set_platform_system(value)

    def _set_platform_system(self, value):
        self._property_platform_system = value

    @abstractmethod
    def measure(self, **kwargs):
        raise NotImplementedError

    @property
    def position(self):
        """The sensor position on a 3D Cartesian plane, expressed as a 3x1 array of Cartesian
        coordinates in the order :math:`x,y,z` in the order :math:`x,y,z`.

        This property delegates that actual calculation of position to the platform on which the
        sensor is mounted."""
        return self.platform_system().get_sensor_position(self)

    @property
    def orientation(self):
        """A 3x1 array of angles (rad), specifying the sensor orientation in terms of the
        counter-clockwise rotation around each Cartesian axis in the order :math:`x,y,z`.
        The rotation angles are positive if the rotation is in the counter-clockwise
        direction when viewed by an observer looking along the respective rotation axis,
        towards the origin.

        This property delegates that actual calculation of orientation to the platform on which the
        sensor is mounted."""
        return self.platform_system().get_sensor_orientation(self)
