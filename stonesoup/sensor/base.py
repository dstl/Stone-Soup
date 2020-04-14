# -*- coding: utf-8 -*-
import weakref
from abc import abstractmethod, ABC

from ..types.array import StateVector
from ..platform import Platform

from ..base import Base, Property


class BaseSensor(Base, ABC):
    """Sensor base class

    .. warning::
        This class is private and should not be used or subclassed directly. Instead use the
        :class:`~.Sensor` class which is needed to achieve the functionality described in this
        class's documentation.

    """
    platform_system = Property(weakref.ref, default=None,
                               doc='``weakref`` to the platform on which the sensor is mounted')

    @property
    def platform(self) -> Platform:
        """Return the platform system to which the sensor is attached. Resolves the ``weakref``
        stored in the :attr:`platform_system` Property."""
        return self.platform_system()

    # noinspection PyPropertyDefinition
    @platform_system.setter
    def set_platform_system(self, value):
        # this slightly odd construction is to allow overriding by the Sensor subclass
        self._set_platform_system(value)

    def _set_platform_system(self, value):
        self._property_platform_system = value

    @abstractmethod
    def measure(self, **kwargs):
        raise NotImplementedError

    @property
    def position(self) -> StateVector:
        """The sensor position on a 3D Cartesian plane, expressed as a 3x1 :class:`StateVector`
        of Cartesian coordinates in the order :math:`x,y,z`.

        .. note::
            This property delegates the actual calculation of position to the platform on which the
            sensor is mounted.

            It is settable if, and only if, the sensor holds its own internal platform."""
        return self.platform_system().get_sensor_position(self)

    @position.setter
    def position(self, value):
        if self._has_internal_platform:
            self.platform.position = value
        else:
            raise AttributeError('Cannot set sensor position unless the sensor has its own '
                                 'default platform')

    @property
    def orientation(self):
        """A 3x1 StateVector of angles (rad), specifying the sensor orientation in terms of the
        counter-clockwise rotation around each Cartesian axis in the order :math:`x,y,z`.
        The rotation angles are positive if the rotation is in the counter-clockwise
        direction when viewed by an observer looking along the respective rotation axis,
        towards the origin.

        .. note::
            This property delegates the actual calculation of orientation to the platform on which
            the sensor is mounted.

            It is settable if, and only if, the sensor holds its own internal platform."""
        return self.platform_system().get_sensor_orientation(self)

    @orientation.setter
    def orientation(self, value):
        if self._has_internal_platform:
            self.platform.position = value
        else:
            raise AttributeError('Cannot set sensor position unless the sensor has its own '
                                 'default platform')

    @property
    def _has_internal_platform(self):
        return False
