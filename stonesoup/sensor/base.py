# -*- coding: utf-8 -*-
from abc import ABC
from typing import Optional

from stonesoup.movable import Movable, FixedMovable
from stonesoup.types.state import State

from ..base import Base, Property
from ..types.array import StateVector


class PlatformMountable(Base, ABC):
    """Base class for any object that can be mounted on a platform.

    All PlatformMountables must be mounted on a platform to calculate their position and
    orientation. To make this easier, if the sensor has a position and/or orientation specified in
    the constructor, and no :attr:`platform_system`, then the default is to create an internally
    held "private" platform for the Sensor. This allows the Sensor to control (and set) its own
    position and orientation.

    """

    rotation_offset: StateVector = Property(
        default=None,
        doc="A StateVector containing the sensor rotation "
            "offsets from the platform's primary axis (defined as the "
            "direction of motion). Defaults to a zero vector with the "
            "same length as the Platform's :attr:`velocity_mapping`")
    mounting_offset: StateVector = Property(
        default=None,
        doc="A StateVector containing the sensor translation "
            "offsets from the platform's reference point. Defaults to "
            "a zero vector with length 3")

    movement_controller: Movable = Property(
        default=None,
        doc="The :class:`~.`Movable` object that controls the movement of this sensor. Will be "
            "set by the platform if the sensor is assigned to a platform."
    )

    def __init__(self, *args, **kwargs):
        position = kwargs.pop('position', None)
        orientation = kwargs.pop('orientation', None)
        self._internal_movement_controller = None
        super().__init__(*args, **kwargs)
        if position is not None or orientation is not None:
            if position is None:
                # assuming 3d for a default platform
                position = StateVector([0, 0, 0])
            if orientation is None:
                orientation = StateVector([0, 0, 0])
            self._internal_movement_controller = FixedMovable(
                states=State(state_vector=position),
                position_mapping=list(range(len(position))),
                orientation=orientation)
            self._property_movement_controller = self._internal_movement_controller
            self._set_mounting_rotation_defaults()

    def _set_mounting_rotation_defaults(self):
        if self.movement_controller is None:
            return
        if self.mounting_offset is None:
            self.mounting_offset = StateVector([0]
                                               * len(self.movement_controller.position_mapping))

        if self.rotation_offset is None:
            self.rotation_offset = StateVector([0] * 3)

    @movement_controller.setter
    def movement_controller(self, value):
        if self._has_internal_controller:
            raise AttributeError('The movement controller cannot be set on a Sensor with a private'
                                 'internal controller')
        self._property_movement_controller = value
        self._set_mounting_rotation_defaults()

    @property
    def position(self) -> Optional[StateVector]:
        """The sensor position on a 3D Cartesian plane, expressed as a 3x1 :class:`StateVector`
        of Cartesian coordinates in the order :math:`x,y,z`.

        .. note::
            This property delegates the actual calculation of position to the Sensor's
            :attr:`movement_controller`

            It is settable if, and only if, the sensor holds its own internal movement_controller.
            """
        if self.movement_controller is None:
            return None
        return (self.movement_controller.position
                + self.movement_controller._get_rotated_offset(self.mounting_offset))

    @position.setter
    def position(self, value: StateVector):
        if self._has_internal_controller:
            self.movement_controller.position = value
        else:
            raise AttributeError('Cannot set sensor position unless the sensor has its own '
                                 'default movement_controller')

    @property
    def orientation(self) -> Optional[StateVector]:
        """A 3x1 StateVector of angles (rad), specifying the sensor orientation in terms of the
        counter-clockwise rotation around each Cartesian axis in the order :math:`x,y,z`.
        The rotation angles are positive if the rotation is in the counter-clockwise
        direction when viewed by an observer looking along the respective rotation axis,
        towards the origin.

        .. note::
            This property delegates the actual calculation of orientation to the Sensor's
            :attr:`movement_controller`

            It is settable if, and only if, the sensor holds its own internal movement_controller.
            """
        if self.movement_controller is None:
            return None
        return self.movement_controller.orientation + self.rotation_offset

    @orientation.setter
    def orientation(self, value: StateVector):
        if self._has_internal_controller:
            self.movement_controller.orientation = value
        else:
            raise AttributeError('Cannot set sensor position unless the sensor has its own '
                                 'default movement_controller')

    @property
    def _has_internal_controller(self):
        return self._internal_movement_controller is not None

    @property
    def velocity(self) -> Optional[StateVector]:
        """The sensor velocity on a 3D Cartesian plane, expressed as a 3x1 :class:`StateVector`
        of Cartesian coordinates in the order :math:`x,y,z`.

        .. note::
            This property delegates the actual calculation of velocity to the Sensor's
            :attr:`movement_controller`

            It is settable if, and only if, the sensor holds its own internal movement_controller
            which is a :class:`~.MovingMovable`."""
        return self.movement_controller.velocity
