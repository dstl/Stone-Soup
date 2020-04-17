# -*- coding: utf-8 -*-
from abc import ABC
from typing import List, Optional

import numpy as np
from math import cos, sin
from scipy.linalg import expm
import weakref
from functools import lru_cache

from ..sensor.base import BaseSensor
from ..functions import rotz
from ..base import Property
from ..types.state import StateVector
from .base import Platform, MovingPlatform, FixedPlatform


class SensorPlatformMixin(Platform, ABC):
    """A platform mixin that can carry a number of different sensors and is
    capable of moving based upon the :class:`~.TransitionModel`.

    The location of platform mounted sensors will be maintained relative to
    the sensor position. Simple platforms move within a 2 or 3 dimensional
    rectangular cartesian space.

    A simple platform is considered to always be aligned with its principle
    velocity. It does not take into account issues such as bank angle or body
    deformation (e.g. flex).


    .. note:: This class abstract and not intended to be instantiated. To get the behaviour of
        this class use a subclass which combines this class with the `Platform` movement
        behaviours. Currently these are :class:`~.FixedSensorPlatform` and
        :class:`~.MovingSensorPlatform`

    """

    sensors = Property(List[BaseSensor], doc="A list of N mounted sensors", default=[])
    mounting_offsets = Property(List[StateVector], default=None,
                                doc="A list of StateVectors containing the sensor translation "
                                    "offsets from the platform's reference point. Defaults to "
                                    "a zero vector with the same length as the Platform's "
                                    ":attr:`position_mapping`")
    rotation_offsets = Property(List[StateVector], default=None,
                                doc="A list of StateVectors containing the sensor rotation "
                                    "offsets from the platform's primary axis (defined as the "
                                    "direction of motion). Defaults to a zero vector with the "
                                    "same length as the Platform's :attr:`position_mapping`")

    # TODO: Determine where a platform coordinate frame should be maintained

    def __init__(self, *args, **kwargs):
        """
        Ensure that the platform location and the sensor locations are
        consistent at initialisation.
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.mounting_offsets is None:
            self.mounting_offsets = [StateVector([0] * self.ndim)] * len(self.sensors)

        if self.rotation_offsets is None:
            self.rotation_offsets = [StateVector([0] * 3)] * len(self.sensors)

        if len(self.sensors) != len(self.mounting_offsets):
            raise ValueError(
                "Number of sensors associated with the platform does not "
                "match the number of sensor mounting offsets specified")

        if len(self.sensors) != len(self.rotation_offsets):
            raise ValueError(
                "Number of sensors associated with the platform does not "
                "match the number of sensor rotation offsets specified")

        # Store the platform weakref in each of the child sensors
        for sensor in self.sensors:
            sensor.platform_system = weakref.ref(self)

    def add_sensor(self, sensor: BaseSensor, mounting_offset: Optional[StateVector] = None,
                   rotation_offset: Optional[StateVector] = None) -> None:
        """ Add a sensor to the platform

        Parameters
        ----------
        sensor : :class:`~.BaseSensor`
            The sensor object to add
        mounting_offset : :class:`~.StateVector`, optional
            A StateVector with the mounting offset of the new sensor. If not supplied, defaults to
            a zero vector
        rotation_offset : :class:`~.StateVector`, optional
            A StateVector with the rotation offset of the new sensor. If not supplied, defaults to
            a zero vector.
        """
        self.sensors.append(sensor)
        sensor.platform_system = weakref.ref(self)

        if mounting_offset is None:
            mounting_offset = StateVector([0] * self.ndim)
        if rotation_offset is None:
            rotation_offset = StateVector([0] * 3)

        self.mounting_offsets.append(mounting_offset)
        self.rotation_offsets.append(rotation_offset)

    def get_sensor_position(self, sensor: BaseSensor) -> StateVector:
        """Return the position of the given sensor, which should be already attached to the
        platform. If the sensor is not attached to the platform, raises a :class:`ValueError`.

        Parameters
        ----------
        sensor : :class:`~.BaseSensor`
            The sensor for which to return the position.
        Returns
        -------
        : :class:`StateVector`
            The position of the sensor, taking into account the platform position and orientation
            and the mounting offset of the sensor.
        """
        i = self.sensors.index(sensor)
        if self.is_moving:
            offset = self._get_rotated_offset(i)
        else:
            offset = self.mounting_offsets[i]
        new_sensor_pos = self.position + offset
        return new_sensor_pos

    def get_sensor_orientation(self, sensor: BaseSensor) -> StateVector:
        """Return the orientation of the given sensor, which should be already attached to the
        platform. If the sensor is not attached to the platform, raises a :class:`ValueError`.

        Parameters
        ----------
        sensor : :class:`~.BaseSensor`
            The sensor for which to return the orientation.
        Returns
        -------
        : :class:`StateVector`
            The orientation of the sensor, taking into account the platform orientation
            and the rotation offset of the sensor.
        """
        # TODO handle roll?
        i = self.sensors.index(sensor)
        offset = self.rotation_offsets[i]
        return self.orientation + offset

    def _get_rotated_offset(self, i: int) -> np.ndarray:
        """ Determine the sensor mounting offset for the platforms relative
        orientation.

        Parameters
        ----------
        i : :class:`int`
            Integer reference to the sensor index

        Returns
        -------
        : :class:`np.ndarray`
            Sensor mounting offset rotated relative to platform motion
        """

        vel = self.velocity

        rot = _get_rotation_matrix(vel)
        return rot @ self.mounting_offsets[i]


class FixedSensorPlatform(SensorPlatformMixin, FixedPlatform):
    """ A moving sensor platform that simply combines the functionality of the
    :class:`~.FixedPlatform` with the :class:`~.SensorPlatformMixin`. This and
    :class:`~.MovingSensorPlatform` are the primary user facing classes for platforms.
        """
    pass


class MovingSensorPlatform(SensorPlatformMixin, MovingPlatform):
    """ A moving sensor platform that simply combines the functionality of the
    :class:`~.MovingPlatform` with the :class:`~.SensorPlatformMixin`. This and
    :class:`~.FixedSensorPlatform` are the primary user facing classes for platforms.
    """
    pass


def _get_rotation_matrix(vel: StateVector) -> np.ndarray:
    """ Generates a rotation matrix which can be used to determine the
    corrected sensor offsets.

    In the 2d case this returns the following rotation matrix
    [cos[theta] -sin[theta]]
    [cos[theta]  sin[theta]]

    In the 2d case this will be a 3x3 matrix which rotates around the Z axis
    followed by a rotation about the new Y axis.

    Parameters
    ----------
    vel : StateVector
        Dx1 vector denoting platform velocity in D dimensions

    Returns
    -------
    np.array
        DxD rotation matrix
    """
    if len(vel) == 3:
        return _rot3d(vel)
    elif len(vel) == 2:
        theta = _get_angle(vel, np.array([[1, 0]]))
        if vel[1] < 0:
            theta *= -1
        return np.array([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]])


def _get_angle(vec: StateVector, axis: np.ndarray) -> float:
    """ Returns the angle between a pair of vectors. Used to determine the
    angle of rotation required between relative rectangular cartesian
    coordinate frame of reference and platform inertial frame of reference.

    Parameters
    ----------
    vec : StateVector
        1xD array denoting platform velocity
    axis : np.ndarray
        Dx1 array denoting sensor offset relative to platform

    Returns
    -------
    Angle : float
        Angle, in radians, between the two vectors
    """
    vel_norm = vec / np.linalg.norm(vec)
    axis_norm = axis / np.linalg.norm(axis)

    return np.arccos(np.clip(np.dot(axis_norm, vel_norm), -1.0, 1.0))


def _rot3d(vec: np.ndarray) -> np.ndarray:
    """
    This approach determines the platforms attitude based upon its velocity
    component. It does not take into account potential platform roll, nor
    are the components calculated to account for physical artifacts such as
    platform trim (e.g. aircraft yaw whilst flying forwards).

    The process determines the yaw (x-y) and pitch (z to x-y plane) angles.
    The rotation matrix for a rotation by yaw around the Z-axis is then
    calculated, the rotated Y axis is then determined and used to calculate the
    rotation matrix which takes into account the platform pitch

    Parameters
    ----------
    vec: StateVector
        platform velocity

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    return _rot3d_tuple(tuple(vec.flat))


@lru_cache(maxsize=128)
def _rot3d_tuple(vec: tuple) -> np.ndarray:
    """ Private method. Should not be called directly, only from `_rot3d`

    Params and returns as :func:`~_rot3d`

    This wrapped method takes a tuple rather than a state vector. This allows caching, which
    is important as the new sensor approach means `_rot3d` is called on each call to get_position,
    and becomes a significant performance hit.

    """
    # TODO handle platform roll
    yaw = np.arctan2(vec[1], vec[0])
    pitch = np.arctan2(vec[2],
                       np.sqrt(vec[0] ** 2 + vec[1] ** 2)) * -1
    rot_z = rotz(yaw)
    # Modify to correct for new y axis
    y_axis = np.array([0, 1, 0])
    rot_y = expm(np.cross(np.eye(3), np.dot(rot_z, y_axis) * pitch))

    return np.dot(rot_y, rot_z)
