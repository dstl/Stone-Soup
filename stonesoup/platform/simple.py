# -*- coding: utf-8 -*-
from abc import ABC
from typing import List

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
    """A simple Platform that can carry a number of different sensors and is
    capable of moving based upon the :class:`~.TransitionModel`.

    The location of platform mounted sensors will be maintained relative to
    the sensor position. Simple platforms move within a 2 or 3 dimensional
    rectangular cartesian space.

    A simple platform is considered to always be aligned with its principle
    velocity. It does not take into account issues such as bank angle or body
    deformation (e.g. flex).

    """

    sensors = Property([BaseSensor], doc="A list of N mounted sensors", default=[])
    mounting_offsets = Property(List[StateVector], default=None,
                                doc="A list of StateVectors containing the sensor translation "
                                    "offsets from the platform's reference point. Defaults to "
                                    "a zero vector with the same length as the Platform's mapping")
    rotation_offsets = Property(List[StateVector], default=None,
                                doc="A list of StateVectors containing the sensor translation "
                                    "offsets from the platform's primary axis (defined as the "
                                    "direction of motion). Defaults to a zero vector with the "
                                    "same length as the Platform's mapping")

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

    def add_sensor(self, sensor: BaseSensor, mounting_offset: StateVector = None,
                   rotation_offset: StateVector = None):
        """ TODO
                Parameters
                ----------
                sensor : :class:`stonesoup.sensor.sensor.Sensor`
                    The sensor object to add
                mounting_offset : :class:`StateVector`
                    A 1xN array with the mounting offset of the new sensor
                """
        self.sensors.append(sensor)
        sensor.platform_system = weakref.ref(self)

        if mounting_offset is None:
            mounting_offset = StateVector([0] * self.ndim)
        if rotation_offset is None:
            rotation_offset = StateVector([0] * 3)

        self.mounting_offsets.append(mounting_offset)
        self.rotation_offsets.append(rotation_offset)

    def get_sensor_position(self, sensor: BaseSensor):
        # TODO docs
        i = self.sensors.index(sensor)
        if self.is_moving:
            offset = self._get_rotated_offset(i)
        else:
            offset = self.mounting_offsets[i]
        new_sensor_pos = self.position + offset
        return new_sensor_pos

    def get_sensor_orientation(self, sensor: BaseSensor):
        # TODO docs
        # TODO handle roll?
        i = self.sensors.index(sensor)
        offset = self.rotation_offsets[i]
        return self.orientation + offset

    def _get_rotated_offset(self, i):
        """ Determine the sensor mounting offset for the platforms relative
        orientation.

        Parameters
        ----------
        i : int
            Integer reference to the sensor index

        Returns
        -------
        np.ndarray
            Sensor mounting offset rotated relative to platform motion
        """

        vel = self.velocity

        rot = _get_rotation_matrix(vel)
        return rot @ self.mounting_offsets[i]


class FixedSensorPlatform(SensorPlatformMixin, FixedPlatform):
    pass


class MovingSensorPlatform(SensorPlatformMixin, MovingPlatform):
    pass


def _get_rotation_matrix(vel):
    """ Generates a rotation matrix which can be used to determine the
    corrected sensor offsets.

    In the 2d case this returns the following rotation matrix
    [cos[theta] -sin[theta]]
    [cos[theta]  sin[theta]]

    In the 2d case this will be a 3x3 matrix which rotates around the Z axis
    followed by a rotation about the new Y axis.

    Parameters
    ----------
    vel : np.ndarrary
        1xD vector denoting platform velocity in D dimensions

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


def _get_angle(vec, axis):
    """ Returns the angle between a pair of vectors. Used to determine the
    angle of rotation required between relative rectangular cartesian
    coordinate frame of reference and platform inertial frame of reference.

    Parameters
    ----------
    vec : np.ndarray
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


def _rot3d(vec):
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
    vec: np.ndarray
        platform velocity

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    return _rot3d_tuple(tuple(vec.flat))


@lru_cache(maxsize=128)
def _rot3d_tuple(vec):
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
