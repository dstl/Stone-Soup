# -*- coding: utf-8 -*-
import numpy as np
from math import cos, sin
from scipy.linalg import expm
import weakref
from functools import lru_cache

from stonesoup.sensor.base import Sensor3DCartesian
from ..base import Property
from ..types.state import StateVector
from ..functions import cart2pol
from .base import Platform


class PlatformSensor(Sensor3DCartesian):
    platform_system = Property(Platform, default=None,
                               doc='`weakref` to the platform on which the '
                                   'sensor is mounted')

    def measure(self, **kwargs):
        raise NotImplementedError

    def get_position(self):
        if (hasattr(self, 'platform_system')
                and self.platform_system is not None):
            return self.platform_system().get_sensor_position(self)
        else:
            return self.position

    def get_orientation(self):
        if (hasattr(self, 'platform_system')
                and self.platform_system is not None):
            return self.platform_system().get_sensor_orientation(self)
        else:
            return self.orientation


class SensorPlatform(Platform):
    """A simple Platform that can carry a number of different sensors and is
    capable of moving based upon the :class:`~.TransitionModel`.

    The location of platform mounted sensors will be maintained relative to
    the sensor position. Simple platforms move within a 2 or 3 dimensional
    rectangular cartesian space.

    A simple platform is considered to always be aligned with its principle
    velocity. It does not take into account issues such as bank angle or body
    deformation (e.g. flex).

    """

    sensors = Property([PlatformSensor], doc="A list of N mounted sensors")
    mounting_offsets = Property(
        [np.array], doc="A list of sensor offsets (For now expressed\
                            as a Nxn array of nD Cartesian coordinates)")
    mounting_mappings = Property(
        [np.array], doc="Mappings between the platform state vector and the\
                            individuals sensors mounting offset (For now\
                            expressed as a Nxn array of nD Cartesian\
                            coordinates or a 1xn array where a single\
                            mapping will be applied to all sensors)")

    # TODO: Determine where a platform coordinate frame should be maintained

    def __init__(self, *args, **kwargs):
        """
        Ensure that the platform location and the sensor locations are
        consistent at initialisation.
        """
        super().__init__(*args, **kwargs)
        if self.mounting_mappings.max() > len(self.state.state_vector):
            raise IndexError(
                "Platform state vector length and sensor mounting mapping "
                "are incompatible")

        if len(self.sensors) != self.mounting_offsets.shape[0]:
            raise IndexError(
                "Number of sensors associated with the platform does not "
                "match the number of sensor mounting offsets specified")

        if ((len(self.sensors) != self.mounting_mappings.shape[0]) and
                (self.mounting_mappings.shape[0] != 1)):
            raise IndexError(
                "Number of sensors associated with the platform does not "
                "match the number of mounting mappings specified")

        if ((self.mounting_mappings.shape[0] == 1) and
                len(self.sensors) > 1):
            mapping_array = np.empty((0, self.mounting_mappings.shape[1]), int)
            for i in range(len(self.sensors)):
                mapping_array = np.append(mapping_array,
                                          self.mounting_mappings,
                                          axis=0)
            self.mounting_mappings = mapping_array
        for sensor in self.sensors:
            sensor.platform_system = weakref.ref(self)
            sensor.position = None
            sensor.orientation = None

    def add_sensor(self, sensor, mounting_offset, mounting_mapping=None):
        """ Determine the sensor mounting offset for the platforms relative
                orientation.

                Parameters
                ----------
                sensor : :class:`Sensor`
                    The sensor object to add
                mounting_offset : :class:`np.ndarray`
                    A 1xN array with the mounting offset of the new sensor
                mounting_mapping : :class:`np.ndarray`, optional
                    A 1xN array with the mounting mapping of the new sensor.
                    If `None` (default) then use the same mapping as all
                    previous sensors. If all sensor do not a have the same
                    mapping then raise ValueError
                """
        self.sensors.append(sensor)
        sensor.platform_system = weakref.ref(self)
        sensor.position = None
        sensor.orientation = None

        if mounting_mapping is None:
            if not np.all(self.mounting_mappings
                          == self.mounting_mappings[0, :]):
                raise ValueError('Mapping must be specified unless all '
                                 'sensors have the same mapping')
            mounting_mapping = self.mounting_mappings[0, :]

        if mounting_offset.ndim == 1:
            mounting_offset = mounting_offset[np.newaxis, :]
        if mounting_mapping.ndim == 1:
            mounting_mapping = mounting_mapping[np.newaxis, :]

        if len(self.sensors) == 1:
            # This is the first sensor added, so no mounting mappings/offsets
            # to maintain
            self.mounting_offsets = mounting_offset
        else:
            self.mounting_offsets = np.concatenate([self.mounting_offsets,
                                                    mounting_offset])

        if len(self.sensors) == 1:
            # This is the first sensor added, so no mounting
            # mappings/offsets to maintain
            self.mounting_mappings = mounting_mapping
        else:
            self.mounting_mappings = np.concatenate([self.mounting_mappings,
                                                     mounting_mapping])

    def get_sensor_position(self, sensor):
        i = self.sensors.index(sensor)
        if self.is_moving():
            offsets = self._get_rotated_offset(i)
        else:
            offsets = StateVector(self.mounting_offsets[i, :])
        new_sensor_pos = self.get_position() + offsets
        return StateVector(new_sensor_pos)

    def get_sensor_orientation(self, sensor):
        i = self.sensors.index(sensor)
        vel = np.zeros([self.mounting_mappings.shape[1], 1])
        for j in range(self.mounting_mappings.shape[1]):
            vel[j, 0] = self.state.state_vector[
                self.mounting_mappings[i, j] + 1]
        abs_vel, heading = cart2pol(vel[0, 0], vel[1, 0])
        return StateVector([[0], [0], [heading]])

    def is_moving(self):
        return (hasattr(self, 'transition_model')
                and self.transition_model is not None
                and np.any(self.get_velocity() != 0))

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

        vel = self.get_velocity()

        rot = _get_rotation_matrix(vel)
        return np.transpose(np.dot(rot, self.mounting_offsets[i])[np.newaxis])


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
    return _rot3d_tuple(tuple(v[0] for v in vec))


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
    rot_z = _rot_z(yaw)
    # Modify to correct for new y axis
    y_axis = np.array([0, 1, 0])
    rot_y = expm(np.cross(np.eye(3), np.dot(rot_z, y_axis) * pitch))

    return np.dot(rot_y, rot_z)


def _rot_z(theta):
    """ Returns a rotation matrix which will rotate a vector around the Z axis
    in a counter clockwise direction by theta radians.

    Parameters
    ----------
    theta : float
        Required rotation angle in radians

    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])
