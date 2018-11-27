# -*- coding: utf-8 -*-
import numpy as np
from math import cos, sin
from scipy.linalg import expm

from ..base import Property
from ..types import StateVector
from ..sensor import Sensor
from .base import Platform


class SensorPlatform(Platform):
    """A simple Platform that can carry a number of different sensors and is\
    capable of moving based upon the :class:`~.TransitionModel`.

    The location of platform mounted sensors will be maintained relative to \
    the sensor position. Simple platforms move within a 2 or 3 dimensional \
    rectangular cartesian space.

    """

    sensors = Property([Sensor], doc="A list of N mounted sensors")
    mounting_offsets = Property(
        [np.array], doc="A list of sensor offsets (For now expressed\
                            as a Nxn array of nD Cartesian coordinates)")
    mounting_mappings = Property(
        [np.array], doc="Mappings between the platform state vector and the\
                            individuals sensors mounting offset (For now\
                            expressed as a nxN array of nD Cartesian\
                            coordinates or a 1xN array where a single\
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

        self._move_sensors()

    # TODO: create add_sensor method

    def move(self, timestamp=None, **kwargs):
        """Propagate the platform position using the :attr:`transition_model`,\
        and use _move_sensors method to update sensor positions, this in turn \
        calls _get_rotated_offset to modify sensor offsets relative to the \
        platforms velocity vector

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the maneuver completes \
            (the default is `None`)

        """
        # Call superclass method to update platform state
        super().move(timestamp=timestamp, **kwargs)
        # Move the platforms sensors relative to the platform
        self._move_sensors()

    def _move_sensors(self):
        """ Propogate the Sensor positions based upon the mounting
        offsets and the platform position and heading post manoeuvre.

        Notes
        -----
        Method assumes that if a platform has a transition model it will have
        velocity components. A sensor offset will therefore be rotated based
        upon the platforms velocity (i.e. direction of motion)
        """

        # Update the positions of all sensors relative to the platform
        for i in range(len(self.sensors)):
            if (hasattr(self, 'transition_model') &
                    (np.absolute(self.state.state_vector[
                             self.mounting_mappings[0]+1]).max() > 0)):
                new_sensor_pos = self._get_rotated_offset(i)
                for j in range(self.mounting_offsets.shape[1]):
                    new_sensor_pos[j] = new_sensor_pos[j] + \
                                           (self.state.state_vector[
                                                self.mounting_mappings[i, j]])
            else:
                new_sensor_pos = np.zeros([self.mounting_offsets.shape[1], 1])
                for j in range(self.mounting_offsets.shape[1]):
                    new_sensor_pos[j] = (self.state.state_vector[
                                            self.mounting_mappings[i, j]] +
                                         self.mounting_offsets[i, j])
            self.sensors[i].set_position(StateVector(new_sensor_pos))

    def _get_rotated_offset(self, i):
        """ _get_rotated_offset - determines the sensor mounting offset for the
        platforms relative orientation.

        :param self: Platform object
        :param i: Sensor index within Platform object
        :return: Sensor mounting offset rotated relative to platform motion
        """

        vel = np.zeros([self.mounting_mappings.shape[1], 1])
        for j in range(self.mounting_mappings.shape[1]):
            vel[j, 0] = self.state.state_vector[
                self.mounting_mappings[i, j] + 1]

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

    :param vel: 1xD vector denoting platform velocity in D dimensions
    :return: DxD rotation matrix
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

    :param vec: 1xD array denoting platform velocity
    :param axis: Dx1 array denoting sensor offset relative to platform
    :return: Angle between the two vectors in radians
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

    :param vec: platform velocity
    :return: np.array 3x3 rotation matrix
    """
    # TODO handle platform roll
    yaw = np.arctan2(vec[[1]], vec[[0]])
    pitch = np.arctan2(vec[[2]],
                       np.sqrt(vec[[0]] ** 2 + vec[[1]] ** 2)) * -1
    rot_z = _rot_z(yaw)
    # Modify to correct for new y axis
    y_axis = np.array([0, 1, 0])
    rot_y = expm(np.cross(np.eye(3), np.dot(rot_z, y_axis) * pitch))

    return np.dot(rot_y, rot_z)


def _rot_z(theta):
    """ Returns a rotation matrix which will rotate a vector around the Z axis
    in a counter clockwise direction by theta radians.

    :param theta:
    :return: 3x3 rotation matrix
    """
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])
