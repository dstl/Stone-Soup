# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm, norm
from math import cos, sin

from ..base import Property
from ..types import StateVector
from ..sensor import Sensor
from .base import Platform


class SensorPlatform(Platform):
    """A simple Platform that can carry a number of different sensors and is\
    capable of moving based upon the :class:`~.TransitionModel`.

    The location of platform mounted sensors will be maintained relative to \
    the sensor position. Simple platforms move within a cartesian space.

    Notes
    -----
    The current implementation of this class assumes an 2D Cartesian plane.

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
            #print("Sensor number ", i)
            # if False:
            if (hasattr(self, 'transition_model') &
                    (self.state.state_vector[
                             self.mounting_mappings[0]+1].max() > 0)):
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
            # print("Initial sensor position: ", self.mounting_offsets[i])
            self.sensors[i].set_position(StateVector(new_sensor_pos))
            # print("Sensor position updated to: ", self.sensors[i].position)

    def _get_rotated_offset(self, i):
        """ _get_rotated_offset - determines the sensor mounting offset for the
        platforms relative orientation.

        :param self: Platform object
        :param i: Sensor index within Platform object
        :return: Sensor mounting offset rotated relative to platform motion
        """
        axis = np.zeros([self.mounting_offsets.shape[1], 1])
        axis[0] = 1
        axis = np.transpose(axis)
        # vel = self.state.state_vector[self.mounting_mappings[0] + 1]
        vel = np.zeros([self.mounting_mappings.shape[1], 1])
        for j in range(self.mounting_mappings.shape[1]):
            vel[j, 0] = self.state.state_vector[
                self.mounting_mappings[i, j] + 1]
        # print("Platform velocity:", vel)
        # print("Default axis:", axis)
        theta = _get_angle(vel, axis)
        # print("Calculated angle:", theta)
        rot = _get_rotation_matrix(vel, theta)
        # print("Calculated rotation matrix:", rot)
        return np.transpose(np.dot(rot, self.mounting_offsets[i])[np.newaxis])


def _get_rotation_matrix(vel, theta):
    """ Generates a rotation matrix which can be used to determine the
    corrected sensor offsets.

    In the 2d case this returns the following rotation matrix
    [cos[theta] -sin[theta]]
    [cos[theta]  sin[theta]]

    :param vel: 1xD vector denoting platform velocity in D dimensions
    :param theta: rotation angle in radians
    :return: DxD rotation matrix
    """
    #print("Platform velocity:", vel)
    #print("length:", len(vel))
    if len(vel) == 3:
        return expm(np.cross(np.eye(3), np.transpose(vel / norm(vel) * theta)))
    elif len(vel) == 2:
        return np.array([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]])


def _get_angle(vel, axis):
    """ Returns the angle between two vectors, used to rotate the sensor offset
    relative to the platform velocity vector.

    :param vel: 1xD array denoting platform velocity
    :param axis: Dx1 array denoting sensor offset relative to platform
    :return: Angle between the two vectors in radians
    """
    vel_norm = vel / norm(vel)
    axis_norm = axis / norm(axis)

    return np.arccos(np.clip(np.dot(axis_norm, vel_norm), -1.0, 1.0))
