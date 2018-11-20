# -*- coding: utf-8 -*-
import numpy as np
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

    In the 2d case this will be a 3x3 matrix which rotates around the Z axis
    followed by a rotation about the new Y axis.

    :param vel: 1xD vector denoting platform velocity in D dimensions
    :param theta: rotation angle in radians
    :return: DxD rotation matrix
    """
    if len(vel) == 3:
        return _rot3d(vel)
    elif len(vel) == 2:
        return np.array([[cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)]])


def _rot3d(vel):
    """
    This approach determines the platforms attitude based upon its velocity
    component. It does not take into account potential platform roll, nor
    are the components calculated to account for physical artifacts such as
    platform trim (e.g. aircraft).

    This involves specifying its rotation about the vertical (Z) and forward
    (Y) axis. These are captured as Yaw, measured as a counter-clockwise
    rotation in the X-Y plane around the Z axis, and Pitch, measured as the
    angle that the platfrom intersects makes with the X-Y plane and applied as
    a rotation to the post rotated Y axis.

    Notes
    -----
    numpys Arccos output is only defined over range [0:pi], therefore sign of
    input must be considered

    :param vel:
    :return:
    """
    if np.absolute(vel[[1, 2]]).sum() == 0:
        # Static or all velocity in primary axis
        return np.eye(3)
    elif ((np.absolute(vel[[0, 2]]).sum() == 0) or
            ((vel[[0]] != 0) and (vel[[1]] != 0) and (vel[[2]] == 0)) or
            ((np.absolute(vel[[1, 2]]).sum() == 0) and vel[[0]] < 0)):
        # Velocity in [x,y] plane or [y] axis
        #  vel_norm = vel[[0, 1]] / np.linalg.norm(vel[[0, 1]])
        gamma = _get_angle(np.array([[1, 0]]),
                           vel[[0, 1]] / np.linalg.norm(vel[[0, 1]]))
        # gamma = np.arccos(
        #     np.clip(np.dot(np.array([[1, 0]]), vel_norm), -1.0, 1.0))
        if vel[[1]] < 0:
            gamma *= -1
        return _rotZ(gamma)
    elif ((np.absolute(vel[[0,1]]).sum() == 0) or
            ((vel[[0]] != 0) and (vel[[1]] == 0) and (vel[[2]] != 0))):
        # Velocity in [x,z] plane or [z] axis
        # vel_norm = vel[[0, 2]] / np.linalg.norm(vel[[0, 2]])
        #  beta = np.arccos(
        #    np.clip(np.dot(np.array([[1, 0]]), vel_norm), -1.0, 1.0))
        beta = _get_angle(np.array([[1, 0]]),
                        vel[[0, 2]] / np.linalg.norm(vel[[0, 2]]))
        if vel[[2]] > 0:
            beta *= -1
        return _rotY(beta)
    else:
        # Velocity in [y,z] plane or all axis
        #  vel_norm = vel[[0, 1]] / np.linalg.norm(vel[[0, 1]])
        #  gamma = np.arccos(
        #      np.clip(np.dot(np.array([[1, 0]]), vel_norm), -1.0, 1.0))
        gamma = _get_angle(np.array([[1, 0]]),
                           vel[[0, 1]] / np.linalg.norm(vel[[0, 1]]))
        vel_norm = vel / np.linalg.norm(vel)
        interim = np.arround(np.transpose(
            np.dot(_rotZ(gamma), vel_norm))[0], 16)
        #  beta = np.arccos(np.clip(np.dot(interim, vel_norm), -1.0, 1.0))
        beta = _get_angle(interim, vel_norm)
        if vel[[2]] > 0:
            beta *= -1
        if vel[[1]] < 0:
            gamma *= -1
            if vel[[2]] > 0:
                beta += (np.pi / 2)
        return np.dot(np.around(_rotZ(gamma), 16), np.around(_rotY(beta), 16))


def _get_angle(vel, axis):
    """ Returns the angle between two vectors, used to rotate the sensor offset
    relative to the platform velocity vector.

    :param vel: 1xD array denoting platform velocity
    :param axis: Dx1 array denoting sensor offset relative to platform
    :return: Angle between the two vectors in radians
    """
    vel_norm = vel / np.linalg.norm(vel)
    axis_norm = axis / np.linalg.norm(axis)

    return np.arccos(np.clip(np.dot(axis_norm, vel_norm), -1.0, 1.0))


def _rotY(theta):
    """ Returns a rotation matrix which will rotate a vector around the Y axis
    in a counter clockwise direction by theta radians

    :param theta:
    :return: 3x3 rotation matrix
    """
    return np.array([[cos(theta), 0, sin(theta)],
                     [0, 1, 0],
                     [-sin(theta), 0, cos(theta)]])


def _rotZ(theta):
    """ Returns a rotation matrix which will rotate a vector around the Z axis
    in a counter clockwise direction by theta radians.

    :param theta:
    :return: 3x3 rotation matrix
    """
    return np.array([[cos(theta), -sin(theta), 0],
                     [sin(theta), cos(theta), 0],
                     [0, 0, 1]])


