# -*- coding: utf-8 -*-
import datetime
import weakref
from abc import abstractmethod, ABC
from functools import lru_cache
from math import cos, sin
from typing import Sequence, MutableSequence, Optional, TYPE_CHECKING

from scipy.linalg import expm
import numpy as np

from ..functions import cart2sphere, cart2pol, rotz
from ..types.array import StateVector
from ..base import Property
from ..types.state import State, StateMutableSequence
from ..models.transition import TransitionModel

if TYPE_CHECKING:
    from ..sensor.base import BaseSensor


class Platform(StateMutableSequence, ABC):
    """A platform that can carry a number of different sensors.

        The location of platform mounted sensors will be maintained relative to
        the sensor position. Platforms move within a 2 or 3 dimensional
        rectangular cartesian space.

        A simple platform is considered to always be aligned with its principle
        velocity. It does not take into account issues such as bank angle or body
        deformation (e.g. flex).


        .. note:: This class abstract and not intended to be instantiated. To get the behaviour of
            this class use a subclass which gives movement
            behaviours. Currently these are :class:`~.FixedPlatform` and
            :class:`~.MovingPlatform`

        """
    states: Sequence[State] = Property(
        doc="A list of States which enables the platform's history to be "
            "accessed in simulators and for plotting. Initiated as a "
            "state, for a static platform, this would usually contain its "
            "position coordinates in the form ``[x, y, z]``. For a moving "
            "platform it would contain position and velocity interleaved: "
            "``[x, vx, y, vy, z, vz]``")
    position_mapping: Sequence[int] = Property(
        doc="Mapping between platform position and state vector. For a "
            "position-only 3d platform this might be ``[0, 1, 2]``. For a "
            "position and velocity platform: ``[0, 2, 4]``")
    rotation_offsets: MutableSequence[StateVector] = Property(
        default=None, readonly=True,
        doc="A list of StateVectors containing the sensor rotation "
            "offsets from the platform's primary axis (defined as the "
            "direction of motion). Defaults to a zero vector with the "
            "same length as the Platform's :attr:`position_mapping`")
    mounting_offsets: MutableSequence[StateVector] = Property(
        default=None, readonly=True,
        doc="A list of StateVectors containing the sensor translation "
            "offsets from the platform's reference point. Defaults to "
            "a zero vector with the same length as the Platform's "
            ":attr:`position_mapping`")
    sensors: MutableSequence['BaseSensor'] = Property(
        default=None, readonly=True,
        doc="A list of N mounted sensors. Defaults to an empty list")
    velocity_mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping between platform velocity and state dims. If not "
            "set, it will default to ``[m+1 for m in position_mapping]``")

    # TODO: Determine where a platform coordinate frame should be maintained

    def __init__(self, *args, **kwargs):
        """
        Ensure that the platform location and the sensor locations are
        consistent at initialisation.
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided

        if self.sensors is None:
            self._property_sensors = []

        if self.velocity_mapping is None:
            self.velocity_mapping = [p + 1 for p in self.position_mapping]

        if self.mounting_offsets is None:
            self._property_mounting_offsets = [StateVector([0] * self.ndim)] * len(self.sensors)

        if self.rotation_offsets is None:
            self._property_rotation_offsets = [StateVector([0] * 3)] * len(self.sensors)

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

    @property
    def position(self) -> StateVector:
        """Return the position of the platform.

        Extracted from the state vector of the platform using the platform's
        :attr:`position_mapping`. This property is settable for fixed platforms, but not for
        movable ones, where the position must be set by moving the model with a transition model.
        """
        return self.state_vector[self.position_mapping, :]

    @position.setter
    def position(self, value: StateVector) -> None:
        self._set_position(value)

    @staticmethod
    def _tuple_or_none(value):
        return None if value is None else tuple(value)

    @sensors.getter
    def sensors(self):
        return self._tuple_or_none(self._property_sensors)

    @mounting_offsets.getter
    def mounting_offsets(self):
        return self._tuple_or_none(self._property_mounting_offsets)

    @rotation_offsets.getter
    def rotation_offsets(self):
        return self._tuple_or_none(self._property_rotation_offsets)

    @property
    def ndim(self) -> int:
        """Convenience property to return the number of dimensions in which the platform operates.

        Given by the length of the :attr:`position_mapping`
        """
        return len(self.position_mapping)

    @property
    @abstractmethod
    def orientation(self) -> StateVector:
        """Return the orientation of the platform.

        Implementation is case dependent and left to the Fixed/Moving subclasses
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def velocity(self) -> StateVector:
        """Return the velocity of the platform.

        Implementation is case dependent and left to the Fixed/Moving subclasses
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_moving(self) -> bool:
        """Return the ``True`` if the platform is moving, ``False`` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def move(self, timestamp: datetime.datetime, **kwargs) -> None:
        """Update the platform position using the :attr:`transition_model`.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the end of the maneuver \
            (the default is ``None``)

        Notes
        -----
        This methods updates the value of :attr:`position`.

        Any provided ``kwargs`` are forwarded to the :attr:`transition_model`.

        If :attr:`transition_model` or ``timestamp`` is ``None``, the method has
        no effect, but will return successfully.

        """
        raise NotImplementedError

    @abstractmethod
    def _set_position(self, value: StateVector) -> None:
        raise NotImplementedError

    def add_sensor(self, sensor: "BaseSensor", mounting_offset: Optional[StateVector] = None,
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
        self._property_sensors.append(sensor)
        sensor.platform_system = weakref.ref(self)

        if mounting_offset is None:
            mounting_offset = StateVector([0] * self.ndim)
        if rotation_offset is None:
            rotation_offset = StateVector([0] * 3)

        self._property_mounting_offsets.append(mounting_offset)
        self._property_rotation_offsets.append(rotation_offset)

    def remove_sensor(self, sensor: "BaseSensor") -> None:
        """ Remove a sensor from the platform

        Parameters
        ----------
        sensor : :class:`~.BaseSensor`
            The sensor object to remove
        """
        self.pop_sensor(self._property_sensors.index(sensor))

    def pop_sensor(self, index: int):
        """ Remove a sensor from the platform by index

                Parameters
                ----------
                index : int
                    The index of the sensor to remove
                """
        self._property_sensors.pop(index)
        self._property_mounting_offsets.pop(index)
        self._property_rotation_offsets.pop(index)

    def get_sensor_position(self, sensor: "BaseSensor") -> StateVector:
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

    def get_sensor_orientation(self, sensor: "BaseSensor") -> StateVector:
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


class FixedPlatform(Platform):
    """Fixed platform base class

        A platform represents a random object defined as a :class:`~.State`
        with fixed (but settable) position and orientation.

        .. note:: Position and orientation are a read/write properties in this class.
        """
    orientation: StateVector = Property(
        default=None,
        doc='A fixed orientation of the static platform. Defaults to the zero vector')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.orientation is None:
            self.orientation = StateVector([0, 0, 0])

    def _set_position(self, value: StateVector) -> None:
        self.state_vector[self.position_mapping, :] = value

    @property
    def velocity(self) -> StateVector:
        """Return the velocity of the platform.

        For a fixed platform this is always a zero vector of length :attr:`ndim`.
        """
        return StateVector([0] * self.ndim)

    @property
    def is_moving(self) -> bool:
        return False

    def move(self, timestamp: datetime.datetime, **kwargs) -> None:
        """For a fixed platform this method has no effect other than to update the timestamp."""
        # TODO Is this a sensible implementation?
        # Return without moving static platforms
        self.state.timestamp = timestamp


class MovingPlatform(Platform):
    """Moving platform base class

    A platform represents a random object defined as a :class:`~.State`
    that moves according to a given :class:`~.TransitionModel`.

    .. note:: Position and orientation are a read only properties in this class.
    """
    transition_model: TransitionModel = Property(doc="Transition model")

    @property
    def velocity(self) -> StateVector:
        """Return the velocity of the platform.

        Extracted from the state vector of the platform using the platform's
        :attr:`velocity_mapping`. If the state vector is too short and does not contain the
        elements specified in the :attr:`velocity_mapping` this raises an :class:`AttributeError`
        """
        try:
            return self.state_vector[self.velocity_mapping, :]
        except IndexError:
            raise AttributeError('Velocity is not defined for this platform')

    @property
    def orientation(self) -> StateVector:
        """Return the orientation of the platform.

        This is defined as a 3x1 StateVector of angles (rad), specifying the sensor orientation in
        terms of the counter-clockwise rotation around each Cartesian axis in the order
        :math:`x,y,z`. The rotation angles are positive if the rotation is in the counter-clockwise
        direction when viewed by an observer looking along the respective rotation axis,
        towards the origin.

        The orientation of this platform is defined as along the direction of its velocity, with
        roll always set to zero (as this is the angle the platform is rotated about the velocity
        axis, which is not defined in this approximation).

        Notes
        -----
        A non-moving platform (``self.is_moving == False``) does not have a defined orientation in
        this approximations and so raises an :class:`AttributeError`
        """
        if not self.is_moving:
            raise NotImplementedError('Orientation of a zero-velocity moving platform is not'
                                      'defined')
        velocity = self.velocity

        if self.ndim == 3:
            _, bearing, elevation = cart2sphere(*velocity.flat)
            return StateVector([0, elevation, bearing])
        elif self.ndim == 2:
            _, bearing = cart2pol(*velocity.flat)
            return StateVector([0, 0, bearing])
        else:
            raise NotImplementedError('Orientation of a moving platform is only implemented for 2'
                                      'and 3 dimensions')

    @property
    def is_moving(self) -> bool:
        """Return the ``True`` if the platform is moving, ``False`` otherwise.

        Equivalent (for this class) to ``all(v == 0 for v in self.velocity)``
        """
        # Note: a platform without a transition model can be given a velocity as part of it's
        # StateVector. It just won't move
        # This inconsistency is handled in the move logic
        return np.any(self.velocity != 0)

    def _set_position(self, value: StateVector):
        # The logic below is this: if a moving platform is being built from (say) input
        # real-world data then its transition model would not be set, and so it would be fine to
        # set its position. However, if the transition model is set, then setting the position is
        # both unexpected and may cause odd effects, so is forbidden
        if self.transition_model is None:
            self.state_vector[self.position_mapping, :] = value
        else:
            raise AttributeError('Cannot set the position of a moving platform with a '
                                 'transition model')

    def move(self, timestamp=None, **kwargs) -> None:
        """Propagate the platform position using the :attr:`transition_model`.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the end of the maneuver \
            (the default is ``None``)

        Notes
        -----
        This methods updates the value of :attr:`position`.

        Any provided ``kwargs`` are forwarded to the :attr:`transition_model`.

        If :attr:`transition_model` or ``timestamp`` is ``None``, the method has
        no effect, but will return successfully.

        """

        if self.state.timestamp is None:
            self.state.timestamp = timestamp
            return

        # Compute time_interval
        try:
            time_interval = timestamp - self.state.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            return

        if self.transition_model is None:
            raise AttributeError('Platform without a transition model cannot be moved')

        self.states.append(State(
            state_vector=self.transition_model.function(
                state=self.state,
                noise=True,
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs),
            timestamp=timestamp))


class MultiTransitionMovingPlatform(MovingPlatform):
    """Moving platform with multiple transition models

    A list of transition models are given with corresponding transition times, dictating the
    movement behaviour of the platform for given durations.
    """

    transition_models: Sequence[TransitionModel] = Property(doc="List of transition models")
    transition_times: Sequence[datetime.timedelta] = Property(doc="Durations for each listed "
                                                                  "transition model")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.transition_models) != len(self.transition_times):
            raise AttributeError('transition_models and transition_times must be same length')

        self.transition_index = 0
        self.current_interval = self.transition_times[0]

    @property
    def transition_model(self):
        return self.transition_models[self.transition_index]

    def move(self, timestamp=None, **kwargs) -> None:
        """Propagate the platform position using the :attr:`transition_model`.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying the end of the maneuver (the default is ``None``)

        Notes
        -----
        This methods updates the value of :attr:`position`.

        Any provided ``kwargs`` are forwarded to the :attr:`transition_model`.

        If :attr:`transition_model` or ``timestamp`` is ``None``, the method has
        no effect, but will return successfully.

        This method updates :attr:`transition_model`, :attr:`transition_index` and
        :attr:`current_interval`:
        If the timestamp provided gives a time delta greater than :attr:`current_interval` the
        :attr:`transition_model` is called for the rest of its corresponding duration, and the move
        method is called again on the next transition model (by incrementing
        :attr:`transition_index`) in :attr:`transition_models` with the residue time delta.
        If the time delta is less than :attr:`current_interval` the :attr:`transition_model` is
        called for that duration and :attr:`current_interval` is reduced accordingly.
        """
        if self.state.timestamp is None:
            self.state.timestamp = timestamp
            return
        try:
            time_interval = timestamp - self.state.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            return

        temp_state = self.state
        while time_interval != 0:
            if time_interval >= self.current_interval:
                temp_state = State(
                    state_vector=self.transition_model.function(
                        state=temp_state,
                        noise=True,
                        time_interval=self.current_interval,
                        **kwargs),
                    timestamp=timestamp
                )
                time_interval -= self.current_interval
                self.transition_index = (self.transition_index + 1) % len(self.transition_models)
                self.current_interval = self.transition_times[self.transition_index]

            else:
                temp_state = State(
                    state_vector=self.transition_model.function(
                        state=temp_state,
                        noise=True,
                        time_interval=time_interval,
                        **kwargs),
                    timestamp=timestamp
                )
                self.current_interval -= time_interval
                time_interval = 0
        self.states.append(temp_state)


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
    else:
        raise NotImplementedError


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
