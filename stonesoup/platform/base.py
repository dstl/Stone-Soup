# -*- coding: utf-8 -*-
import datetime
from abc import abstractmethod, ABC

from ..functions import cart2sphere, cart2pol, coerce_to_valid_mapping
from ..types.array import StateVector
from ..base import Base, Property
from ..types.state import State
from ..models.transition import TransitionModel
import numpy as np


class Platform(Base, ABC):
    state = Property(State, doc="The platform state at any given point")
    position_mapping = Property(np.ndarray, doc="Mapping between platform position and state dims")
    velocity_mapping = Property(np.ndarray, default=None,
                                doc="Mapping between platform velocity and state dims. If not "
                                    "set, it will default to ``[m+1 for m in position_mapping]``")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_mapping = coerce_to_valid_mapping(self.position_mapping)
        if self.velocity_mapping is None:
            self.velocity_mapping = self.position_mapping + 1

    @property
    def state_vector(self) -> StateVector:
        """Convenience property to return the state vector of the state."""
        return self.state.state_vector

    @property
    def position(self) -> StateVector:
        """Return the position of the platform.

        Extracted from the state vector of the platform using the platform's
        :attr:`position_mapping`. This property is settable for fixed platforms, but not for
        movable ones, where the position must be set by moving the model with a transition model.
        """
        return self.state_vector[self.position_mapping]

    @position.setter
    def position(self, value: StateVector) -> None:
        self._set_position(value)

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


class FixedPlatform(Platform):
    """Fixed platform base class

        A platform represents a random object defined as a :class:`~.State`
        with fixed (but settable) position and orientation.

        .. note:: Position and orientation are a read/write properties in this class.
        """
    orientation = Property(StateVector, default=StateVector([0, 0, 0]),
                           doc='A fixed orientation of the static platform')

    def _set_position(self, value: StateVector) -> None:
        self.state_vector[self.position_mapping] = value

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
        # TODO should this be the case?
        # Return without moving static platforms
        self.state.timestamp = timestamp


class MovingPlatform(Platform):
    """Moving platform base class

    A platform represents a random object defined as a :class:`~.State`
    that moves according to a given :class:`~.TransitionModel`.

    .. note:: Position and orientation are a read only properties in this class.
    """
    transition_model = Property(
        TransitionModel, doc="Transition model")

    @property
    def velocity(self) -> StateVector:
        """Return the velocity of the platform.

        Extracted from the state vector of the platform using the platform's
        :attr:`velocity_mapping`. If the state vector is too short and does not contain the
        elements specified in the :attr:`velocity_mapping` this raises an :class:`AttributeError`
        """
        try:
            return self.state_vector[self.velocity_mapping]
        except IndexError:
            raise AttributeError('Velocity is not defined for this platform')

    @property
    def orientation(self) -> StateVector:
        """Return the orientation of the platform.

        This is defined as three-element :class:`~.StateVector` in the form ``[roll, yaw, pitch]``
        or equivalently ``[roll, bearing, elevation]``.

        The orientation of this platform is defined as along the direction of its velocity, with
        roll always set to zero (as this is the angle the platform is rotated about the velocity
        axis, which is not defined in this approximation).

        Notes
        -----
        A non-moving platform (``self.is_moving == False``) does not have a defined orientation in
        this approximations and so raises an :class:`AttributeError`
        """
        if not self.is_moving:
            raise AttributeError('Orientation of a zero-velocity moving platform is not defined')
        velocity = self.velocity

        if self.ndim == 3:
            _, bearing, elevation = cart2sphere(*velocity.flat)
            return StateVector([0, bearing, elevation])
        elif self.ndim == 2:
            _, bearing = cart2pol(*velocity.flat)
            return StateVector([0, bearing])

    @property
    def is_moving(self):
        """Return the ``True`` if the platform is moving, ``False`` otherwise.

        Equivalent (for this class) to ``all(v == 0 for v in self.velocity)``
        """
        # Note: a platform without a transition model can be given a velocity as part of it's
        # StateVector. It just won't move
        # This inconsistency is handled in the move logic
        return np.any(self.velocity != 0)

    def _set_position(self, value):
        raise AttributeError('Cannot set the position of a moving platform')

    def move(self, timestamp=None, **kwargs):
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
        # Compute time_interval
        try:
            time_interval = timestamp - self.state.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            return

        if self.transition_model is None:
            raise AttributeError('Platform without a transition model cannot be moved')

        self.state = State(
            state_vector=self.transition_model.function(
                state=self.state,
                noise=True,
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs),
            timestamp=timestamp)
