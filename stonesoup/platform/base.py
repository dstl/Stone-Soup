# -*- coding: utf-8 -*-
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
                                    "set, it will default to `[m+1 for m in position_mapping]`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_mapping = coerce_to_valid_mapping(self.position_mapping)
        if self.velocity_mapping is None:
            self.velocity_mapping = self.position_mapping + 1

    @property
    def state_vector(self):
        return self.state.state_vector

    @property
    def position(self):
        return self.state_vector[self.position_mapping]

    @position.setter
    def position(self, value):
        self._set_position(value)

    @property
    def ndim(self):
        return len(self.position_mapping)

    @property
    @abstractmethod
    def orientation(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def velocity(self):
        raise NotImplementedError

    @abstractmethod
    def is_moving(self):
        raise NotImplementedError

    @abstractmethod
    def move(self, timestamp):
        raise NotImplementedError

    @abstractmethod
    def _set_position(self, value):
        raise NotImplementedError


class FixedPlatform(Platform):
    orientation = Property(StateVector, default=StateVector([0, 0, 0]),
                           doc='A fixed orientation of the static platform')

    def _set_position(self, value):
        self.state_vector[self.position_mapping] = value

    @property
    def velocity(self):
        return StateVector([0] * self.ndim)

    @property
    def is_moving(self):
        return False

    def move(self, timestamp):
        # Return without moving static platforms
        self.state.timestamp = timestamp


class MovingPlatform(Platform):
    """Moving platform base class

    A platform represents a random object defined as a :class:`~.State`
    that moves according to a given :class:`~.TransitionModel`.
    """
    transition_model = Property(
        TransitionModel, doc="Transition model")

    @property
    def velocity(self):
        # TODO docs
        # TODO return zeros?
        try:
            return self.state_vector[self.velocity_mapping]
        except IndexError:
            raise AttributeError('Velocity is not defined for this platform')

    @property
    def orientation(self):
        # TODO docs
        # TODO handle roll?
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
        # TODO docs
        # Note: a platform without a transition model can be given a velocity as part of it's
        # StateVector. It just won't move
        # This inconsistency will be handled in the move logic
        return np.any(self.velocity != 0)

    def _set_position(self, value):
        raise AttributeError('Cannot set the position of a moving platform')

    def move(self, timestamp=None, **kwargs):
        """Propagate the platform position using the :attr:`transition_model`.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the end of the maneuver \
            (the default is `None`)

        Notes
        -----
        This methods updates the value of :attr:`position`.

        Any provided `kwargs` are forwarded to the :attr:`transition_model`.

        If :attr:`transition_model` or `timestamp` is `None`, the method has
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
