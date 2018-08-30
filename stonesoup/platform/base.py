# -*- coding: utf-8 -*-

from ..base import Base, Property
from ..types.state import State
from ..models.transition import TransitionModel


class Platform(Base):
    """Platform base class

    A platform represents a random object that moves according to a given
    :class:`~.TransitionModel`.
    """

    state = Property(State, doc="The platform state at any given point")
    transition_model = Property(
        TransitionModel, doc="Transition model")

    def move(self, timestamp=None, **kwargs):
        """Propagate the platform position using the :attr:`transition_model`.

        Parameters
        ----------
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the end of the maneuver \
            (the default is `None`)

        Notes
        -----
        This methods also updates the value of :attr:`position`.

        Any provided `kwargs` are forwarded to the :attr:`transition_model`.

        If :attr:`transition_model` is `None`, the method has no effect, but
        will return successfully.

        """

        # Compute time_interval
        try:
            time_interval = timestamp - self.state.timestamp
        except TypeError as e:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None
        self.state = State(
            state_vector=self.transition_model.function(
                state_vector=self.state.state_vector,
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs),
            timestamp=timestamp)
