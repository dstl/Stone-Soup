import numpy as np
from typing import MutableSequence

from ..base import Property
from ..types.state import State, StateVector
from ..sensor.actionable import Actionable, ActionableProperty
from ..movable.movable import MovingMovable
from ..functions import cart2pol, cart2sphere
from .action.jerk_action import JerkActionGenerator


class ActionableMovementController(MovingMovable, Actionable):
    """An actionable movement controller where the list of `states` is
    an :class:`~.ActionableProperty`."""

    states: MutableSequence[State] = ActionableProperty(generator_cls=JerkActionGenerator)
    constraints: tuple = Property(doc="Max speed and acceleration.")
    position_mapping: tuple = Property()
    velocity_mapping: tuple = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.transition_model:
            self.transition_model = None
        self._cached_orientation = None

    def validate_timestamp(self):
        if self.timestamp:
            return True
        else:
            return False

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
            try:
                return self._cached_orientation
            except AttributeError:
                raise AttributeError(
                    'Orientation of a zero-velocity moving platform is not defined')
        velocity = self.velocity

        if self.ndim == 3:
            _, bearing, elevation = cart2sphere(*velocity.flat)
            self._cached_orientation = StateVector([0, elevation, bearing])
        elif self.ndim == 2:
            if len(self) >= 2 and np.all(self.velocity < 1e-6):
                c_pos = self.position
                p_pos = self[-2].state_vector[self.position_mapping, ]
                _, bearing = cart2pol(*(c_pos - p_pos))
            else:
                _, bearing = cart2pol(*velocity.flat)
            self._cached_orientation = StateVector([0, 0, bearing])
        else:
            raise NotImplementedError('Orientation of a moving platform is only implemented for 2'
                                      'and 3 dimensions')

        return self._cached_orientation

    def move(self, *args, **kwargs):
        return super().act(*args, **kwargs)
