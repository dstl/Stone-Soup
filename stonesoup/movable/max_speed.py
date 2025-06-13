from collections.abc import Sequence
import numpy as np

from stonesoup.base import Property
from stonesoup.movable import FixedMovable
from stonesoup.types.state import State

from .action.move_position_action import MaxSpeedPositionActionGenerator


class MaxSpeedActionableMovable(FixedMovable):
    """Class for movables can move in any direction. To be used with
    :class:`~.MoveToActionGenerator`."""

    generator = MaxSpeedPositionActionGenerator
    _generator_kwargs = {'action_space', 'action_mapping', 'resolution', 'angle_resolution',
                         'max_speed'}

    action_space: np.ndarray = Property(
        default=None,
        doc="The bounds of the action space that should not be exceeded. Of shape (ndim, 2) "
            "where ndim is the length of the action_mapping. For example, "
            ":code:`np.array([[xmin, xmax], [ymin, ymax]])`.")

    action_mapping: Sequence[int] = Property(
        default=(0, 1),
        doc="The state dimensions that actions are applied to.")

    resolution: float = Property(
        default=1,
        doc="The interval in distance travelled for each action.")

    angle_resolution: float = Property(
        default=np.pi/2,
        doc="The interval in angle for each action.")

    max_speed: float = Property(
        default=1,
        doc="The maximum speed that the movable can travel at.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_action = None

    def actions(self, timestamp, start_timestamp=None):
        """Method to return a set of action generators available up to a provided timestamp.

        A generator is returned for each actionable property that the sensor has.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time of action finish.
        start_timestamp: datetime.datetime, optional
            Time of action start.

        Returns
        -------
        : set of :class:`~.MoveToActionGenerator`
            Set of grid action generators, that describe the bounds of each action space.
        """
        if start_timestamp is None:
            start_timestamp = self.states[-1].timestamp
        generators = set()
        generators.add(self.generator(
            owner=self,
            attribute="position",
            start_time=start_timestamp,
            end_time=timestamp,
            **{name: getattr(self, name) for name in type(self)._generator_kwargs}))

        return generators

    def move(self, timestamp, *args, **kwargs):
        current_time = self.states[-1].timestamp
        new_state = State.from_state(self.state, timestamp=timestamp)
        new_state.state_vector = new_state.state_vector.copy()
        self.states.append(new_state)
        action = self._next_action
        if action is not None:
            self.position = action.act(current_time, timestamp, self.position)
        self._next_action = None

    def add_actions(self, actions):
        self._next_action = actions[0]
        return True

    def act(self, timestamp, *args, **kwargs):
        self.move(timestamp, *args, **kwargs)
