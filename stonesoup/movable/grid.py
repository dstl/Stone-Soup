from typing import Sequence
import numpy as np
from ..base import Property

from stonesoup.movable import FixedMovable
from stonesoup.movable.action.move_position_action import NStepDirectionalGridActionGenerator
from stonesoup.types.state import State


class _GridActionableMovable(FixedMovable):
    """Base class for movables which move in a grid like fashion. To be used with
    :class:`~.GridActionGenerator`."""

    generator = None
    _generator_kwargs = {'action_space', 'action_mapping', 'resolution'}

    action_space: np.ndarray = Property(
        default=None,
        doc="The bounds of the action space that should not be exceeded. Of shape (ndim, 2) "
            "where ndim is the length of the action_mapping. For example, "
            ":code:`np.array([[xmin, xmax], [ymin, ymax]])`."
    )

    action_mapping: Sequence[int] = Property(
        default=(0, 1),
        doc="The state dimensions that actions are applied to."
    )

    resolution: float = Property(
        default=1,
        doc="The size of each grid cell. Cells are assumed square."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_action = None
        self._generator_kwargs = _GridActionableMovable._generator_kwargs

    def actions(self, timestamp, start_timestamp=None):
        """Method to return a set of grid action generators available up to a provided timestamp.

        A generator is returned for each actionable property that the sensor has.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time of action finish.
        start_timestamp: datetime.datetime, optional
            Time of action start.

        Returns
        -------
        : set of :class:`~.GridActionGenerator`
            Set of grid action generators, that describe the bounds of each action space.
        """
        generators = set()
        generators.add(self.generator(
            owner=self,
            attribute="position",
            start_time=start_timestamp,
            end_time=timestamp,
            **{name: getattr(self, name) for name in self._generator_kwargs}))

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


# Actionable.register(_GridActionableMovable)


class NStepDirectionalGridMovable(_GridActionableMovable):
    """This is a movable that enables movement in a grid like fashion according
    to a number of steps and step sizes. Actions are applied symmetrically on the
    action space allowing for movement in both positive and negative directions
    of each axis. This movable implements the :class:`~.NStepDirectionalGridActionGenerator`"""

    generator = NStepDirectionalGridActionGenerator
    _generator_kwargs = _GridActionableMovable._generator_kwargs.update({'n_steps',
                                                                         'step_size'})

    n_steps: int = Property(
        default=1,
        doc="The number of steps that can be moved in either direction "
            "along specified dimensions"
    )

    step_size: int = Property(
        default=1,
        doc="The number of grid cells per step"
    )
