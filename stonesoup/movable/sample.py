from collections.abc import Sequence

import numpy as np

from ..base import Property
from ..types.state import State
from . import FixedMovable
from .action.move_position_action import CircleSamplePositionActionGenerator


class _SampleActionableMovable(FixedMovable):
    """Base class for movables which sample actions. To be
    used with :class:`~.SamplePositionActionGenerator` types."""

    generator = None
    _generator_kwargs = {'action_space', 'action_mapping', 'n_samples'}

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

    n_samples: int = Property(
        default=10,
        doc="Number of samples to generate. This does not include the action "
        "to remain at the current position, meaning :attr:`n_samples` +1 "
        "actions will be generated."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_action = None

    def actions(self, timestamp, start_timestamp=None):
        """Method to return a set of sample action generators available up to a provided timestamp.

        A generator is returned for each actionable property that the sensor has.

        Parameters
        ----------
        timestamp: datetime.datetime
            Time of action finish.
        start_timestamp: datetime.datetime, optional
            Time of action start.

        Returns
        -------
        : set of :class:`~.SamplePositionActionGenerator`
            Set of grid action generators, that describe the bounds of each action space.
        """
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


class CircleSampleActionableMovable(_SampleActionableMovable):
    """This movable implements sampling based movement according to a circle
    around the current position, defined by the maximum travel defined by
    the user. Samples are uniformly generated within the circle. To be used
    with :class:`~.CircleSamplePositionActionGenerator` """

    generator = CircleSamplePositionActionGenerator
    _generator_kwargs = _SampleActionableMovable._generator_kwargs | {'maximum_travel'}

    maximum_travel: float = Property(
        default=1.0,
        doc="Maximum possible travel distance. Specifies the radius of "
        "sampling area."
    )
