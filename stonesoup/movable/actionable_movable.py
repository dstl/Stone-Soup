from typing import Callable
from ..base import Property

from stonesoup.movable import FixedMovable
from stonesoup.movable.action.move_position_action import GridActionGenerator
from stonesoup.types.state import State


class GridActionableMovable(FixedMovable):
    """  """

    generator: Callable = Property(
        doc="The generator used to provide possible actions"
    )

    generator_params: dict = Property(
        default={},
        doc="Dictionary of specific parameters for :attr:`generator`"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._next_action = None
        if not issubclass(self.generator, GridActionGenerator):
            raise TypeError('Only GridActionGenerator types are compatible with this movable.')

    #         self.scheduled_actions = dict()

    def actions(self, timestamp, start_timestamp=None):
        generators = set()
        generators.add(self.generator(owner=self,
                                      attribute="position",
                                      start_time=start_timestamp,
                                      end_time=timestamp,
                                      **self.generator_params))

        return generators

    def move(self, timestamp, *args, **kwargs):
        current_time = self.states[-1].timestamp
        super().move(timestamp, *args, **kwargs)
        action = self._next_action
        if action is not None:
            self.position = action.act(current_time, timestamp, self.position)
        self._next_action = None

    def add_actions(self, actions):
        self._next_action = actions[0]
        #         for name in self._actionable_properties:
        #             for action in actions:
        #                 if action.generator.attribute ==name:
        #                     self.scheduled_actions[name] = action
        return True

    def act(self, timestamp, *args, **kwargs):
        self.move(timestamp, *args, **kwargs)
