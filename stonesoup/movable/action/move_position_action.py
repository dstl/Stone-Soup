import numpy as np
from abc import abstractmethod
from typing import Iterator, Sequence

from ...base import Property
from ...sensor.action import ActionGenerator, Action
from ...types.state import StateVector


class MovePositionAction(Action):
    """This is the base class for an action that changes the
    position of a platform or sensor."""

    def act(self, current_time, timestamp, init_value):
        return self.target_value


class GridActionGenerator(ActionGenerator):
    """This is the base class for generators that generate actions in a grid like fashion."""

    action_space: np.ndarray = Property(
        default=None,
        doc="The bounds of the action space that should not be exceeded. Of shape (ndim, 2)"
    )

    action_mapping: Sequence[int] = Property(
        default=(0, 1),
        doc="The state dimensions that actions are applied to."
    )

    resolution: float = Property(
        default=1,
        doc="The size of each grid cell. Cells are assumed square."
    )

    def __contains__(self, item):
        return item in iter(self)

    @abstractmethod
    def __iter__(self) -> Iterator[MovePositionAction]:
        raise NotImplementedError


class NStepDirectionalGridActionGenerator(GridActionGenerator):
    """This is a grid action generator that enables movement by a number
    of steps in the specified directions. Actions are applied symmetrically so
    can move by a number of steps in positive and negative directions along
    the specified dimensions."""

    n_steps: int = Property(
        default=1,
        doc="The number of steps that can be moved in either direction "
            "along specified dimensions"
    )

    step_size: int = Property(
        default=1,
        doc="The number of grid cells per step"
    )

    @property
    def default_action(self):
        return MovePositionAction(generator=self,
                                  end_time=self.end_time,
                                  target_value=self.current_value)

    def __iter__(self):
        yield MovePositionAction(generator=self,
                                 end_time=self.end_time,
                                 target_value=self.current_value)

        action_deltas = np.linspace(-1*self.n_steps*self.step_size*self.resolution,
                                    self.n_steps*self.step_size*self.resolution,
                                    2*self.n_steps+1)

        for dim in self.action_mapping:
            for n in action_deltas:
                if n == 0:
                    continue
                value = StateVector(np.zeros(len(self.current_value)))
                value[dim] += n
                if self.action_space is not None and \
                    (np.any(self.current_value + value < self.action_space[:, 0])
                     or np.any(self.current_value + value > self.action_space[:, 1])):
                    continue
                else:
                    yield MovePositionAction(generator=self,
                                             end_time=self.end_time,
                                             target_value=self.current_value + value)
