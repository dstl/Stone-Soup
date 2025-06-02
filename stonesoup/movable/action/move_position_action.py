from abc import abstractmethod
from collections.abc import Iterator, Sequence
from itertools import product

import numpy as np

from ...base import Property
from ...sensormanager.action import ActionGenerator, Action
from ...types.state import StateVector


class MovePositionAction(Action):
    """This is the base class for an action that changes the
    position of a platform or sensor."""

    def act(self, current_time, timestamp, init_value):
        return self.target_value


class MovePositionActionGenerator(ActionGenerator):
    """This is the base class for generators that generate :class:`~MovePositionAction`s."""

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

    epsilon = 1e-6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.action_space is not None:
            if len(self.action_space) != len(self.action_mapping):
                raise ValueError(f"Dimensions of action_space {self.action_space.shape} "
                                 f"are not compatible with action_mapping of length "
                                 f"{len(self.action_mapping)}. action_space should be "
                                 f"of shape (ndim, 2) where ndim is the length of the "
                                 f"action_mapping.")

            if (np.any(self.current_value[self.action_mapping, :] < self.action_space[:, [0]])
                    or np.any(self.current_value[self.action_mapping, :] > self.action_space[:, [1]])):  # noqa: E501
                raise ValueError(f"Initial platform location {self.current_value} is not within "
                                 f"the bounds of the action space {self.action_space}.")

    def __contains__(self, item):
        return item in iter(self)

    @property
    @abstractmethod
    def maximum_travel(self):
        pass

    def action_from_value(self, value=None) -> MovePositionAction:
        """
        Given a value for the position, generates the action would achieve that value.

        Parameters
        ----------
        value: StateVector
            Property value for which the action is required.

        Returns
        -------
        MovePositionAction
            Action which will achieve this position.
        """
        if value not in self:
            return None

        return MovePositionAction(generator=self,
                                  end_time=self.end_time,
                                  target_value=value)


class GridActionGenerator(MovePositionActionGenerator):
    """This is the base class for generators that generate actions in a grid like fashion."""

    resolution: float = Property(
        default=1,
        doc="The size of each grid cell. Cells are assumed square."
    )

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

    @property
    def maximum_travel(self):
        return self.step_size * self.n_steps

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
                target_value = self.current_value + value
                if self.action_space is None or \
                    (np.all(target_value[self.action_mapping, :] >= self.action_space[:, [0]])
                     and np.all(target_value[self.action_mapping, :] <= self.action_space[:, [1]])):  # noqa: E501
                    yield MovePositionAction(generator=self,
                                             end_time=self.end_time,
                                             target_value=target_value)


class SamplePositionActionGenerator(MovePositionActionGenerator):
    """Base action generator for sampling approaches to action generation. The action
    generator requires the user to define a number of samples to generate
    (:attr:`n_samples`) according to the defined sampling technique."""

    n_samples: int = Property(
        default=10,
        doc="Number of samples to generate. This does not include the action "
        "to remain at the current position, meaning :attr:`n_samples` +1 "
        "actions will be generated."
    )

    @property
    def default_action(self):
        return MovePositionAction(generator=self,
                                  end_time=self.end_time,
                                  target_value=self.current_value)

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class CircleSamplePositionActionGenerator(SamplePositionActionGenerator):
    """Action generator which samples candidate future positions uniformly within
    a circle around the current position. Circle radius is defined by the user
    with :attr:`maximum_travel`. This generator is only applicable to 2D position
    actions."""

    maximum_travel: float = Property(
        default=1.0,
        doc="Maximum possible travel distance. Specifies the radius of "
        "sampling area."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.action_mapping) != 2:
            raise ValueError(f"Action mapping {self.action_mapping} does "
                             f"not have 2 dimensions. "
                             f":class:`~.CircleSamplePositionActionGenerator` "
                             f"is designed for 2D action generation only.")

    def __iter__(self):

        yield MovePositionAction(generator=self,
                                 end_time=self.end_time,
                                 target_value=self.current_value)

        radius_angle_samples = np.random.uniform([0, 0], [1, 2*np.pi], (self.n_samples, 2))

        sample_values = self.maximum_travel*np.sqrt(radius_angle_samples[:, 0]) *\
            np.array([np.sin(radius_angle_samples[:, 1]), np.cos(radius_angle_samples[:, 1])])
        values = np.zeros((self.current_value.shape[0], self.n_samples))
        values[self.action_mapping,] = sample_values

        target_values = self.current_value + values

        if self.action_space is not None:
            while (np.any(target_values[self.action_mapping, :] < self.action_space[:, [0]])
                    or np.any(target_values[self.action_mapping, :] > self.action_space[:, [1]])):

                _, idx = np.where(np.logical_or(
                    target_values[self.action_mapping, :] > self.action_space[:, [1]],
                    target_values[self.action_mapping, :] < self.action_space[:, [0]]))
                radius_angle_samples = np.random.uniform([0, 0], [1, 2*np.pi], (len(idx), 2))
                sample_values = self.maximum_travel*np.sqrt(radius_angle_samples[:, 0]) *\
                    np.array([np.sin(radius_angle_samples[:, 1]),
                              np.cos(radius_angle_samples[:, 1])])
                values = np.zeros((self.current_value.shape[0], len(idx)))
                values[self.action_mapping,] = sample_values
                target_values[:, idx] = self.current_value + values

        for target_value in target_values.T:
            yield MovePositionAction(generator=self,
                                     end_time=self.end_time,
                                     target_value=StateVector(target_value))

    def __contains__(self, item):
        if isinstance(item, MovePositionAction):
            item = item.target_value
        distance = np.sqrt(sum((item - self.current_value)**2))
        return distance <= self.maximum_travel + self.epsilon


class MaxSpeedPositionActionGenerator(MovePositionActionGenerator):
    """Action generator which generates actions uniformly within a circle around the current
    position. Circle radius is defined by the :attr:`max_speed` of the platform and the duration of
    the action. This generator is only applicable to 2D position actions."""

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

    max_speed: float = Property(
        default=1,
        doc="The maximum speed that the platform can move in m/s."
    )

    resolution: float = Property(
        default=1,
        doc="The interval in distance travelled for each action."
    )

    angle_resolution: float = Property(
        default=np.pi/2,
        doc="The interval in angle for each action."
    )

    epsilon = 1e-6

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.action_mapping) != 2:
            raise ValueError(f"Action mapping {self.action_mapping} does not have 2 dimensions."
                             ":class:`~.RangeAngleActionGenerator` is designed for 2D action "
                             "generation only.")

        if self.action_space is not None:
            if len(self.action_space) != len(self.action_mapping):
                raise ValueError(f"Dimensions of action_space {self.action_space.shape} are not "
                                 "compatible with action_mapping of length "
                                 f"{len(self.action_mapping)}. action_space should be of shape "
                                 "(ndim, 2) where ndim is the length of the action_mapping.")

            if (np.any(self.current_value[self.action_mapping, :] < self.action_space[:, [0]])
                    or np.any(self.current_value[self.action_mapping, :] > self.action_space[:, [1]])):  # noqa: E501
                raise ValueError(f"Initial platform location {self.current_value} is not within "
                                 f"the bounds of the action space {self.action_space}.")

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def maximum_travel(self):
        return self.duration.total_seconds() * self.max_speed

    @property
    def angle_deltas(self):
        return np.arange(0, 2*np.pi, self.angle_resolution)

    @property
    def range_deltas(self):
        return np.arange(0, self.maximum_travel + self.epsilon, self.resolution)

    def __contains__(self, item):
        if isinstance(item, MovePositionAction):
            item = item.target_value
        distance = np.sqrt(sum((item - self.current_value)**2))
        return distance <= self.maximum_travel + self.epsilon

    @property
    def default_action(self):
        return MovePositionAction(generator=self,
                                  end_time=self.end_time,
                                  target_value=self.current_value)

    def __iter__(self):
        yield MovePositionAction(generator=self,
                                 end_time=self.end_time,
                                 target_value=self.current_value)

        for angle_, range_ in product(self.angle_deltas, self.range_deltas):
            if range_ == 0:
                continue
            value = np.zeros(self.current_value.shape)
            value[self.action_mapping,] = np.array([[range_*np.cos(angle_)],
                                                    [range_*np.sin(angle_)]])
            target_value = self.current_value + value
            if self.action_space is None or \
                    (np.all(target_value[self.action_mapping, :] >= self.action_space[:, [0]])
                     and np.all(target_value[self.action_mapping, :] <= self.action_space[:, [1]])):  # noqa: E501
                yield MovePositionAction(generator=self,
                                         end_time=self.end_time,
                                         target_value=target_value)
