import datetime
from collections.abc import Iterator
from copy import copy
from abc import abstractmethod

import numpy as np

from ...sensormanager.action import Action, RealNumberActionGenerator
from ...base import Property
from ...types.angle import Angle


class ChangeAngleAction(Action):
    """The base action for changing a sensor's :class:`~.ActionableProperty` where the property is
    described in terms of an angle."""

    rotation_end_time: datetime.datetime = Property(readonly=True,
                                                    doc="End time of rotation.")
    increasing_angle: bool = Property(default=None, readonly=True,
                                      doc="Indicates the direction of change in the angle.")

    def act(self, current_time, timestamp, init_value, **kwargs):
        """Assumes that duration keeps within the action end time

        Parameters
        ----------
        current_time: datetime.datetime
            Current time
        timestamp: datetime.datetime
            Modification of attribute ends at this time stamp
        init_value: Any
            Current value of the property

        Returns
        -------
        Any
            The new value of the property"""

        if self.increasing_angle is None:
            return init_value

        if current_time >= self.rotation_end_time:
            return init_value

        if timestamp <= self.rotation_end_time:
            # rotate for duration
            duration = timestamp - current_time
        else:
            # timestamp > rot end time
            # so rotate then stay
            duration = self.rotation_end_time - current_time

        # in case value is mutable
        actionable_value = np.asarray(copy(init_value), dtype=np.float64)

        angle_delta = duration.total_seconds() * self.generator.rps * 2 * np.pi
        if self.increasing_angle:
            actionable_value[0, 0] = actionable_value[0, 0] + angle_delta
        else:
            actionable_value[0, 0] = actionable_value[0, 0] - angle_delta

        return actionable_value


class AngleActionsGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing an actionable property of a sensor in a given
    time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "`resolution`.")
    resolution: Angle = Property(default=np.radians(1),
                                 doc="Resolution of the action space.")
    rpm: float = Property(default=60,
                          doc="The number of rotations per minute (RPM).")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = Angle(np.radians(1e-6))

    @property
    @abstractmethod
    def default_action(self):
        return NotImplementedError

    def __call__(self, resolution=None, epsilon=None):
        """
        Parameters
        ----------
        resolution : Angle
            Resolution of yielded action target values
        epsilon: float
            Tolerance of equality check in iteration
        """
        if resolution is not None:
            self.resolution = resolution
        if epsilon is not None:
            self.epsilon = epsilon

    @property
    def initial_value(self):
        return Angle(self.current_value[0, 0])

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def rps(self):
        return self.rpm / 60

    @property
    def angle_delta(self):
        return Angle(self.duration.total_seconds() * self.rps * 2 * np.pi)

    @property
    def min(self):
        return Angle(self.initial_value - self.angle_delta)

    @property
    def max(self):
        return Angle(self.initial_value + self.angle_delta)

    def __contains__(self, item):

        if self.angle_delta >= np.pi:
            # Left turn and right turn are > 180, so all angles hit
            return True

        if isinstance(item, ChangeAngleAction):
            item = item.target_value

        if isinstance(item, (float, int)):
            item = Angle(item)

        return self.min <= item <= self.max

    def _end_time_direction(self, angle):
        """Given a target bearing, should the property rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        angle = Angle(angle)

        if self.initial_value - self.epsilon \
                <= angle \
                <= self.initial_value + self.epsilon:
            return self.start_time, None  # no rotation, target bearing achieved

        angle_delta = np.abs(angle - self.initial_value)

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps * 2 * np.pi)),
            angle > self.initial_value
        )

    @abstractmethod
    def __iter__(self) -> Iterator[ChangeAngleAction]:
        """Returns ChangeAngleAction types, where the value is a possible value of the [0, 0]
        element of the property's state vector."""
        raise NotImplementedError

    def action_from_value(self, value):
        """Given a value for the property, what action would achieve that value.

        Parameters
        ----------
        value: Any
            Property value for which the action is required.

        Returns
        -------
        ChangeAngleAction
            Action which will achieve this property.
        """

        if isinstance(value, (int, float)):
            value = Angle(value)
        if not isinstance(value, Angle):
            raise ValueError("Can only generate action from an Angle/float/int type")

        if value not in self:
            return None

        # Find the number of resolutions that fit between initial value and target
        n = (value - self.initial_value) / self.resolution
        if np.isclose(n, round(n), 1e-6):
            n = round(n)
        else:
            n = int(n)

        target_value = self.initial_value + self.resolution * n

        rot_end_time, increasing = self._end_time_direction(target_value)

        return ChangeAngleAction(rotation_end_time=rot_end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 target_value=target_value,
                                 increasing_angle=increasing)
