from collections.abc import Iterator

import numpy as np

from ...base import Property
from ...types.angle import Angle, Elevation
from ...functions import mod_elevation
from .base import ChangeAngleAction, AngleActionsGenerator


class ChangeTiltAction(ChangeAngleAction):
    """The action of changing the tilt centre of sensors where `tilt_centre` is an
    :class:`~.ActionableProperty`"""

    def act(self, current_time, timestamp, init_value, **kwargs):
        """Assumes that duration keeps within the action end time.

        Parameters
        ----------
        current_time : datetime.datetime
            Current time.
        timestamp : datetime.datetime
            Modification of attribute ends at this time stamp.
        init_value : Any
            Current value of the tilt centre.

        Returns
        -------
        Any
            The new value of the tilt centre."""

        tilt_centre = super().act(current_time, timestamp, init_value, **kwargs)
        tilt_centre[0, 0] = mod_elevation(tilt_centre[0, 0])
        return tilt_centre


class TiltActionsGenerator(AngleActionsGenerator):
    """Generates possible actions for changing the tilt centre of a sensor in a given
    time period."""

    max_tilt: float = Property(default=np.radians(90))
    min_tilt: float = Property(default=np.radians(-90))

    @property
    def default_action(self):
        return ChangeTiltAction(rotation_end_time=self.end_time,
                                generator=self,
                                end_time=self.end_time,
                                target_value=self.initial_value,
                                increasing_angle=None)

    @property
    def min(self):
        return max(Angle(self.initial_value - self.angle_delta), self.min_tilt)

    @property
    def max(self):
        return min(Angle(self.initial_value + self.angle_delta), self.max_tilt)

    def __iter__(self) -> Iterator[ChangeTiltAction]:
        """Returns ChangeTiltAction types, where the value is a possible value of the [0, 0]
        element of the tilt centre's state vector."""

        current_angle = self.initial_value
        while current_angle - self.resolution >= max(self.min_tilt, self.min-self.epsilon):
            current_angle -= self.resolution
        while current_angle <= self.max + self.epsilon:
            rot_end_time, increasing = self._end_time_direction(current_angle)
            yield ChangeTiltAction(rotation_end_time=rot_end_time,
                                   generator=self,
                                   end_time=self.end_time,
                                   target_value=Elevation(current_angle),
                                   increasing_angle=increasing)
            current_angle += self.resolution

    def action_from_value(self, value):
        """Given a value for tilt centre, what action would achieve that tilt centre
        value.

        Parameters
        ----------
        value: Any
            Tilt centre value for which the action is required.

        Returns
        -------
        ChangeTiltAction
            Action which will achieve this dwell centre.
        """

        angle_action = super().action_from_value(value)
        if angle_action is None:
            return angle_action
        return ChangeTiltAction(rotation_end_time=angle_action.rotation_end_time,
                                generator=self,
                                end_time=self.end_time,
                                target_value=angle_action.target_value,
                                increasing_angle=angle_action.increasing_angle)
