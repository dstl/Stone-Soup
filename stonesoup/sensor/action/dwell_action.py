from collections.abc import Iterator

from ...types.angle import Bearing
from ...functions import mod_bearing
from .base import ChangeAngleAction, AngleActionsGenerator


class ChangeDwellAction(ChangeAngleAction):
    """The action of changing the dwell centre of sensors where `dwell_centre` is an
    :class:`~.ActionableProperty`"""

    def act(self, current_time, timestamp, init_value, **kwargs):
        """Assumes that duration keeps within the action end time

        Parameters
        ----------
        current_time: datetime.datetime
            Current time
        timestamp: datetime.datetime
            Modification of attribute ends at this time stamp
        init_value: Any
            Current value of the dwell centre

        Returns
        -------
        Any
            The new value of the dwell centre"""

        dwell_centre = super().act(current_time, timestamp, init_value, **kwargs)
        dwell_centre[0, 0] = mod_bearing(dwell_centre[0, 0])
        return dwell_centre


class DwellActionsGenerator(AngleActionsGenerator):
    """Generates possible actions for changing the dwell centre of a sensor in a given
    time period."""

    @property
    def default_action(self):
        return ChangeDwellAction(rotation_end_time=self.end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 target_value=self.initial_value,
                                 increasing_angle=True)

    def __iter__(self) -> Iterator[ChangeDwellAction]:
        """Returns ChangeDwellAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""

        current_angle = self.min

        while current_angle <= self.max + self.epsilon:
            rot_end_time, increasing = self._end_time_direction(current_angle)
            yield ChangeDwellAction(rotation_end_time=rot_end_time,
                                    generator=self,
                                    end_time=self.end_time,
                                    target_value=Bearing(current_angle),
                                    increasing_angle=increasing)
            current_angle += self.resolution

    def action_from_value(self, value):
        """Given a value for dwell centre, what action would achieve that dwell centre
        value.

        Parameters
        ----------
        value: Any
            Dwell centre value for which the action is required.

        Returns
        -------
        ChangeDwellAction
            Action which will achieve this dwell centre.
        """
        angle_action = super().action_from_value(value)
        if angle_action is None:
            return angle_action
        return ChangeDwellAction(rotation_end_time=angle_action.rotation_end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 target_value=angle_action.target_value,
                                 increasing_angle=angle_action.increasing_angle)
