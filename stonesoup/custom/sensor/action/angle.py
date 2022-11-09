from typing import Iterator

import numpy as np

from stonesoup.base import Property
from stonesoup.custom.functions import get_nearest
from stonesoup.sensor.action import Action, RealNumberActionGenerator
from stonesoup.types.angle import Angle


class ChangeAngleAction(Action):
    """The action of changing the pan & tilt of sensors where `pan_tilt` is an
        :class:`~.ActionableProperty`"""

    increasing_angle: bool = Property(
        default=None, readonly=True,
        doc="Indicated the direction of change in the dwell centre angle. The first element "
            "relates to bearing, the second to elevation.")

    def act(self, current_time, timestamp, init_value):
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

        if timestamp >= self.end_time:
            return self.target_value  # target direction
        else:
            return init_value  # same direction


class ChangePanAction(ChangeAngleAction):
    pass


class ChangeTiltAction(ChangeAngleAction):
    pass


class AngleUAVActionsGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing the dwell centre of a sensor in a given
        time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "dwell-centre attributes")
    resolution: Angle = Property(default=np.radians(10), doc="Resolution of action space")

    _action_cls = ChangeAngleAction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default_action(self):
        return self._action_cls(generator=self,
                                end_time=self.end_time,
                                target_value=self.current_value,
                                increasing_angle=None)

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
        return self.current_value

    @property
    def min(self):
        # Pan can rotate freely, while tilt is limited to +/- 90 degrees
        return Angle(-np.pi / 2)

    @property
    def max(self):
        # Pan can rotate freely, while tilt is limited to +/- 90 degrees
        return Angle(np.pi / 2)

    def __contains__(self, item):

        if isinstance(item, self._action_cls):
            item = item.target_value

        return self.min <= item <= self.max

    def _get_direction(self, angle):
        angle = Angle(angle)

        if self.initial_value - self.resolution / 2 \
                <= angle \
                <= self.initial_value + self.resolution / 2:
            return None  # no rotation, target angle achieved

        return angle > self.initial_value

    def __iter__(self) -> Iterator[ChangeAngleAction]:
        """Returns all possible ChangePanTiltAction types"""
        possible_angles = np.arange(self.min, self.max, self.resolution)

        yield self.default_action
        for angle in possible_angles:
            increasing = self._get_direction(angle)
            yield self._action_cls(generator=self,
                                   end_time=self.end_time,
                                   target_value=angle,
                                   increasing_angle=increasing)

    def action_from_value(self, value):
        if value not in self:
            return None
        possible_angles = np.arange(self.min, self.max, self.resolution)
        angle = get_nearest(possible_angles, value)
        increasing = self._get_direction(angle)
        return self._action_cls(generator=self,
                                end_time=self.end_time,
                                target_value=angle,
                                increasing_angle=increasing)


class PanUAVActionsGenerator(AngleUAVActionsGenerator):
    _action_cls = ChangePanAction
    pass


class TiltUAVActionsGenerator(AngleUAVActionsGenerator):
    _action_cls = ChangeTiltAction
    pass

