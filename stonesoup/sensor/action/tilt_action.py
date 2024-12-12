import datetime
from copy import copy
from typing import Iterator

import numpy as np

from stonesoup.sensormanager.action import Action, RealNumberActionGenerator
from stonesoup.base import Property
from stonesoup.types.angle import Angle, Bearing
from stonesoup.functions import mod_bearing


class ChangeTiltAction(Action):
    """The action of changing the tilt centre of sensors where `tilt_centre` is an
    :class:`~.ActionableProperty`"""

    rotation_end_time: datetime.datetime = Property(readonly=True,
                                                    doc="End time of rotation.")
    increasing_angle: bool = Property(default=None, readonly=True,
                                      doc="Indicates the direction of change in the "
                                          "tilt centre angle.")

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

        if current_time >= self.rotation_end_time:
            return init_value

        if timestamp <= self.rotation_end_time:
            # rotate for duration
            duration = timestamp - current_time
        else:
            # timestamp > rot end time
            # so rotate then stay
            duration = self.rotation_end_time - current_time

        tilt_centre = np.asarray(copy(init_value), dtype=np.float64)  # in case value is mutable

        angle_delta = duration.total_seconds() * self.generator.rps * 2 * np.pi

        if self.increasing_angle is None:
            return init_value
        elif self.increasing_angle:
            tilt_centre[0, 0] = mod_bearing(tilt_centre[0, 0] + angle_delta)
        else:
            tilt_centre[0, 0] = mod_bearing(tilt_centre[0, 0] - angle_delta)

        return tilt_centre


class TiltActionsGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing the tilt centre of a sensor in a given
    time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "`resolution`.")
    resolution: Angle = Property(default=np.radians(1),
                                 doc="Resolution of the action space.")
    rpm: float = Property(default=60,
                          doc="The number of rotations per minute (RPM).")
    max_tilt: float = Property(default=np.radians(90))
    min_tilt: float = Property(default=np.radians(-90))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = Angle(np.radians(1e-6))

    @property
    def default_action(self):
        return ChangeTiltAction(rotation_end_time=self.end_time,
                                generator=self,
                                end_time=self.end_time,
                                target_value=self.initial_value,
                                increasing_angle=None)

    def __call__(self, resolution=None, epsilon=None):
        """
        Parameters
        ----------
        resolution : Angle
            Resolution of yielded action target values.
        epsilon : float
            Tolerance of equality check in iteration.
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
        return max(Angle(self.initial_value - self.angle_delta), self.min_tilt)

    @property
    def max(self):
        return min(Angle(self.initial_value + self.angle_delta), self.max_tilt)

    def __contains__(self, item):

        if self.angle_delta >= np.pi:
            # Left turn and right turn are > 180, so all angles hit
            return True

        if isinstance(item, ChangeTiltAction):
            item = item.target_value

        if isinstance(item, (float, int)):
            item = Angle(item)

        return self.min <= item <= self.max

    def _end_time_direction(self, angle):
        """Given a target bearing, should the tilt centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        angle = Angle(angle)
        angle_delta = np.abs(angle - self.initial_value)

        if angle - self.initial_value > 0:
            increasing = True
        elif angle - self.initial_value < 0:
            increasing = False
        else:
            increasing = None

        if self.rps == 0:
            return self.start_time + datetime.timedelta(seconds=0), None
        else:
            return (
                self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps * 2 * np.pi)),
                increasing)

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
                                   target_value=Bearing(current_angle),
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

        return ChangeTiltAction(rotation_end_time=rot_end_time,
                                generator=self,
                                end_time=self.end_time,
                                target_value=target_value,
                                increasing_angle=increasing)
