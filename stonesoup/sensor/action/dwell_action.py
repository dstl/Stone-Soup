# -*- coding: utf-8 -*-
import datetime
from copy import copy
from typing import Iterator

import numpy as np

from . import Action, RealNumberActionGenerator
from ...base import Property
from ...functions import mod_bearing
from ...types.angle import Angle, Bearing


class ChangeDwellAction(Action):
    rotation_end_time: datetime.datetime = Property(readonly=True)
    increasing_angle: bool = Property(default=None, readonly=True)

    def act(self, current_time, timestamp, init_value):
        """Assumes that duration keeps within the action end time."""

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

        dwell_centre = np.asfarray(copy(init_value))  # in case value is mutable

        angle_delta = duration.total_seconds() * self.generator.rps * 2 * np.pi
        if self.increasing_angle:
            dwell_centre[0, 0] = mod_bearing(dwell_centre[0, 0] + angle_delta)
        else:
            dwell_centre[0, 0] = mod_bearing(dwell_centre[0, 0] - angle_delta)

        return dwell_centre


class DwellActionsGenerator(RealNumberActionGenerator):
    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "dwell-centre attributes")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = Angle(np.radians(1))
        self.epsilon = Angle(np.radians(1e-6))

    @property
    def default_action(self):
        return ChangeDwellAction(rotation_end_time=self.end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 target_value=self.initial_value,
                                 increasing_angle=True)

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
        return self.owner.rpm / 60

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

        if isinstance(item, ChangeDwellAction):
            item = item.target_value

        if isinstance(item, (float, int)):
            item = Angle(item)

        return self.min <= item <= self.max

    def _end_time_direction(self, angle):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
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

        if isinstance(value, (int, float, Angle)):
            value = Angle(value)
        else:
            raise ValueError("Can only generate action from an Angle/float/int type")

        if value not in self:
            return None  # Should this raise an error?

        value -= value % self.resolution

        rot_end_time, increasing = self._end_time_direction(value)

        return ChangeDwellAction(rotation_end_time=rot_end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 target_value=value,
                                 increasing_angle=increasing)
