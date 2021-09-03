# -*- coding: utf-8 -*-
import datetime
from copy import copy
from typing import Iterator

import numpy as np

from . import Action, RealNumberActionGenerator
from .action_utils import contains_bearing
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
        self.epsilon = 1e-6

    @property
    def default_action(self):
        return ChangeDwellAction(rotation_end_time=self.end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 target_value=self.initial_value,
                                 increasing_angle=True)

    def __call__(self, resolution=None, epsilon=1e-6):
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
        return self.current_value[0, 0]

    @property
    def initial_float_value(self):
        return float(self.initial_value)

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
        if self.angle_delta >= np.pi:
            return Bearing(-np.pi)
        else:
            return Bearing(self.initial_value - self.angle_delta)

    @property
    def max(self):
        if self.angle_delta >= np.pi:
            return Bearing(np.pi)
        else:
            return Bearing(self.initial_value + self.angle_delta)

    def __contains__(self, item):

        if isinstance(item, ChangeDwellAction):
            item = item.target_value

        if isinstance(item, (float, int)):
            item = Bearing(item)

        return contains_bearing(self.min, self.max, item)

    def _end_time_direction(self, bearing):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""

        bearing = float(bearing)

        if self.initial_float_value - self.epsilon \
                <= bearing \
                <= self.initial_float_value + self.epsilon:
            return self.start_time, None  # no rotation, target bearing achieved

        if self.initial_float_value < bearing:
            if bearing - self.initial_float_value <= np.radians(180):
                # increase bearing to reach target
                angle_delta = bearing - self.initial_float_value
                increasing = True
            else:
                # decrease bearing to reach target
                angle_delta = 2 * np.pi + self.initial_float_value - bearing
                increasing = False
        else:
            if self.initial_float_value - bearing <= np.radians(180):
                # decrease bearing to reach target
                angle_delta = self.initial_float_value - bearing
                increasing = False
            else:
                # increase angle to reach target
                angle_delta = 2 * np.pi + bearing - self.initial_float_value
                increasing = True

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps * 2 * np.pi)),
            increasing
        )

    def __iter__(self) -> Iterator[ChangeDwellAction]:
        """Returns ChangeDwellAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""

        current_bearing = float(self.min)

        if self.max < self.min:
            limit = float(self.max) + 2 * np.pi
        else:
            limit = float(self.max)

        while current_bearing <= limit + self.epsilon:
            rot_end_time, increasing = self._end_time_direction(current_bearing)
            yield ChangeDwellAction(rotation_end_time=rot_end_time,
                                    generator=self,
                                    end_time=self.end_time,
                                    target_value=Bearing(current_bearing),
                                    increasing_angle=increasing)
            current_bearing += float(self.resolution)

    def action_from_value(self, value):

        if isinstance(value, (int, float, Angle)):
            value = Bearing(value)
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
