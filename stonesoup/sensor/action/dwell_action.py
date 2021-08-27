# -*- coding: utf-8 -*-
import datetime
from typing import Iterator
from copy import copy
import numpy as np

from . import Action, RealNumberActionGenerator
from ...base import Property
from ...functions import mod_bearing
from ...types.angle import Angle
from .action_utils import contains_angle


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
    attribute: str = Property()
    start_time: datetime.datetime = Property()
    end_time: datetime.datetime = Property()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = Angle(np.radians(1))

    @property
    def default_action(self):
        return ChangeDwellAction(rotation_end_time=self.end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 increasing_angle=True)

    def __call__(self, resolution=None):
        if resolution is not None:
            self.resolution = resolution

    @property
    def initial_value(self):
        return self.current_value[0, 0]

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
            return Angle(-np.pi)
        else:
            return Angle(self.initial_value - self.angle_delta)

    @property
    def max(self):
        if self.angle_delta >= np.pi:
            return Angle(np.pi)
        else:
            return Angle(self.initial_value + self.angle_delta)

    def __contains__(self, item):

        if isinstance(item, ChangeDwellAction):
            item = item.value

        if isinstance(item, (float, int)):
            item = Angle(item)

        left, right = Angle(self.min), Angle(self.max)

        return contains_angle(left, right, item)

    def _get_end_time_direction(self, bearing):
        """Given a target bearing, should the dwell centre rotate so as to increase its angle
        value, or decrease? And how long until it reaches the target."""
        if self.initial_value <= bearing:
            if bearing - self.initial_value < self.initial_value + 2 * np.pi - bearing:
                angle_delta = bearing - self.initial_value
                increasing = True
            else:
                angle_delta = self.initial_value + 2 * np.pi - bearing
                increasing = False
        else:
            if self.initial_value - bearing < bearing + 2 * np.pi - self.initial_value:
                angle_delta = self.initial_value - bearing
                increasing = False
            else:
                angle_delta = bearing + 2 * np.pi - self.initial_value
                increasing = True

        return (
            self.start_time + datetime.timedelta(seconds=angle_delta / (self.rps * 2 * np.pi)),
            increasing
        )

    def __iter__(self) -> Iterator[ChangeDwellAction]:
        """Returns ChangeDwellAction types, where the value is a possible value of the [0, 0]
        element of the dwell centre's state vector."""
        current_bearing = self.min
        while current_bearing <= self.max:
            rot_end_time, increasing = self._get_end_time_direction(current_bearing)
            yield ChangeDwellAction(rotation_end_time=rot_end_time,
                                    generator=self,
                                    end_time=self.end_time,
                                    increasing_angle=increasing)
            current_bearing += self.resolution

    def action_from_value(self, value):

        if isinstance(value, (int, float)):
            value = Angle(value)
        if not isinstance(value, Angle):
            raise ValueError("Can only generate action from an Angle/float/int type")

        if value not in self:
            return None  # Should this raise an error?

        value -= value % self.resolution

        rot_end_time, increasing = self._get_end_time_direction(value)

        return ChangeDwellAction(rotation_end_time=rot_end_time,
                                 generator=self,
                                 end_time=self.end_time,
                                 increasing_angle=increasing)
