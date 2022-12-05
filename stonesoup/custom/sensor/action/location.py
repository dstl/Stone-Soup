import itertools

import numpy as np
from typing import Iterator

from stonesoup.custom.functions import get_nearest
from stonesoup.types.array import StateVector

from stonesoup.base import Property

from stonesoup.sensor.action import Action, RealNumberActionGenerator


class ChangeLocationAction(Action):
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


class LocationActionGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing the dwell centre of a sensor in a given
        time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "dwell-centre attributes")
    resolution: float = Property(default=10, doc="Resolution of action space")
    minmax: StateVector = Property(doc="Min and max values of the action space",
                                   default=StateVector([-100, 100]))

    _action_cls = ChangeLocationAction

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default_action(self):
        return self._action_cls(generator=self,
                                end_time=self.end_time,
                                target_value=self.current_value)

    def __call__(self, resolution=None, epsilon=None):
        """
        Parameters
        ----------
        resolution : float
            Resolution of yielded action target values
        epsilon: float
            Epsilon value for action target values

        Returns
        -------
        :class:`.Action`
            Action with target value
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
        return self.minmax[0]

    @property
    def max(self):
        # Pan can rotate freely, while tilt is limited to +/- 90 degrees
        return self.minmax[1]

    def __contains__(self, item):

        if isinstance(item, self._action_cls):
            item = item.target_value

        return self.min <= item <= self.max

    def __iter__(self) -> Iterator[ChangeLocationAction]:
        """Returns all possible ChangePanTiltAction types"""
        possible_values = np.arange(self.min, self.max, self.resolution, dtype=float)

        yield self.default_action
        for angle in possible_values:
            yield self._action_cls(generator=self,
                                   end_time=self.end_time,
                                   target_value=angle)

    def action_from_value(self, value):
        if value not in self:
            return None
        possible_values = np.arange(self.min, self.max, self.resolution, dtype=float)
        angle = get_nearest(possible_values, value)
        return self._action_cls(generator=self,
                                end_time=self.end_time,
                                target_value=angle)