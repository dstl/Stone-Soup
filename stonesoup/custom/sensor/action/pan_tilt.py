import datetime
from itertools import product
from typing import Sequence, Iterator

import numpy as np

from stonesoup.base import Property
from stonesoup.custom.functions import get_nearest
from stonesoup.sensor.action import Action, RealNumberActionGenerator
from stonesoup.types.angle import Angle, Elevation, Bearing
from stonesoup.types.array import StateVector


class ChangePanTiltAction(Action):
    """The action of changing the pan & tilt of sensors where `pan_tilt` is an
    :class:`~.ActionableProperty`"""

    increasing_angle: Sequence[bool] = Property(
        default=[None, None], readonly=True,
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


class PanTiltActionsGenerator(RealNumberActionGenerator):
    """Generates possible actions for changing the dwell centre of a sensor in a given
    time period."""

    owner: object = Property(doc="Object with `timestamp`, `rpm` (revolutions per minute) and "
                                 "dwell-centre attributes")
    resolution: Angle = Property(default=np.radians(1), doc="Resolution of action space")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def default_action(self):
        return ChangePanTiltAction(generator=self,
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
        return np.array([Angle(-2*np.pi), Angle(-np.pi/2)])

    @property
    def max(self):
        # Pan can rotate freely, while tilt is limited to +/- 90 degrees
        return np.array([Angle(2*np.pi), Angle(np.pi/2)])

    def __contains__(self, item):

        if isinstance(item, ChangePanTiltAction):
            item = item.target_value

        return self.min[0] <= item[0] <= self.max[0] and self.min[1] <= item[1] <= self.max[1]

    def _get_direction(self, angle, idx):
        angle = Angle(angle)

        if self.initial_value[idx] - self.resolution/2 \
                <= angle \
                <= self.initial_value[idx] + self.resolution/2:
            return None  # no rotation, target angle achieved

        return angle > self.initial_value[idx]

    def __iter__(self) -> Iterator[ChangePanTiltAction]:
        """Returns all possible ChangePanTiltAction types"""
        possible_pan_angles = np.arange(self.min[0], self.max[0], self.resolution)
        possible_tilt_angles = np.arange(self.min[1], self.max[1], self.resolution)
        for (pan_angle, tilt_angle) in product(possible_pan_angles, possible_tilt_angles):
            increasing_p = self._get_direction(pan_angle, 0)
            increasing_t = self._get_direction(tilt_angle, 1)
            yield ChangePanTiltAction(generator=self,
                                      end_time=self.end_time,
                                      target_value=StateVector([Angle(pan_angle),
                                                                Angle(tilt_angle)]),
                                      increasing_angle=[increasing_p, increasing_t])

    def action_from_value(self, value):
        if value not in self:
            return None
        pan_angle = Angle(value[0])
        tilt_angle = Angle(value[1])
        possible_pan_angles = np.arange(self.min[0], self.max[0], self.resolution)
        possible_tilt_angles = np.arange(self.min[1], self.max[1], self.resolution)
        pan_angle = get_nearest(possible_pan_angles, pan_angle)
        tilt_angle = get_nearest(possible_tilt_angles, tilt_angle)
        increasing_p = self._get_direction(pan_angle, 0)
        increasing_t = self._get_direction(tilt_angle, 1)
        return ChangePanTiltAction(generator=self,
                                   end_time=self.end_time,
                                   target_value=StateVector([pan_angle, tilt_angle]),
                                   increasing_angle=[increasing_p, increasing_t])


class PanTiltUAVActionsGenerator(PanTiltActionsGenerator):

    @property
    def min(self):
        # Pan and tilt are limited to +/- 90 degrees
        return np.array([Angle(-np.pi / 2), Angle(-np.pi / 2)])

    @property
    def max(self):
        # Pan and tilt are limited to +/- 90 degrees
        return np.array([Angle(np.pi / 2), Angle(np.pi / 2)])