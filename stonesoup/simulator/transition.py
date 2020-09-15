# -*- coding: utf-8 -*-
from datetime import timedelta
from copy import deepcopy

import numpy as np

from stonesoup.models.transition.linear import ConstantTurn, ConstantVelocity,\
    CombinedLinearGaussianTransitionModel
from stonesoup.types.array import StateVector


def create_smooth_transition_models(initial_state, x_coords, y_coords, times, turn_rate):
    """Generate a list of constant-turn and constant acceleration transition models alongside a
    list of transition times to provide smooth transitions between 2D cartesian coordinates and
    time pairs.
    An assumption is that the initial_state's x, y coordinates are the first elements of x_ccords
    and y_coords respectively. Ie. The platform starts at the first coordinates.

    Parameters
    ----------
    initial_state: :class:`~.State` The initial state of the platform.
    x_coords:
        A list of int/float x-coordinates (cartesian) in the order that the target must follow.
    y_coords:
        A list of int/float y-coordinates (cartesian) in the order that the target must follow.
    times:
        A list of :class:`~.datetime.datetime` dictating the times at which the target must be at
        each corresponding coordinate.
    turn_rate: Float
        Angular turn rate (radians/second) measured anti-clockwise from positive x-axis.

    Returns
    -------
    transition_models:
        A list of :class:`~.ConstantTurn` and :class:`~.Point2PointConstantAcceleration` transition
        models.
    transition_times:
        A list of :class:`~.datetime.timedelta` dictating the transition time for each
        corresponding transition model in transition_models.

    Notes
    -----
    x_coords, y_coords and times must be of same length.
    This method assumes a cartesian state space with velocities eg. (x, vx, y, vy). It returns
    transition models for 2 cartesian coordinates and their corresponding velocities.
    """

    state = deepcopy(initial_state)  # don't alter platform state with calculations

    if not len(x_coords) == len(y_coords) == len(times):
        raise ValueError('x_coords, y_coords and times must be same length')

    transition_models = []
    transition_times = []

    for x_coord, y_coord, time in zip(x_coords[1:], y_coords[1:], times[1:]):

        dx = x_coord - state.state_vector[0]  # distance to next x-coord
        dy = y_coord - state.state_vector[2]  # distance to next y-coord

        if dx == 0 and dy == 0:
            a = 0  # if initial and second target coordinates are same, set arbitrary bearing of 0

        vx = state.state_vector[1]  # initial x-speed
        vy = state.state_vector[3]  # initial y-speed

        if vx != 0 or vy != 0:  # if velocity is 0, keep previous bearing
            a = np.arctan2(vy, vx)  # initial bearing

        if dx == 0 and dy == 0 and vx == 0 and vy == 0:  # if at destination with 0 speed, stay
            transition_times.append(
                timedelta(seconds=(time - times[times.index(time) - 1]).total_seconds()))
            transition_models.append(CombinedLinearGaussianTransitionModel((ConstantVelocity(0),
                                                                            ConstantVelocity(0))))
            continue

        d = np.sqrt(dx**2 + dy**2)  # distance to next coord

        v = np.sqrt(vx**2 + vy**2)  # initial speed

        b = np.arctan2(dy, dx) - a  # bearing to next coord (anti-clockwise from positive x-axis)

        w = turn_rate  # turn rate (anti-clockwise from positive x-axis)

        if b > np.radians(180):
            b -= 2*np.pi  # get bearing in (0, 180) instead
        elif b <= np.radians(-180):
            b += 2*np.pi  # get bearing in (-180, 0] instead

        if b < 0:
            w = -w  # if bearing is in [-180, 0), turn right instead

        r = v / np.abs(w)  # radius of turn

        if b >= 0:
            p = d * np.cos(b)
            q = r - d*np.sin(b)
        else:
            p = -d*np.cos(b)
            q = r + d*np.sin(b)

        alpha = np.arctan2(p, q)
        beta = np.arccos(r / np.sqrt(p**2 + q**2))

        angle = (alpha + beta) % (2*np.pi) - 2*np.pi  # actual angle turned

        if w > 0:
            angle = (alpha - beta + 2*np.pi) % (2*np.pi)  # quadrant adjustment

        t1 = angle / w  # turn time

        if t1 > 0:
            # make turn model and add to list
            turn_model = ConstantTurn(turn_noise_diff_coeffs=(0, 0), turn_rate=w)
            state.state_vector = turn_model.function(state=state, time_interval=timedelta(
                seconds=t1))  # move platform through turn
            state.timestamp += timedelta(seconds=t1)
            transition_times.append(timedelta(seconds=t1))
            transition_models.append(turn_model)

        dx = x_coord - state.state_vector[0]  # get remaining distance to next x-coord
        dy = y_coord - state.state_vector[2]  # get remaining distance to next y-coord

        d = np.sqrt(dx**2 + dy**2)  # remaining distance to next coord

        t2 = (time - state.timestamp).total_seconds()  # time remaining before platform should
        # be at next coord

        if d > 0:  # if platform is not already at target coord, add linear acceleration model

            try:
                accel_model = Point2PointConstantAcceleration(state=state,
                                                              destination=(x_coord, y_coord),
                                                              duration=timedelta(seconds=t2))
            except OvershootError:
                # if linear accel leads to overshoot, apply model to stop at target coord instead
                accel_model = Point2PointStop(state=state,
                                              destination=(x_coord, y_coord),
                                              duration=timedelta(seconds=t2))
            state.state_vector = accel_model.function(state=state,
                                                      time_interval=timedelta(seconds=t2))
            state.timestamp += timedelta(seconds=t2)
            transition_times.append(timedelta(seconds=t2))
            transition_models.append(accel_model)

    return transition_models, transition_times


class OvershootError(Exception):
    pass


class Point2PointConstantAcceleration:
    r"""Constant acceleration transition model for 2D cartesian coordinates

    The platform is assumed to move with constant acceleration between two given cartesian
    coordinates.
    Motion is determined by the kinematic formulae:

        .. math::
            v &= u + at \\
            s &= ut + \frac{1}{2} at^2

    Where :math:`u, v, a, t, s` are initial speed, final speed, acceleration, transition time and
    distance travelled respectively.
    """

    def __init__(self, state, destination, duration):
        dx = destination[0] - state.state_vector[0]  # x-distance to destination
        dy = destination[1] - state.state_vector[2]  # y-distance to destination
        ux = state.state_vector[1]  # initial x-speed
        uy = state.state_vector[3]  # initial y-speed

        t = duration.total_seconds()  # duration of acceleration

        self.ax = 2*(dx - ux*t) / t**2  # x-acceleration
        self.ay = 2*(dy - uy*t) / t**2  # y-acceleration

        vx = ux + self.ax*t  # final x-speed
        vy = uy + self.ay*t  # final y-speed

        if np.sign(ux) != np.sign(vx) or np.sign(uy) != np.sign(vy):
            raise OvershootError()

    @property
    def ndim_state(self):
        return 4

    def covar(self, **kwargs):
        raise NotImplementedError('Covariance not defined')

    def function(self, state, time_interval, **kwargs):

        x = state.state_vector[0]
        y = state.state_vector[2]

        t = time_interval.total_seconds()
        ux = state.state_vector[1]  # initial x-speed
        uy = state.state_vector[3]  # initial y-speed

        dx = ux*t + 0.5*self.ax*(t**2)  # x-distance travelled
        dy = uy*t + 0.5*self.ay*(t**2)  # y-distance travelled
        vx = ux + self.ax*t  # resultant x-speed
        vy = uy + self.ay*t  # resultant y-speed

        return StateVector([x+dx, vx, y+dy, vy])


class Point2PointStop:
    r"""Constant acceleration transition model for 2D cartesian coordinates

    The platform is assumed to move with constant acceleration between two given cartesian
    coordinates.
    Motion is determined by the kinematic formulae:

        .. math::
            v &= u + at \\
            v^2 &= u^2 + 2as

    Where :math:`u, v, a, t, s` are initial speed, final speed, acceleration, transition time and
    distance travelled respectively.
    The platform is decelerated to 0 velocity at the destination point and waits for the remaining
    duration.
    """

    @property
    def ndim_state(self):
        return 4

    def covar(self, **kwargs):
        raise NotImplementedError('Covariance not defined')

    def __init__(self, state, destination, duration):
        dx = destination[0] - state.state_vector[0]  # x-distance to destination
        dy = destination[1] - state.state_vector[2]  # y-distance to destination
        ux = state.state_vector[1]  # initial x-speed
        uy = state.state_vector[3]  # initial y-speed

        self.destination = destination

        if dx == 0:
            self.ax = 0  # x-acceleration (0 if already at destination x-coord)
        else:
            self.ax = -(ux**2) / (2*dx)
        if dy == 0:
            self.ay = 0  # y-acceleration (0 if already at destination y-coord)
        else:
            self.ay = -(uy**2) / (2*dy)

        if self.ax != 0:
            self.t = -ux / self.ax  # deceleration time
        elif self.ay != 0:
            self.t = -uy / self.ay  # deceleration time (if already at x-coord)
        else:
            self.t = 0  # at destination so acceleration time is 0

        self.duration = duration.total_seconds()  # full transition duration
        self.start_time = state.timestamp

    def function(self, state, time_interval, **kwargs):

        t = time_interval.total_seconds()

        decel_time_remaining = self.t - (state.timestamp - self.start_time).total_seconds()

        x = state.state_vector[0]
        y = state.state_vector[2]
        ux = state.state_vector[1]  # initial x-speed
        uy = state.state_vector[3]  # initial y-speed

        if t < decel_time_remaining:  # still some deceleration needed
            dx = ux*t + (0.5*self.ax)*t**2
            dy = uy*t + (0.5*self.ay)*t**2
            vx = ux + self.ax*t
            vy = uy + self.ay*t
            return StateVector([x + dx, vx, y + dy, vy])
        elif decel_time_remaining > 0:  # otherwise decelerate for rest of time needed, and stay
            dx = ux*decel_time_remaining + (0.5*self.ax)*(decel_time_remaining**2)
            dy = uy*decel_time_remaining + (0.5*self.ay)*(decel_time_remaining**2)
            vx = ux + self.ax*decel_time_remaining
            vy = uy + self.ay*decel_time_remaining
            return StateVector([x + dx, vx, y + dy, vy])
        else:
            return state.state_vector  # if already at destination, stay
