# -*- coding: utf-8 -*-
from copy import deepcopy
from datetime import timedelta
from itertools import combinations
from typing import Tuple, Sequence

import numpy as np

from ..base import Property
from ..models.transition.base import TransitionModel
from ..models.transition.linear import ConstantTurn, ConstantVelocity, \
    CombinedLinearGaussianTransitionModel
from ..types.array import StateVector
from ..types.state import State


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
            transition_times.append(time - times[times.index(time) - 1])
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

        angle = (alpha + beta + np.pi) % (2*np.pi) - np.pi  # actual angle turned

        if w > 0:
            angle = (alpha - beta + np.pi) % (2*np.pi) - np.pi  # quadrant adjustment

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
                accel_model = Point2PointConstantAcceleration(state=deepcopy(state),
                                                              destination=(x_coord, y_coord),
                                                              duration=timedelta(seconds=t2))
            except OvershootError:
                # if linear accel leads to overshoot, apply model to stop at target coord instead
                accel_model = Point2PointStop(state=deepcopy(state),
                                              destination=(x_coord, y_coord))
            state.state_vector = accel_model.function(state=state,
                                                      time_interval=timedelta(seconds=t2))
            state.timestamp += timedelta(seconds=t2)
            transition_times.append(timedelta(seconds=t2))
            transition_models.append(accel_model)

    return transition_models, transition_times


class OvershootError(Exception):
    pass


class Point2PointConstantAcceleration(TransitionModel):
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

    state: State = Property(doc="The initial state, assumed to have x and y cartesian position and"
                                "velocities")
    destination: Tuple[float, float] = Property(doc="Destination coordinates in 2D cartesian"
                                                    "coordinates (x, y)")
    duration: timedelta = Property(doc="Duration of transition in seconds")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dx = self.destination[0] - self.state.state_vector[0]  # x-distance to destination
        dy = self.destination[1] - self.state.state_vector[2]  # y-distance to destination
        ux = self.state.state_vector[1]  # initial x-speed
        uy = self.state.state_vector[3]  # initial y-speed

        t = self.duration.total_seconds()  # duration of acceleration

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

    def pdf(self, state1, state2, **kwargs):
        raise NotImplementedError('pdf not defined')

    def rvs(self, num_samples=1, **kwargs):
        raise NotImplementedError('rvs not defined')

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


class Point2PointStop(TransitionModel):
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

    state: State = Property(doc="The initial state, assumed to have x and y cartesian position and"
                                "velocities")
    destination: Tuple[float, float] = Property(doc="Destination coordinates in 2D cartesian"
                                                    "coordinates (x, y)")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dx = self.destination[0] - self.state.state_vector[0]  # x-distance to destination
        dy = self.destination[1] - self.state.state_vector[2]  # y-distance to destination
        ux = self.state.state_vector[1]  # initial x-speed
        uy = self.state.state_vector[3]  # initial y-speed

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

        self.start_time = self.state.timestamp

    @property
    def ndim_state(self):
        return 4

    def covar(self, **kwargs):
        raise NotImplementedError('Covariance not defined')

    def pdf(self, state1, state2, **kwargs):
        raise NotImplementedError('pdf not defined')

    def rvs(self, num_samples=1, **kwargs):
        raise NotImplementedError('rvs not defined')

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


class ConstantJerkSimulator(TransitionModel):
    r"""Constant, noiseless, jerk transition model for cartesian space.

    The state space has no acceleration or jerk elements. I.E. the only kinematic components are
    position and velocity.

    Solution given by :math:`\vec{\ddddot{x}} = \vec{0}`

    The user will provide an initial and final state, each of which containing initial cartesian
    position and velocities. For example, :math:`(x, vx, y, vy)`.

    Components of the state vector that are not position or velocity are kept constant.
    Initial and final accelerations are uniquely defined by this input.

    Notes
    -----
        * Acceleration instantaneously changes at each target state
    """
    position_mapping: Sequence[int] = Property(
        doc="Mapping between platform position and state vector.")
    velocity_mapping: Sequence[int] = Property(
        default=None,
        doc="Mapping between platform velocity and state vector. Defaults to `[m+1 for m in "
            "position_mapping]`")
    init_state: State = Property(
        doc="Initial state to move from. Must be `ndim_state` dimensions.")
    final_state: State = Property(
        doc="Final state to move to. Must be `ndim_state` dimensions.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.init_state.state_vector.shape[0] != self.final_state.state_vector.shape[0]:
            raise ValueError(
                f"Initial and final states must share the same number of dimensions. Initial "
                f"state has ndim = {self.init_state.state_vector.shape[0]} but final state has "
                f"ndim = {self.final_state.state_vector.shape[0]}")

        if self.velocity_mapping is None:
            self.velocity_mapping = [p + 1 for p in self.position_mapping]

        # Full duration of transition
        self.duration = (self.final_state.timestamp - self.init_state.timestamp).total_seconds()

        # Initial position
        self.init_X = self.init_state.state_vector[self.position_mapping, :]
        # Initial velocity
        self.init_V = self.init_state.state_vector[self.velocity_mapping, :]

        # Final position
        self.final_X = self.final_state.state_vector[self.position_mapping, :]
        # Final velocity
        self.final_V = self.final_state.state_vector[self.velocity_mapping, :]

        self.init_A, self.final_A, self.jerk = list(), list(), list()
        for init_x, init_v, final_x, final_v in zip(self.init_X, self.init_V,
                                                    self.final_X, self.final_V):
            init_a = self.calculate_init_accel(init_x, final_x,
                                               init_v, final_v,
                                               self.duration)
            self.init_A.append(init_a)

            final_a = self.calculate_final_accel(init_v, final_v, init_a, self.duration)
            self.final_A.append(final_a)
            self.jerk.append(self.calculate_jerk(init_a, final_a, self.duration))

    @property
    def ndim_state(self):
        """Number of state space dimensions."""
        return self.init_state.state_vector.shape[0]

    def covar(self, **kwargs):
        """Must be added due to inheritance."""
        raise NotImplementedError('Covariance not defined')

    def pdf(self, state1, state2, **kwargs):
        """Must be added due to inheritance."""
        raise NotImplementedError('pdf not defined')

    def rvs(self, num_samples=1, **kwargs):
        """Must be added due to inheritance."""
        raise NotImplementedError('rvs not defined')

    def function(self, state, time_interval, **kwargs):
        """Apply a constant jerk transition to `state`, for `time_interval` duration, keeping
        elements of state vector that are not position or velocity constant."""

        # Total time that will have passed since initial state up until transition is complete
        delta_t = (state.timestamp + time_interval - self.init_state.timestamp).total_seconds()

        # New position and velocity calculated only from `delta_t`
        # Assumed that `state` lies on the constant jerk path connecting `initial_state` with
        # `final_state`

        new_position = list()
        new_velocity = list()
        for init_x, init_v, init_a, jerk in zip(self.init_X, self.init_V, self.init_A, self.jerk):
            new_position.append(self.calculate_pos(init_x, init_v, init_a, jerk, delta_t))
            new_velocity.append(self.calculate_vel(init_v, init_a, jerk, delta_t))

        # Non-kinematic components remain constant
        new_sv = np.copy(state.state_vector).astype(float)  # May initiate with integers
        new_sv[self.position_mapping, 0] = new_position
        new_sv[self.velocity_mapping, 0] = new_velocity

        return StateVector(new_sv)

    @staticmethod
    def calculate_pos(init_x, init_v, init_a, jerk, T):
        """Calculate position, along a particular axis.

        Parameters
        ----------
        init_x: float
            Initial position along axis
        init_v: float
            Initial velocity along axis
        init_a: float
            Initial acceleration along axis
        jerk: float
            Constant jerk value along axis
        T: float
            Number for seconds to carry-out jerk transition

        Returns
        -------
        float
            New position along axis, given by:
            :math:`X' = \frac{J_0 T^3}{6} + \frac{A_0 T^2}{2} + V_0 T + X_0`
        """
        return (jerk * T ** 3) / 6 + (init_a * T ** 2) / 2 + init_v * T + init_x

    @staticmethod
    def calculate_vel(init_v, init_a, jerk, T):
        """Calculate velocity, along a particular axis.

        Parameters
        ----------
        init_v: float
            Initial velocity along axis
        init_a: float
            Initial acceleration along axis
        jerk: float
            Constant jerk value along axis
        T: float
            Number for seconds to carry-out jerk transition

        Returns
        -------
        float
            New velocity along axis, given by:
            :math:`V' = \frac{J_0 T^2}{2} + A_0 T + V_0`
        """
        return (jerk * T ** 2) / 2 + init_a * T + init_v

    @staticmethod
    def calculate_init_accel(init_x, final_x, init_v, final_v, T):
        """Calculate initial acceleration, along a particular axis.

        Parameters
        ----------
        init_x: float
            Initial position along axis
        final_x: float
            Final position along axis
        init_v: float
            Initial velocity along axis
        final_v: float
            Final velocity along axis
        T: float
            Number for seconds to get from initial to final state

        Returns
        -------
        float
            Initial acceleration along axis, given by:
            :math:`A_0 = \frac{6}{T^2}(X_1 - X_0 - \frac{2 V_0 T}{3} - \frac{V_1 T}{3})`
        """
        return (6 / T ** 2) * (final_x - init_x - (2 * init_v * T / 3) - (final_v * T / 3))

    @staticmethod
    def calculate_final_accel(init_v, final_v, init_a, T):
        """Calculate final acceleration, along a particular axis.

        Parameters
        ----------
        init_v: float
            Initial velocity along axis
        final_v: float
            Final velocity along axis
        init_a: float
            Initial acceleration along axis
        T: float
            Number for seconds to get from initial to final state

        Returns
        -------
        float
            Final acceleration along axis, given by:
            :math:`A_1 = \frac{2}{T} (V_1 - V_0) - A_0`
        """
        if T == 0:
            return 0
        return (2 / T) * (final_v - init_v) - init_a

    @staticmethod
    def calculate_jerk(init_a, final_a, T):
        """Calculate constant jerk, along a particular axis.

        Parameters
        ----------
        init_a: float
            Initial acceleration along axis
        final_a: float
            Final acceleration along axis
        T: float
            Number for seconds to get from initial to final state

        Returns
        -------
        float
            Constant jerk along axis, given by:
            :math:`J = \frac{A_1 - A_0}{T}`
        """
        return (final_a - init_a) / T

    @classmethod
    def create_models(cls, states: Sequence[State], position_mapping, velocity_mapping=None):
        """Generate a list of :class:`~.ConstantJerkSimulator` and transition durations, given a
        list of states."""

        if not all(state.ndim == state2.ndim for state, state2 in combinations(states, 2)):
            raise ValueError("All states must have the same ndim")

        transition_models, transition_times = list(), list()

        for current_state, next_state in zip(states[:-1], states[1:]):

            transition_times.append(next_state.timestamp - current_state.timestamp)

            transition_models.append(
                cls(position_mapping=position_mapping,
                    velocity_mapping=velocity_mapping,
                    init_state=current_state,
                    final_state=next_state)
            )

        return transition_models, transition_times
