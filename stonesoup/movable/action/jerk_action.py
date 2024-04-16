import numpy as np
from datetime import timedelta
from copy import copy, deepcopy
from typing import Sequence, Optional
from scipy.optimize._shgo_lib import sobol_seq

from ...base import Property
from ...types.state import State
from ...sensormanager.action import Action, ActionGenerator
from ...simulator.transition import ConstantJerkSimulator


class ConstantJerkAction(Action):
    """The action of moving a platform to a destination using a
    :class:`~.ConstantJerk` transition model."""

    target_value: State = Property(doc="Destination state of the platform.")

    def act(self, current_time, end_time, init_value, *args, **kwargs):
        """Propagates the platform position using the :attr:`transition_model`.
        Parameters
        ----------
        current_time: :class:`datetime.datetime
            Current time
        end_time: :class:`datetime.datetime
            A timestamp signifying the end of the maneuver (the default is ``None``)
        init_value: Any
            Current platform state
        Returns
        -------
        Any
            New platform state
        Notes
        -----
        This methods updates the value of :attr:`position`.
        Any provided ``kwargs`` are forwarded to the :attr:`transition_model`.
        If :attr:`transition_model` or ``timestamp`` is ``None``, the method has
        no effect, but will return successfully.
        This method updates :attr:`transition_model`, :attr:`transition_index` and
        :attr:`current_interval`:
        If the timestamp provided gives a time delta greater than :attr:`current_interval` the
        :attr:`transition_model` is called for the rest of its corresponding duration, and the move
        method is called again on the next transition model (by incrementing
        :attr:`transition_index`) in :attr:`transition_models` with the residue time delta.
        If the time delta is less than :attr:`current_interval` the :attr:`transition_model` is
        called for that duration and :attr:`current_interval` is reduced accordingly.
        """

        states = copy(init_value)
        # Time step is one increment in the simulation.
        # This function calculates the state of the platform for one time step
        duration = end_time - current_time

        transition_model = ConstantJerkSimulator(
            position_mapping=self.generator.owner.position_mapping,
            velocity_mapping=self.generator.owner.velocity_mapping,
            init_state=states[-1],
            final_state=self.target_value)

        temp_state = State(
            state_vector=transition_model.function(
                state=states[-1],
                time_interval=duration,
                **kwargs),
            timestamp=end_time
        )

        # Update the platform state
        states.append(temp_state)

        return states


class JerkActionGenerator(ActionGenerator):
    """Generates possible actions for moving a platform with a
    :class:`~.ConstantJerk` transition model."""

    constraints: tuple = Property(doc="Max speed and acceleration.")
    position_mapping: Sequence[int] = Property(
        doc="Mapping between platform position and state vector. For a "
            "position-only 3d platform this might be ``[0, 1, 2]``. For a "
            "position and velocity platform: ``[0, 2, 4]``")
    velocity_mapping: Optional[Sequence[int]] = Property(
        default=None,
        doc="Mapping between platform velocity and state dims. If not "
            "set, it will default to ``[m+1 for m in position_mapping]``")
    state: State = Property(doc="Current platform state")  # TODO: update description

    @property
    def default_action(self):
        """
        Default action is for platform to stay in the same location.
        """

        state = deepcopy(self.state)
        # Duration set to be how long it takes to decelerate to 0 velocity
        duration = 30 * self.constraints[0] / self.constraints[1]
        state.state_vector[self.velocity_mapping, ] = 0
        state.timestamp += timedelta(seconds=duration)
        return self.jerk_action_from_state(state)

    def __iter__(self):
        duration = self.end_time - self.state.timestamp

        for x, y in self.movement_grid(duration):
            state = State(state_vector=(0., 0., 0., 0.), timestamp=self.end_time)
            state.state_vector[self.position_mapping[0]] = x
            state.state_vector[self.position_mapping[1]] = y
            yield self.jerk_action_from_state(state)

    def __contains__(self, item):
        point = item.state_vector[self.position_mapping, ]

        return self.is_reachable(point)

    def initial_value(self):
        return self.default_action()

    def distance_travelled(self, v_init, v_max, a_max, duration=None):
        """
        Calculates distance travelled
        Parameters
        ----------
        v_init: float
            Initial velocity
        v_max: float
            Maximum velocity
        a_max: float
            Maximum acceleration
        duration: timedelta
            Duration of action
        Returns
        -------
        float
            Distance travelled given initial velocity and constraints on velocity and acceleration.
        """
        if duration is None:
            duration = self.end_time - self.state.timestamp

        # 1d equation for distance travelled given initial velocity and constraints on
        # velocity and acceleration
        t_vmax = float((v_max - v_init) / a_max)
        if isinstance(duration, timedelta):
            duration = duration.total_seconds()
        if t_vmax <= duration:
            return abs(0.5 * a_max * t_vmax ** 2 + v_init * t_vmax + v_max * (duration - t_vmax))
        else:
            return abs(0.5 * a_max * duration ** 2 + v_init * duration)

    # TODO: make this more generic (see Steve's messages)
    def define_movement_ellipse(self, duration=None):
        """
        Returns the parameters defining an ellipse which represents where the platform
        can move within the given time.
        Parameters
        ----------
        duration: timedelta
            Time platform will be travelling for
        Returns
        -------
        center: tuple (x, y)
            Giving center of ellipse
        a_max: float
            Length of major axis of ellipse
        b_max: float
            Length of minor axis of ellipse
        theta: :class:`~.Angle`
            Angle from x-axis to major axis of ellipse
        """

        if duration is None:
            duration = self.end_time - self.state.timestamp

        v_max, a_max = self.constraints
        v_xy = self.state.state_vector[self.velocity_mapping, ]
        speed = np.hypot(*v_xy)  # Velocity along direction of travel
        norm_v = v_xy / speed if speed > 0 else 0

        # Maximum distance travelled is distance travelled to t_vmax plus distance travelled at
        # v_max for remaining time
        # This gives the furthest point of the ellipse
        d1 = self.distance_travelled(speed, v_max, a_max, duration)

        # Maximum distance travelled when reversing direction
        # This gives the antipode of the furthest point
        d2 = self.distance_travelled(-speed, v_max, a_max, duration)

        # Maximum distance in perpendicular direction (from static)
        # This gives width of ellipse
        d3 = self.distance_travelled(0, v_max, a_max, duration)

        # Longest axis of ellipse
        amax = float((d2 + d1) / 2)

        # Shortest axis of ellipse
        bmax = d3

        # Define ellipse center as half max diameter of ellipse in direction of travel
        center = self.state.state_vector[self.position_mapping, ] \
            + (amax - d2) * norm_v

        # angle to rotate to original frame
        theta = np.arctan2(v_xy[1], v_xy[0])

        return center, amax, bmax, theta

    def is_reachable(self, point, duration=None, tol=1.01):
        """
        Checks if a location is reachable within a given duration by actioning the platform.
        Parameters
        ----------
        point: tuple
            Target destination of platform (x, y)
        duration: timedelta
            Duration of action
        tol: float
            Tolerance
        Returns
        -------
        bool
        """
        if duration is None:
            duration = self.end_time - self.state.timestamp

        A, B = point

        # Get the parameters defining the ellipse where the platform can move
        center, a_max, b_max, theta = self.define_movement_ellipse(duration)

        # angle from ellipse major axis to point
        t = np.arctan2(B - center[1], A - center[0]) - theta
        if t > np.pi:
            t = 2 * np.pi - t
        elif t < -np.pi:
            t += 2 * np.pi

        # point on ellipse at angle t (+ theta)
        x = a_max * np.cos(theta) * np.cos(t) - b_max * np.sin(t) * np.sin(theta) + center[0]
        y = a_max * np.sin(theta) * np.cos(t) + b_max * np.sin(t) * np.cos(theta) + center[1]

        # Distance edge of ellipse
        radius_at_t = np.hypot(x - center[0], y - center[1])

        # Distance to point
        distance = np.hypot(A - center[0], B - center[1])

        return distance <= radius_at_t * tol

    def movement_grid(self, duration=None, npoints=1064):
        """
        Defines the movement grid of points within the movement ellipse.
        Parameters
        ----------
        duration: timedelta
            Duration of action
        npoints: int
            Default is 1064
        """
        # Get the parameters defining the ellipse where the platform can move
        if duration is None:
            duration = self.end_time - self.state.timestamp

        center, amax, bmax, theta = self.define_movement_ellipse(duration)

        # Generate quasi-random numbers (2 dimensions, ranging from 0-1)
        for r, t in sobol_sequence_generator(ndim=2, npoints=npoints):
            # Scale t to go from -pi to pi
            t = 2 * np.pi * t - np.pi

            # Use r to scale the radial component (never exceeding the ellipse bounds)
            a = amax * np.sqrt(r)
            b = bmax * np.sqrt(r)

            # parametric definition of ellipse
            x = (a * np.cos(theta) * np.cos(t) - b * np.sin(t) * np.sin(theta) + center[0])
            y = (a * np.sin(theta) * np.cos(t) + b * np.sin(t) * np.cos(theta) + center[1])
            # x, y is eastings, northings

            yield x, y

    def action_from_value(self, x):
        """
        Generates a :class:`~.ConstantJerkAction` which would enable the platform to reach state x.
        Parameters
        ----------
        x: array
        Returns
        -------
        :class:`~.ConstantJerkAction`
        """
        x = [x[0], 0, x[1], 0]
        return self.jerk_action_from_state(State(x, self.end_time))

    def jerk_action_from_state(self, state):
        """
        Generates a :class:`~.ConstantJerkAction` which would enable the platform to
        reach the given state.
        Parameters
        ----------
        state: :class:`~.State`
        Returns
        -------
        :class:`~.ConstantJerkAction`
        """
        end_time = state.timestamp
        init_state = deepcopy(self.state)
        duration = end_time - init_state.timestamp
        point = state.state_vector[self.position_mapping, ]

        if not self.is_reachable(point, duration):
            raise PointUnreachableError(f"""Point {point[0], point[1]} not reachable from
                                        {init_state.state_vector[self.position_mapping[0]],
                                         init_state.state_vector[self.position_mapping[1]]}
                                        in {duration} seconds.""")
        return ConstantJerkAction(end_time=end_time,
                                  target_value=state,
                                  generator=self)


class PointUnreachableError(Exception):
    """
    Raised when a platform is given a task to move to a point
    that it cannot reach in the given time
    """
    pass


def sobol_sequence_generator(ndim, npoints):
    QRNG = sobol_seq.Sobol()
    yield from ((r, t) for r, t in QRNG.i4_sobol_generate(ndim, npoints))
