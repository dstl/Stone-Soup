from abc import abstractmethod
import numpy as np
from datetime import timedelta
from typing import Sequence, Optional, Iterator
from scipy.optimize._shgo_lib import sobol_seq

from ...base import Property
from ...types.state import State
from ...sensormanager.action import StateVectorActionGenerator, Action


class TransitionModelActionGenerator(StateVectorActionGenerator):
    """A base class for action generators for transition model actions.

       NB: This is currently only implemented for 2D action space (e.g. in the x, y plane)."""
    # TODO: Should this be named to reflect 2D implementation?
    # TODO: Should this raise errors elsewhere?

    constraints: tuple = Property(doc='Max speed and acceleration')
    position_mapping: Sequence[int] = Property(
        doc="Mapping between platform position and state vector. For a "
            "position-only 2d platform this might be ``[0, 1]``. For a "
            "position and velocity platform: ``[0, 2]``")
    velocity_mapping: Optional[Sequence[int]] = Property(
        default=None,
        doc="Mapping between platform velocity and state dims. If not "
            "set, it will default to ``[m+1 for m in position_mapping]``")
    state: State = Property(doc="Current platform state")

    def __contains__(self, item):
        point = item.state_vector[self.position_mapping, ]
        return self.is_reachable(point)

    @abstractmethod
    def __iter__(self) -> Iterator[Action]:
        raise NotImplementedError()

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


def sobol_sequence_generator(ndim, npoints):
    QRNG = sobol_seq.Sobol()
    yield from ((r, t) for r, t in QRNG.i4_sobol_generate(ndim, npoints))
