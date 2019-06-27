# -*- coding: utf-8 -*-
import datetime
import numpy as np

from ..base import Property
from .state import State


class OrbitState(State):

    r"""

    The orbital state base type

    Can create either a Keplerian, Two Line Element (TLE) or a Equinoctial state vector.

    Create using a :class:`State` with :attr:`State.state_vector` :math:`X_{t_{0}}` at Epoch
    :attr:`State.timestamp` :math:`t_0`

    The gravitational parameter :math:`GM` can be defined. If left undefined it defaults to that of the Earth,
    :math:`3.986004418 (\pm 0.000000008) \\times 10^{14} \mathrm{m}^3 \mathrm{s}^{−2}`.

    """

    grav_parameter = Property(
        float, default=3.986004418e14,
        doc="Standard gravitational parameter :math:`\\mu = G M`")

    metadata = Property(
        {}, default={}, doc="Dictionary containing metadata about orbit")

    def __init__(self, state_vector, *args, **kwargs):
        if len(state_vector) != 6:
            raise ValueError("State vector shape should be 6x1 : got {}".format(
                state_vector.shape))
        super().__init__(state_vector, *args, **kwargs)


class KeplerianOrbitState(OrbitState):
    """
    For the Keplerian state vector:

        .. math::

            X_{t_0} = [e, a, i, \Omega, \omega, \\theta]^{T} \\

    where:
    :math:`e` is the orbital eccentricity (unitless),
    :math:`a` the semi-major axis (m),
    :math:`i` the inclination (rad),
    :math:`\Omega` is the longitude of the ascending node (rad),
    :math:`\omega` the argument of periapsis (rad), and
    :math:`\\theta` the true anomaly (rad).

    :reference: Curtis, H.D. 2010, Orbital Mechanics for Engineering Students (3rd Ed), Elsevier Aerospace Engineering Series
    """
    def __init__(self, state_vector, *args, **kwargs):
        if np.less(state_vector[0][0], 0.0) | np.greater(state_vector[0][0], 1.0):
            raise ValueError("Eccentricity should be between 0 and 1: got {}"
                             .format(state_vector[0][0]))
        if np.less(state_vector[2][0], 0.0) | np.greater(state_vector[2][0], np.pi):
            raise ValueError("Inclination should be between 0 and pi: got {}"
                             .format(state_vector[2][0]))
        if np.less(state_vector[3][0], 0.0) | np.greater(state_vector[3][0], 2*np.pi):
            raise ValueError("Longitude of Ascending Node should be between 0 and 2*pi: got {}"
                             .format(state_vector[3][0]))
        if np.less(state_vector[4][0], 0.0) | np.greater(state_vector[4][0], 2*np.pi):
            raise ValueError("Argument of Periapsis should be between 0 and 2*pi: got {}"
                             .format(state_vector[4][0]))
        if np.less(state_vector[5][0], 0.0) | np.greater(state_vector[5][0], 2*np.pi):
            raise ValueError("True Anomaly should be between 0 and 2*pi: got {}"
                             .format(state_vector[5][0]))
        super().__init__(state_vector, *args, **kwargs)

    @property
    def eccentricity(self):
        """

        :return: Orbit eccentricity, :math:`e` (unitless)

        :math:`e = 0` is a circle
        :math:`0 < e < 1`, an elliptical orbit
        :math:`e = 1` is a parabola, and
        :math:`e > 1' denotes a hyperbolic orbit

        """
        eccentricity = self.state_vector[0][0]
        eccentricity = np.remainder(eccentricity, np.pi/2)
        return eccentricity

    @property
    def semimajor_axis(self):
        """

        :return: Orbit semimajor axis, :math:`a` (m)

        Note that positive number indicates an elliptical (or circular) orbit, negative a hyperbolic orbit and
        0 is parabolic

        """
        return self.state_vector[1][0]

    @property
    def inclination(self):
        """

        :return: Orbit inclination (radians)

        Usually defined as the angle between the orbital plane and that defined by the rotational axis of the parent
        body


        """
        # Ensure that what's returned is between 0 and PI/2 (required by some of the astro libraries)
        inclination = self.state_vector[2][0]
        # inclination = np.remainder(inclination, np.pi/2)
        return inclination

    @property
    def long_asc_node(self):
        """

        :return: Longitude of the ascending node :math:`\Omega` (radians)

        For geocentric orbits, often known as the right ascension of the ascending node

        The longitude at which the orbit crosses the parent's rotational plane from negative to positive

        """
        return self.state_vector[3][0]

    @property
    def arg_periapsis(self):
        """

        :return: Argument of periapsis, :math:`\omega` (radians)

        Or the argument of perigee for geocentric orbits, argument of perihelion for heliocentric orbits.

        The angle between the orbital semi-major axis and the longitude of the ascending node (measured anticlockwise
        from the ascending node to the semi-major axis). Note that this is undefined for circular orbits, as circles
        don't have a unique semi-major axis.



        """
        return self.state_vector[4][0]

    @property
    def true_anomaly(self):
        """

        :return: True anomaly (radians)

        The current angle of the orbiting body measured anticlockwise from the argument of periapsis.

        """
        return self.state_vector[5][0]

    @property
    def period(self):
        """

        :return: Orbital period, :math:`T`

        """
        return ((2*np.pi)/np.sqrt(self.grav_parameter))*np.power(self.semimajor_axis, 3/2)

    @property
    def spec_angular_mom(self):
        """

        :return: Magnitude of the specific angular momentum, :math:`h`

        """
        return np.sqrt(self.grav_parameter * self.semimajor_axis * (1 - self.eccentricity**2))

    @property
    def spec_orb_ener(self):
        """

        :return: Specific orbital energy (:math:`\frac{-GM}{2a}`)

        """
        return -self.grav_parameter/(2 * self.semimajor_axis)

    @property
    def mean_anomaly(self):
        """

        :return: Mean anomaly, :math:`M` (radians)

        Uses the eccentric anomaly and Kepler's equation to get mean anomaly from true anomaly and eccentricity

        """

        ecc_anom = 2 * np.arctan(np.sqrt((1-self.eccentricity)/(1+self.eccentricity)) *
                                 np.tan(self.true_anomaly/2))
        return ecc_anom - self.eccentricity * np.sin(ecc_anom) # Kepler's equation


class TLEOrbitState(OrbitState):
    """
    For the TLE state vector:

        .. math::

            X_{t_0} = [e, i, \Omega, \omega, n, M_0]^{T} \\

    where :math:`e` is the orbital eccentricity (unitless),
    :math:`i` the inclination (rad),
    :math:`\Omega` is the longitude of the ascending node (rad),
    :math:`\omega` the argument of perigee (rad),
    :math:`n` the mean motion (rad) and
    :math:'M_0' the mean anomaly (rad).
    """
    def __init__(self, state_vector, *args, **kwargs):
        if np.less(state_vector[0][0], 0.0) | np.greater(state_vector[0][0], 1.0):
            raise ValueError("Eccentricity should be between 0 and 1: got {}"
                             .format(state_vector[0][0]))
        if np.less(state_vector[1][0], 0.0) | np.greater(state_vector[1][0], np.pi):
            raise ValueError("Inclination should be between 0 and pi: got {}"
                             .format(state_vector[1][0]))
        if np.less(state_vector[2][0], 0.0) | np.greater(state_vector[2][0], 2*np.pi):
            raise ValueError("Longitude of Ascending Node should be between 0 and 2*pi: got {}"
                             .format(state_vector[2][0]))
        if np.less(state_vector[3][0], 0.0) | np.greater(state_vector[3][0], 2*np.pi):
            raise ValueError("Argument of Periapsis should be between 0 and 2*pi: got {}"
                             .format(state_vector[3][0]))
        if np.less(state_vector[5][0], 0.0) | np.greater(state_vector[5][0], 2*np.pi):
            raise ValueError("Mean Anomaly should be between 0 and 2*pi: got {}"
                             .format(state_vector[5][0]))
        super().__init__(state_vector, *args, **kwargs)

    @property
    def eccentricity(self):
        """

        :return: Orbit eccentricity, :math:`e` (unitless)

        :math:`e = 0` is a circle
        :math:`0 < e < 1`, an elliptical orbit
        :math:`e = 1` is a parabola, and
        :math:`e > 1' denotes a hyperbolic orbit

        """
        return self.state_vector[0][0]

    @property
    def inclination(self):
        """

        :return: Orbit inclination (radians)

        Usually defined as the angle between the orbital plane and that defined by the rotational axis of the parent
        body


        """
        # Ensure that what's returned is between 0 and PI/2 (required by some of the astro libraries)
        return self.state_vector[1][0]

    @property
    def long_asc_node(self):
        """

        :return: Longitude of the ascending node :math:`\Omega` (radians)

        For geocentric orbits, often known as the right ascension of the ascending node

        The longitude at which the orbit crosses the parent's rotational plane from negative to positive

        """
        return self.state_vector[2][0]

    @property
    def arg_periapsis(self):
        """

        :return: Argument of periapsis, :math:`\omega` (radians)

        Or the argument of perigee for geocentric orbits, argument of perihelion for heliocentric orbits.

        The angle between the orbital semi-major axis and the longitude of the ascending node (measured anticlockwise
        from the ascending node to the semi-major axis). Note that this is undefined for circular orbits, as circles
        don't have a unique semi-major axis.



        """
        return self.state_vector[3][0]

    @property
    def mean_motion(self):
        """

        :return: Mean motion, :math:`n` (radians)

        """
        return self.state_vector[4][0]

    @property
    def mean_anomaly(self):
        """

        :return: Argument of periapsis, :math:`M` (radians)

        """
        return self.state_vector[5][0]


class EquinoctialOrbitState(OrbitState):
    """
    For the Equinoctial state vector:

        .. math::

            X_{t_0} = [a, h, k, p, q, \lambda]^{T} \\

    where :math:`a` the semi-major axis (m),
    :math:`h` is the horizontal component of the eccentricity :math:`e`,
    :math:`k` is the vertical component of the eccentricity :math:`e`,
    :math:`q` is the horizontal component of the inclination :math:`i`,
    :math:`k` is the vertical component of the inclination :math:`i` and
    :math:'lambda' is the mean longitude
    """
    def __init__(self, state_vector, *args, **kwargs):
        if np.less(state_vector[1][0], -1.0) | np.greater(state_vector[1][0], 1.0):
            raise ValueError("Horizontal Eccentricity should be between -1 and 1: got {}"
                             .format(state_vector[1][0]))
        if np.less(state_vector[2][0], -1.0) | np.greater(state_vector[2][0], 1.0):
            raise ValueError("Vertical Eccentricity should be between -1 and 1: got {}"
                             .format(state_vector[2][0]))
        if np.less(state_vector[3][0], -1.0) | np.greater(state_vector[3][0], 1.0):
            raise ValueError("Horizontal Inclination should be between -1 and 1: got {}"
                             .format(state_vector[3][0]))
        if np.less(state_vector[4][0], -1.0) | np.greater(state_vector[4][0], 1.0):
            raise ValueError("Vertical Inclination should be between -1 and -1: got {}"
                             .format(state_vector[4][0]))
        # if np.less(state_vector[5][0], 0.0) | np.greater(state_vector[5][0], 2*np.pi):
        #     raise ValueError("Mean Longitude should be between 0 and 2*pi: got {}"
        #                      .format(state_vector[5][0]))
        super().__init__(state_vector, *args, **kwargs)

    @property
    def semimajor_axis(self):
        """

        :return: Orbit semimajor axis, :math:`a` (m)

        Note that positive number indicates an elliptical (or circular) orbit, negative a hyperbolic orbit and
        0 is parabolic

        """
        return self.state_vector[0][0]

    @property
    # Set of equinoctal elements
    def horizontal_eccentricity(self):
        """

        :return: eccentricity vector in Equinoctal coordinates (h)
        """
        return self.state_vector[1][0]

    @property
    def vertical_eccentricity(self):
        """

        :return: the vertical component of the eccentricity (k)

        """
        return self.state_vector[2][0]

    @property
    def horizontal_inclination(self):
        """

        :return: horizontal component of inclination (Equinoctal p)
        """
        return self.state_vector[3][0]

    @property
    def vertical_inclination(self):
        """

        :return: horizontal component of inclination (Equinoctal q)
        """

        return self.state_vector[4][0]

    @property
    def mean_longitude(self):
        """

        :return: Equinoctal true longitude (\lambda_0)
        """
        return self.state_vector[5][0]

    @property
    def period(self):
        """

        :return: Orbital period, :math:`T`

        """
        return ((2*np.pi)/np.sqrt(self.grav_parameter))*np.power(self.semimajor_axis, 3/2)


class CartesianOrbitState(OrbitalState):
    """
    For the Cartesian state vector:

        .. math::

            X_{t_0} = [x,y,z,x^\dot,y^\dot,z^\dot]^{T} \\

    where:
     :math:`x` is the position in the x axis (m),
     :math:`y` is the position in the y axis (m),
     :math:`z` is the position in the z axis (m),
     :math:`x^\dot` is the velocity in the x axis (m/s),
     :math:`y^\dot` is the velocity in the y axis (m/s),
     :math:`z^\dot` is the velocity in the z axis (m/s),
    """
    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)

    @property
    def position_vector(self):
        """

        :return: Position vector [:math:`x`,:math:`y`,:math:`z`]

        """

        return self.state_vector[0:3][0]

    @property
    def velocity_vector(self):

        """
        :return: Velocity vector [:math:`x^\dot`,:math:`y^\dot`,:math:`z^\dot`]

        """
        return self.state_vector[3::][0]
