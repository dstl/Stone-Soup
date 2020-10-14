# -*- coding: utf-8 -*-

import numpy as np
from ..orbital_functions import keplerian_to_rv, tru_anom_from_mean_anom

from ..base import Property
from .array import CovarianceMatrix, StateVector
from .state import State


class OrbitalState(State):
    r"""The orbital state base type. This is the building block of
    Stone Soup's orbital inference routines and follows the principle
    that you shouldn't have to care which parameterisation you use. The
    class stores relevant information internally and undertakes
    whatever conversions are necessary.

    The :attr:`state_vector` is held as :math:`[\mathbf{r},
    \mathbf{v}]`, the "Orbital State Vector" (as traditionally
    understood in orbital mechanics), where :math:`\mathbf{r}` is the
    (3D) Cartesian position in the primary-centered inertial frame,
    while :math:`\mathbf{v}` is the corresponding velocity vector. All
    other parameters are accessed via functions. Formulae for
    conversions are generally found in, or derived from, [1].

    The object is constructed from the input vector :math:`X_{t_{0}}`
    at epoch :attr:`State.timestamp` :math:`t_0` in the appropriate
    coordinates indicated via the keyword:

        coordinates = "Cartesian" (the orbital state vector),

             .. math::

                X_{t_0} = [r_x, r_y, r_z, \dot{r}_x, \dot{r}_y,
                    \dot{r}_z]^{T}

            where :math:`r_x, r_y, r_z` are the Cartesian position
            coordinates in the Primary-Centered Inertial frame and
            :math:`\dot{r}_x, \dot{r}_y, \dot{r}_z` are the
            corresponding velocity coordinates.

        coordinates = "Keplerian" (Keplarian elements),

            .. math::

                X_{t_0} = [e, a, i, \Omega, \omega, \theta]^{T}

            where :math:`e` is the orbital eccentricity (unitless),
            :math:`a` the semi-major axis ([length]), :math:`i` the
            inclination (radian), :math:`\Omega` is the longitude of the
            ascending node (radian), :math:`\omega` the argument of
            periapsis (radian), and :math:`\theta` the true anomaly
            (radian).

        coordinates = "TLE" (Two-Line elements, [2]),

            .. math::

                X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T}

            where :math:`i` the inclination (radians), :math:`\Omega`
            is the longitude of the ascending node (radian), :math:`e`
            is the orbital eccentricity (unitless), :math:`\omega` the
            argument of perigee (radian), :math:`M_0` the mean anomaly
            (radian) :math:`n` the mean motion (radian
            [time] :math:`^{-1}`)

        coordinates = "Equinoctial" (equinoctial elements, [3])

            .. math::

                X_{t_0} = [a, h, k, p, q, \lambda]^{T}

            where :math:`a` the semi-major axis ([length]),
            :math:`h` is the horizontal component of the
            eccentricity, :math:`k` is the vertical component of the
            eccentricity, :math:`p` is the horizontal component of the
            inclination, :math:`q` is the vertical component of the
            inclination and :math:`\lambda` is the mean longitude

    The gravitational parameter :math:`\mu = GM` can be defined. If
    left undefined it defaults to that of the Earth, :math:`3.986004418
    \, (\pm \, 0.000000008) \times 10^{14} \mathrm{m}^3 \mathrm{s}^{−2}`

    References
    ----------
    1. Curtis, H.D. 2010, Orbital Mechanics for Engineering
    Students (3rd Ed), Elsevier Aerospace Engineering Series

    2. https://spaceflight.nasa.gov/realdata/sightings/SSapplications/\
    Post/JavaSSOP/SSOP_Help/tle_def.html

    3. Broucke, R. A. & Cefola, P. J. 1972, Celestial Mechanics, Volume
    5, Issue 3, pp. 303-310

    """

    coordinates = Property(
        str, default='Cartesian',
        doc="The parameterisation used on initiation. Acceptable values "
            "are 'Cartesian' (default), 'Keplerian', 'TLE', or 'Equinoctial'. "
            "All other inputs will return errors."
    )

    grav_parameter = Property(
        float, default=3.986004418e14,
        doc=r"Standard gravitational parameter :math:`\mu = G M`. The default "
            r"is :math:`3.986004418 \times 10^{14} \,` "
            r":math:`\mathrm{m}^3 \mathrm{s}^{-2}`.")

    covar = Property(
        CovarianceMatrix, default=None,
        doc="The covariance matrix. Care should be exercised in that its "
            "coordinate frame isn't defined, and output will be highly "
            "dependant on which parameterisation is chosen."
    )

    metadata = Property(
        dict, default={}, doc="Dictionary containing metadata about orbit"
    )

    def __init__(self, state_vector, *args, **kwargs):
        r"""Can be initialised in a number of different ways according to
        preference

        Parameters
        ----------
        state_vector : :class:`numpy.array`
            The input vector whose elements depend on the parameterisation
            used. See 'Keywords' below. Must have dimension 6x1.

        Keywords
        --------
        coordinates
            The chosen input coordinate frame. Can be 'Cartesian', 'Keplerian',
            'TLE' or 'Equinoctial'.


        Returns
        -------
        : Constructs the class

        """
        if len(state_vector) != 6:
            raise ValueError("State vector shape should be 6x1 : got {}"
                             .format(state_vector.shape))

        super().__init__(state_vector, *args, **kwargs)

        # Query the coordinates
        if self.coordinates.lower() == 'cartesian':
            #  No need to do any conversions here
            self.state_vector = StateVector(state_vector)

        elif self.coordinates.lower() == 'keplerian':
            # Convert Keplerian elements to Cartesian
            self.state_vector = StateVector(keplerian_to_rv(
                state_vector, grav_parameter=self.grav_parameter))

        elif self.coordinates.upper() == 'TLE':
            # TODO: ensure this works for parabolas and hyperbolas
            # Get the semi-major axis from the mean motion
            semimajor_axis = np.cbrt(self.grav_parameter/state_vector[5, 0]**2)

            # True anomaly from mean anomaly
            tru_anom = tru_anom_from_mean_anom(state_vector[4, 0],
                                               state_vector[2, 0])

            # Use given and derived quantities to convert from Keplarian to
            # Cartesian
            self.state_vector = StateVector(keplerian_to_rv(
                np.array([[state_vector[2, 0]], [semimajor_axis],
                          [state_vector[0, 0]], [state_vector[1, 0]],
                          [state_vector[3, 0]], [tru_anom]]),
                grav_parameter=self.grav_parameter))

        elif self.coordinates.lower() == 'equinoctial':
            # Calculate the Keplarian element quantities
            semimajor_axis = state_vector[0, 0]
            raan = np.arctan2(state_vector[3, 0], state_vector[4, 0])
            inclination = 2*np.arctan(state_vector[3, 0]/np.sin(raan))
            arg_per = np.arctan2(state_vector[1, 0], state_vector[2, 0]) - raan
            mean_anomaly = state_vector[5, 0] - arg_per - raan
            eccentricity = state_vector[1, 0]/(np.sin(arg_per + raan))

            # True anomaly from mean anomaly
            tru_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

            # Convert from Keplarian to Cartesian
            self.state_vector = StateVector(keplerian_to_rv(
                np.array([[eccentricity], [semimajor_axis], [inclination],
                          [raan], [arg_per], [tru_anom]]),
                grav_parameter=self.grav_parameter))

        else:
            raise TypeError("Coordinate keyword not recognised")

    # Some vector quantities
    @property
    def _nodeline(self):
        """The vector node line (defines the longitude of the ascending node
        in the Primary-centered inertial frame)

        Returns
        -------
        : numpy.array
            The node line (defines the longitude of the ascending node)
        """
        k = np.array([0, 0, 1])
        boldh = self.specific_angular_momentum

        boldn = np.cross(k, boldh, axis=0)
        n = np.sqrt(np.dot(boldn.T, boldn).item())

        # If inclination is 0, the node line is [0] and has 0 magnitude. By
        # convention in these situations, we set the node line as a unit vector
        # pointing along x. Note that the magnitude of the vector is not
        # consistent with that produced by the cross product. (But we assume
        # that the node line is only used for directional information.
        if n < np.finfo(n).eps:
            return np.array([1, 0, 0])
        else:
            return boldn

    @property
    def _eccentricity_vector(self):
        r""" The eccentricity vector

        Returns
        -------
        : numpy.array
            The eccentricity vector, :math:`\mathbf{e}`
        """

        rang = self.range
        speed = self.speed
        radial_velocity = np.dot(self.state_vector[0:3].T,
                                 self.state_vector[3:6]).item()/rang

        return (1/self.grav_parameter) * ((speed**2 - self.grav_parameter/rang)
                                          * self.state_vector[0:3] - rang *
                                          radial_velocity *
                                          self.state_vector[3:6])

    @property
    def specific_angular_momentum(self):
        r"""
        Returns
        -------
        : numpy.array
            The specific angular momentum, :math:`\mathbf{h}`

        """
        return np.cross(self.state_vector[0:3], self.state_vector[3:6], axis=0)

    @property
    def cartesian_state_vector(self):
        r"""
        Returns
        -------
        : :class:`~.StateVector`
            The state vector

                .. math::

                    X_{t_0} = [r_x, r_y, r_z, \dot{r}_x, \dot{r}_y,
                    \dot{r}_z]^{T}

            in Primary-Centred' Inertial coordinates, equivalent to ECI
            in the case of the Earth

        """
        return self.state_vector

    # Some scalar quantities
    @property
    def epoch(self):
        """
        Returns
        -------
        : :class:`~.datetime.datetime`
            The epoch, or state timestamp

        """
        return self.timestamp

    @property
    def range(self):
        """
        Returns
        -------
        : float
            The distance to object (from gravitational centre of
            primary)

        """
        return np.sqrt(np.dot(self.state_vector[0:3].T,
                              self.state_vector[0:3])).item()

    @property
    def speed(self):
        """
        Returns
        -------
        : float
            The current instantaneous speed (scalar)

        """
        return np.sqrt(np.dot(self.state_vector[3:6].T,
                              self.state_vector[3:6]).item())

    @property
    def eccentricity(self):
        r"""
        Returns
        -------
        : float
            The orbital eccentricity, :math:`e \; (0 \le e \le 1)`

        Note
        ----
        This version of the calculation uses a form dependent only on scalars

        """
        # TODO Check to see which of the following is quicker/better

        # Either
        # return np.sqrt(np.dot(self._eccentricity_vector.T,
        # self._eccentricity_vector).item())
        # or
        return np.sqrt(1 + (self.mag_specific_angular_momentum ** 2 /
                            self.grav_parameter ** 2) *
                       (self.speed ** 2 - 2 * self.grav_parameter /
                        self.range))

    @property
    def semimajor_axis(self):
        """
        Returns
        -------
        : float
            The orbital semi-major axis

        """
        return (self.mag_specific_angular_momentum**2 / self.grav_parameter) *\
               (1 / (1 - self.eccentricity**2))

        # Used to be this
        # return 1/((2/self.range) - (self.speed**2)/self.grav_parameter)

    @property
    def inclination(self):
        r"""
        Returns
        -------
        : float
            Orbital inclination, :math:`i \; (0 \le i < \pi)`

        """
        boldh = self.specific_angular_momentum
        h = self.mag_specific_angular_momentum

        # Note no quadrant ambiguity
        return np.arccos(np.clip(boldh[2].item()/h, -1, 1))

    @property
    def longitude_ascending_node(self):
        r"""
        Returns
        -------
        : float
            The longitude (or right ascension) of ascending node,
            :math:`\Omega \; (0 \le \Omega < 2\pi)`

        """

        boldn = self._nodeline
        n = np.sqrt(np.dot(boldn.T, boldn).item())

        # Quadrant ambiguity
        if boldn[1].item() >= 0:
            return np.arccos(np.clip(boldn[0].item()/n, -1, 1))
        elif boldn[1].item() < 0:
            return 2*np.pi - np.arccos(np.clip(boldn[0].item()/n, -1, 1))
        else:
            raise ValueError("Really shouldn't be able to arrive here")

    @property
    def argument_periapsis(self):
        r"""
        Returns
        -------
        : float
            The argument of periapsis, :math:`\omega \; (0 \le \omega
            < 2\pi)`

        """

        boldn = self._nodeline
        n = np.sqrt(np.dot(boldn.T, boldn).item())
        bolde = self._eccentricity_vector

        # If eccentricity is 0 then there's no unambiguous longitude of
        # periapsis. In these situations we set the argument of periapsis to
        # 0.
        if self.eccentricity < np.finfo(self.eccentricity).eps:
            return 0

        # Quadrant ambiguity. The clip function is required to mitigate against
        # the occasional floating-point errors which push the ratio outside the
        # -1,1 region.
        if bolde[2].item() >= 0:
            return np.arccos(np.clip(np.dot(boldn.T, bolde).item() /
                             (n * self.eccentricity), -1, 1))
        elif bolde[2].item() < 0:
            return 2*np.pi - np.arccos(np.clip(np.dot(boldn.T, bolde).item() /
                                       (n * self.eccentricity), -1, 1))
        else:
            raise ValueError("This shouldn't ever happen")

    @property
    def true_anomaly(self):
        r"""
        Returns
        -------
        : float
            The true anomaly, :math:`\theta \; (0 \le \theta < 2\pi)`

        """
        # Resolve the quadrant ambiguity.The clip function is required to
        # mitigate against floating-point errors which push the ratio outside
        # the -1,1 region.
        radial_velocity = np.dot(self.state_vector[0:3].T,
                                 self.state_vector[3:6]).item() / self.speed

        if radial_velocity >= 0:
            return np.arccos(np.clip(
                np.dot(self._eccentricity_vector.T / self.eccentricity,
                       self.state_vector[0:3] / self.range).item(), -1, 1))
        elif radial_velocity < 0:
            return 2*np.pi - np.arccos(np.clip(
                np.dot(self._eccentricity_vector.T / self.eccentricity,
                       self.state_vector[0:3] / self.range).item(), -1, 1))
        else:
            raise ValueError("Shouldn't arrive at this point")

    @property
    def eccentric_anomaly(self):
        r"""
        Returns
        -------
        : float
            The eccentric anomaly, :math:`E \; (0 \le E < 2\pi)`

        Note
        ----
            This computes the quantity exactly via the Keplerian
            eccentricity and true anomaly rather than via the mean
            anomaly using an iterative procedure.

        """
        return np.remainder(2 * np.arctan(np.sqrt((1 - self.eccentricity) /
                                                  (1 + self.eccentricity)) *
                                          np.tan(self.true_anomaly / 2)),
                            2*np.pi)

    @property
    def mean_anomaly(self):
        r"""
        Returns
        -------
        : float
            Mean anomaly, :math:`M \; (0 \le M < 2\pi`)

        Note
        ----
            Uses the eccentric anomaly and Kepler's equation to get
            mean anomaly from true anomaly and eccentricity.

        """

        return self.eccentric_anomaly - self.eccentricity * \
            np.sin(self.eccentric_anomaly)  # Kepler's equation

    @property
    def period(self):
        """
        Returns
        -------
        : float
            Orbital period, :math:`T` ([time])

        """
        return ((2 * np.pi) / np.sqrt(self.grav_parameter)) * \
            np.power(self.semimajor_axis, 3 / 2)

    @property
    def mean_motion(self):
        r"""
        Returns
        -------
        : float
            The mean motion, :math:`\frac{2 \pi}{T}`, where :math:`T` is the
            period, (rad / [time])

        """
        return 2 * np.pi / self.period

    @property
    def mag_specific_angular_momentum(self):
        """
        Returns
        -------
        : float
            The magnitude of the specific angular momentum, :math:`h`

        """
        boldh = self.specific_angular_momentum
        return np.sqrt(np.dot(boldh.T, boldh).item())

        # Alternative via scalars
        # return np.sqrt(self.grav_parameter * self.semimajor_axis *
        # (1 - self.eccentricity ** 2))

    @property
    def specific_orbital_energy(self):
        r"""
        Returns
        -------
        : float
            Specific orbital energy (:math:`\frac{-GM}{2a}`)

        """
        return -self.grav_parameter / (2 * self.semimajor_axis)

    @property
    def equinoctial_h(self):
        r"""
        Returns
        -------
        : float
            The horizontal component of the eccentricity in equinoctial
            coordinates is :math:`h = e \sin (\omega + \Omega)`

        """

        return self.eccentricity * np.sin(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinoctial_k(self):
        r"""
        Returns
        -------
        : float
            The vertical component of the eccentricity in equinoctial
            coordinates is :math:`k = e \cos (\omega + \Omega)`

        """

        return self.eccentricity * np.cos(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinoctial_p(self):
        r"""
        Returns
        -------
        : float
            The horizontal component of the inclination in equinoctial
            coordinates is :math:`p = \tan (i/2) \sin \Omega`

        """

        return np.tan(self.inclination/2) * \
            np.sin(self.longitude_ascending_node)

    @property
    def equinoctial_q(self):
        r"""
        Returns
        -------
        : float
            The vertical component of the inclination in equinoctial
            coordinates is :math:`q = \tan (i/2) \cos \Omega`

        """

        return np.tan(self.inclination / 2) * \
            np.cos(self.longitude_ascending_node)

    @property
    def mean_longitude(self):
        r"""
        Returns
        -------
        : float
            The mean longitude, defined as :math:`\lambda = M_0 + \omega +
            \Omega`

        """
        return self.mean_anomaly + self.argument_periapsis + \
            self.longitude_ascending_node

    # The following return vectors of complete sets of elements
    @property
    def keplerian_elements(self):
        r"""
        Returns
        -------
        : StateVector

            The vector of Keplerian elements

            .. math::

                    X = [e, a, i, \Omega, \omega, \theta]^{T}

            where :math:`e` is the orbital eccentricity (unitless),
            :math:`a` the semi-major axis ([length]), :math:`i` the
            inclination (radian), :math:`\Omega` is the longitude of
            the ascending node (radian), :math:`\omega` the argument of
            periapsis (radian), and :math:`\theta` the true anomaly
            (radian)

        """

        return StateVector([self.eccentricity,
                            self.semimajor_axis,
                            self.inclination,
                            self.longitude_ascending_node,
                            self.argument_periapsis,
                            self.true_anomaly])

    @property
    def two_line_element(self):
        r"""
        Returns
        -------
        : StateVector

            The Two-Line Element vector

                .. math::

                    X = [i, \Omega, e, \omega, M_0, n]^{T}

            where :math:`i` the inclination (radian) :math:`\Omega` is
            the longitude of the ascending node (radian), :math:`e` is
            the orbital eccentricity (unitless), :math:`\omega` the
            argument of periapsis (radian), :math:`M_0` the mean
            anomaly (radian) :math:`n` the mean motion (rad/[time]) [2]
        """
        return StateVector([self.inclination,
                            self.longitude_ascending_node,
                            self.eccentricity,
                            self.argument_periapsis,
                            self.mean_anomaly,
                            self.mean_motion])

    @property
    def equinoctial_elements(self):
        r"""
        Returns
        -------
        : StateVector

            .. math::

                    X = [a, h, k, p, q, \lambda]^{T}

            where :math:`a` the semi-major axis ([length]), :math:`h`
            and :math:`k` are the horizontal and vertical components of
            the eccentricity respectively (unitless), :math:`p` and
            :math:`q` are the horizontal and vertical components of the
            inclination respectively (radian) and :math:`\lambda` is the
            mean longitude (radian) [3]

        """
        return StateVector([self.semimajor_axis,
                            self.equinoctial_h,
                            self.equinoctial_k,
                            self.equinoctial_p,
                            self.equinoctial_q,
                            self.mean_longitude])


class KeplerianOrbitalState(OrbitalState):
    r"""Merely a shell for the OrbitalState(coordinates='Keplerian')
    class, but includes some boundary checking. Construct via:

        .. math::

            X_{t_0} = [e, a, i, \Omega, \omega, \theta]^{T} \\

    where:
    :math:`e` is the orbital eccentricity (unitless),
    :math:`a` the semi-major axis ([length]),
    :math:`i` the inclination (radian),
    :math:`\Omega` is the longitude of the ascending node (radian),
    :math:`\omega` the argument of periapsis (radian), and
    :math:`\theta` the true anomaly (radian)

    """
    coordinates = Property(
        str, default='Keplerian', doc="Fixed as Keplerian coordinates"
    )

    def __init__(self, state_vector, *args, **kwargs):
        # Ensure that the coordinates keyword is set to 'Keplerian' and do some
        # additional checks.

        if np.less(state_vector[0, 0], 0.0) | np.greater(state_vector[0, 0],
                                                         1.0):
            raise ValueError("Eccentricity should be between 0 and 1: got {}"
                             .format(state_vector[0, 0]))
        if np.less(state_vector[2, 0], 0.0) | np.greater(state_vector[2, 0],
                                                         np.pi):
            raise ValueError("Inclination should be between 0 and pi: got {}"
                             .format(state_vector[2, 0]))
        if np.less(state_vector[3, 0], 0.0) | np.greater(state_vector[3, 0],
                                                         2 * np.pi):
            raise ValueError("Longitude of Ascending Node should be between 0 "
                             "and 2*pi: got {}"
                             .format(state_vector[3, 0]))
        if np.less(state_vector[4, 0], 0.0) | np.greater(state_vector[4, 0],
                                                         2 * np.pi):
            raise ValueError("Argument of Periapsis should be between 0 and "
                             "2*pi: got {}"
                             .format(state_vector[4, 0]))
        if np.less(state_vector[5, 0], 0.0) | np.greater(state_vector[5, 0],
                                                         2 * np.pi):
            raise ValueError("True Anomaly should be between 0 and 2*pi: got "
                             "{}"
                             .format(state_vector[5, 0]))

        # And go ahead and initialise as previously
        super().__init__(state_vector, coordinates='keplerian', *args,
                         **kwargs)


class TLEOrbitalState(OrbitalState):
    r"""For the TLE state vector:

        .. math::

            X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T}

    where :math:`i` the inclination (radian),
    :math:`\Omega` is the longitude of the ascending node (radian),
    :math:`e` is the orbital eccentricity (unitless),
    :math:`\omega` the argument of perigee (radian),
    :math:`M_0` the mean anomaly (radian) and
    :math:`n` the mean motion (radian / [time]).

    """
    coordinates = Property(
        str, default='TLE', doc="Fixed as TLE coordinates"
    )

    def __init__(self, state_vector, *args, **kwargs):
        if np.less(state_vector[2, 0], 0.0) | np.greater(state_vector[2, 0],
                                                         1.0):
            raise ValueError("Eccentricity should be between 0 and 1: got {}"
                             .format(state_vector[0, 0]))
        if np.less(state_vector[0, 0], 0.0) | np.greater(state_vector[0, 0],
                                                         np.pi):
            raise ValueError("Inclination should be between 0 and pi: got {}"
                             .format(state_vector[1, 0]))
        if np.less(state_vector[1, 0], 0.0) | np.greater(state_vector[1, 0],
                                                         2*np.pi):
            raise ValueError("Longitude of Ascending Node should be between 0 "
                             "and 2*pi: got {}"
                             .format(state_vector[2, 0]))
        if np.less(state_vector[3, 0], 0.0) | np.greater(state_vector[3, 0],
                                                         2*np.pi):
            raise ValueError("Argument of Periapsis should be between 0 and "
                             "2*pi: got {}"
                             .format(state_vector[3, 0]))
        if np.less(state_vector[4, 0], 0.0) | np.greater(state_vector[4, 0],
                                                         2*np.pi):
            raise ValueError("Mean Anomaly should be between 0 and 2*pi: got "
                             "{}"
                             .format(state_vector[4, 0]))

        super().__init__(state_vector, coordinates='TLE', *args, **kwargs)


class EquinoctialOrbitalState(OrbitalState):
    r"""For the Equinoctial state vector:

        .. math::

            X_{t_0} = [a, h, k, p, q, \lambda]^{T} \\

    where :math:`a` the semi-major axis ([length]),
    :math:`h` is the horizontal component of the eccentricity
    (unitless),
    :math:`k` is the vertical component of the eccentricity (unitless),
    :math:`q` is the horizontal component of the inclination (radian),
    :math:`k` is the vertical component of the inclination (radian),
    :math:`\lambda` is the mean longitude (radian)
    """

    coordinates = Property(
        str, default='Equinoctial', doc="Fixed as equinoctial coordinates"
    )

    def __init__(self, state_vector, *args, **kwargs):
        if np.less(state_vector[1, 0], -1.0) | np.greater(state_vector[1, 0],
                                                          1.0):
            raise ValueError("Horizontal Eccentricity should be between -1 "
                             "and 1: got {}"
                             .format(state_vector[1, 0]))
        if np.less(state_vector[2, 0], -1.0) | np.greater(state_vector[2, 0],
                                                          1.0):
            raise ValueError("Vertical Eccentricity should be between -1 and "
                             "1: got {}"
                             .format(state_vector[2, 0]))
        """Don't know where these next few lines came from. They're wrong.
        #if np.less(state_vector[3, 0], -1.0) | np.greater(state_vector[3, 0],
        1.0):
            raise ValueError("Horizontal Inclination should be between -1 and
            1: got {}"
                             .format(state_vector[3, 0]))
        if np.less(state_vector[4, 0], -1.0) | np.greater(state_vector[4, 0],
        1.0):
            raise ValueError("Vertical Inclination should be between -1 and -1:
            got {}"
                             .format(state_vector[4, 0]))"""
        if np.less(state_vector[5, 0], 0.0) | np.greater(state_vector[5, 0],
                                                         2*np.pi):
            raise ValueError("Mean Longitude should be between 0 and 2*pi: got"
                             " {}"
                             .format(state_vector[5, 0]))

        super().__init__(state_vector, coordinates='Equinoctial', *args,
                         **kwargs)
