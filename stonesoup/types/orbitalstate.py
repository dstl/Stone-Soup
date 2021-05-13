# -*- coding: utf-8 -*-
import datetime
from enum import Enum
from typing import Mapping, Any

import numpy as np


from ..base import Property
from ..functions.orbital import keplerian_to_rv, tru_anom_from_mean_anom
from .array import StateVector
from .state import State, GaussianState
from .angle import Inclination, EclipticLongitude


class CoordinateSystem(Enum):
    """Enumerates the allowable coordinate systems. See OrbitalState help for full explanation of
    what each of the elements does.
    """
    CARTESIAN = "Cartesian"
    KEPLERIAN = "Keplerian"
    TLE = "TLE"
    EQUINOCTIAL = "Equinoctial"

    # To allow case insensitivity and the use of "TwoLineElement" as a string to mean TLE
    @classmethod
    def _missing_(cls, value):

        for element in cls:
            if element.value.lower() == value.lower():
                return element
            if element.value == "TLE" and value.lower() == "twolineelement":
                return element

        raise ValueError("%r is not a valid %s" % (value, cls.__name__))


class OrbitalState(State):
    r"""The orbital state base type. This is the building block of Stone Soup's orbital inference
    routines and follows the principle that you shouldn't have to care which parameterisation you
    use. The class stores relevant information internally and undertakes whatever conversions are
    necessary.

    The :attr:`state_vector` is held as :math:`[\mathbf{r}, \dot{\mathbf{r}}]`, the "Orbital State
    Vector" (as traditionally understood in orbital mechanics), where :math:`\mathbf{r}` is the
    (3D) Cartesian position in the primary-centered inertial frame, while :math:`\dot{\mathbf{r}}`
    is the corresponding velocity vector. All other parameters are accessed via functions. Formulae
    for conversions are generally found in, or derived from [1]_.

    The gravitational parameter :math:`\mu = GM` can be defined. If left undefined it defaults to
    that of the Earth, :math:`3.986004418 \, (\pm \, 0.000000008) \times 10^{14} \mathrm{m}^3
    \mathrm{s}^{−2}`

    The object is constructed from the input vector :math:`X_{t_{0}}` at epoch
    :attr:`State.timestamp`, :math:`t_0`. The coordinates of :math:`X_{t_{0}}` are Cartesian
    Earth-Centered Inertial (ECI) [m] by default, but may be selected via the "coordinates" keyword
    by passing a :class:`~.CoordinateSystem` object, or an appropriate string. Allowable coordinate
    systems are,

        coordinates = "Cartesian", the input state vector is

             .. math::

                X_{t_0} = [r_x, r_y, r_z, \dot{r}_x, \dot{r}_y,
                    \dot{r}_z]^{T}

        where :math:`r_x, r_y, r_z` are the Cartesian position coordinates in the Primary-Centered
        Inertial frame and :math:`\dot{r}_x, \dot{r}_y, \dot{r}_z` are the corresponding velocity
        coordinates.

        coordinates = "Keplerian" (Keplarian elements), construct using input state vector:

            .. math::

                X_{t_0} = [e, a, i, \Omega, \omega, \theta]^{T} \\

        where:
        :math:`e` is the orbital eccentricity (unitless),
        :math:`a` the semi-major axis ([length]),
        :math:`i` the inclination (radian),
        :math:`\Omega` is the longitude of the ascending node (radian),
        :math:`\omega` the argument of periapsis (radian), and
        :math:`\theta` the true anomaly (radian)

        coordinates = "TLE" (Two-Line Elements [2]_), initiates using input vector

            .. math::

                X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T}

        where :math:`i` the inclination (radian),
        :math:`\Omega` is the longitude of the ascending node (radian),
        :math:`e` is the orbital eccentricity (unitless),
        :math:`\omega` the argument of perigee (radian),
        :math:`M_0` the mean anomaly (radian) and
        :math:`n` the mean motion (radian / [time]).

        This can also be constructed by passing `state_vector=None` and using the metadata. In this
        instance the metadata must conform to the TLE standard format [2]_ and be included in the
        metadata dictionary as 'line_1' and 'line_2'.

        coordinates = "Equinoctial" (equinoctial elements [3]_),

            .. math::

                X_{t_0} = [a, h, k, p, q, \lambda]^{T} \\

        where :math:`a` the semi-major axis ([length]),
        :math:`h` is the horizontal component of the eccentricity (unitless),
        :math:`k` is the vertical component of the eccentricity (unitless),
        :math:`q` is the horizontal component of the inclination (radian),
        :math:`k` is the vertical component of the inclination (radian),
        :math:`\lambda` is the mean longitude (radian)


    References
    ----------
    .. [1] Curtis, H.D. 2010, Orbital Mechanics for Engineering Students (3rd Ed), Elsevier
           Aerospace Engineering Series

    .. [2] NASA, Definition of Two-line Element Set Coordinate System, [spaceflight.nasa.gov](
           https://spaceflight.nasa.gov/realdata/sightings/SSapplications/Post/JavaSSOP/
           SSOP_Help/tle_def.html)

    .. [3] Broucke, R. A. & Cefola, P. J. 1972, Celestial Mechanics, Volume 5, Issue 3, pp. 303-310

    """

    coordinates: CoordinateSystem = Property(
        default=CoordinateSystem.CARTESIAN,
        doc="The parameterisation used on initiation. Acceptable values "
            "are 'CARTESIAN' (default), 'KEPLERIAN', 'TLE', or 'EQUINOCTIAL'. "
            "All other inputs will return errors. Will accept string inputs."
    )

    grav_parameter: float = Property(
        default=3.986004418e14,
        doc=r"Standard gravitational parameter :math:`\mu = G M`. The default "
            r"is :math:`3.986004418 \times 10^{14} \,` "
            r":math:`\mathrm{m}^3 \mathrm{s}^{-2}`.")

    metadata: Mapping[Any, Any] = Property(
        default=None, doc="Dictionary containing metadata about orbit"
    )

    def __init__(self, state_vector, *args, **kwargs):
        """"""
        if 'coordinates' in kwargs:
            coordinates = CoordinateSystem(kwargs['coordinates'])
        else:
            coordinates = CoordinateSystem.CARTESIAN

        # Check to see if the initialisation is via metadata
        if coordinates.name == 'TLE' and (state_vector is None or len(state_vector) == 0):
            if 'metadata' in kwargs and kwargs['metadata'] is not None:
                line1 = kwargs['metadata']['line_1']
                line2 = kwargs['metadata']['line_2']

                # Resolve the timestamp
                year = 2000 + int(line1[17:20])
                day = line1[20:23]

                hour = float(line1[23:32]) * 24
                fhour = int(np.floor(hour))

                minu = (hour - fhour) * 60
                fminu = int(np.floor(minu))

                seco = (minu - fminu) * 60
                fseco = int(np.floor(seco))

                mics = (seco - fseco) * 1e6
                fmics = int(np.round(mics))

                timestamp = datetime.datetime.strptime(
                    str(year) + " " + str(day) + " " + str(fhour) + " " + str(fminu) + " " +
                    str(fseco) + " " + str(
                        fmics), "%Y %j %H %M %S %f")

                state_vector = StateVector([float(line2[8:16]) * np.pi / 180,
                                            float(line2[17:25]) * np.pi / 180,
                                            float('.' + line2[26:33]),
                                            float(line2[34:42]) * np.pi / 180,
                                            float(line2[43:51]) * np.pi / 180,
                                            float(line2[52:63]) * 2 * np.pi / 86400])

                kwargs['timestamp'] = timestamp
                super().__init__(state_vector, *args, **kwargs)

            else:
                raise TypeError("State vector and metadata cannot both be empty")

        # Otherwise check that the state vector is the right size
        elif len(state_vector) != 6:
            raise ValueError(
                "State vector shape should be 6x1 : got {}".format(state_vector.shape))

        # Coordinate type checks
        if coordinates.name == 'CARTESIAN':
            super().__init__(state_vector, *args, **kwargs)

        elif coordinates.name == 'KEPLERIAN':

            if np.less(state_vector[0], 0.0) | np.greater(state_vector[0], 1.0):
                raise ValueError("Eccentricity should be between 0 and 1: got {}"
                                 .format(state_vector[0]))

            # And go ahead and initialise as previously
            super().__init__(state_vector, *args, **kwargs)

            # Convert Keplerian elements to Cartesian

            # First enforce the correct type
            state_vector[2] = Inclination(state_vector[2])
            state_vector[3] = EclipticLongitude(state_vector[3])
            state_vector[4] = EclipticLongitude(state_vector[4])
            state_vector[5] = EclipticLongitude(state_vector[5])

            self.state_vector = keplerian_to_rv(state_vector, grav_parameter=self.grav_parameter)

        elif coordinates.name == 'TLE':

            if np.less(state_vector[2], 0.0) | np.greater(state_vector[2], 1.0):
                raise ValueError("Eccentricity should be between 0 and 1: got {}"
                                 .format(state_vector[0]))

            super().__init__(state_vector, *args, **kwargs)

            # First enforce the correct type
            state_vector[0] = Inclination(state_vector[0])
            state_vector[1] = EclipticLongitude(state_vector[1])
            state_vector[3] = EclipticLongitude(state_vector[3])
            state_vector[4] = EclipticLongitude(state_vector[4])

            # Get the semi-major axis from the mean motion
            semimajor_axis = np.cbrt(self.grav_parameter / state_vector[5] ** 2)

            # True anomaly from mean anomaly
            tru_anom = tru_anom_from_mean_anom(state_vector[4], state_vector[2])

            # Use given and derived quantities to convert from Keplarian to
            # Cartesian
            self.state_vector = keplerian_to_rv(StateVector([state_vector[2], semimajor_axis,
                                                             state_vector[0], state_vector[1],
                                                             state_vector[3], tru_anom]),
                                                grav_parameter=self.grav_parameter)

        elif coordinates.name == 'EQUINOCTIAL':

            if np.less(state_vector[1], -1.0) | np.greater(state_vector[1], 1.0):
                raise ValueError("Horizontal Eccentricity should be between -1 "
                                 "and 1: got {}".format(state_vector[1]))
            if np.less(state_vector[2], -1.0) | np.greater(state_vector[2], 1.0):
                raise ValueError("Vertical Eccentricity should be between -1 and "
                                 "1: got {}".format(state_vector[2]))

            super().__init__(state_vector, *args, **kwargs)

            # First enforce the correct type for mean longitude
            state_vector[5] = EclipticLongitude(state_vector[5])

            # Calculate the Keplarian element quantities
            semimajor_axis = state_vector[0]
            raan = np.arctan2(state_vector[3], state_vector[4])
            inclination = 2 * np.arctan(state_vector[3] / np.sin(raan))
            arg_per = np.arctan2(state_vector[1], state_vector[2]) - raan
            mean_anomaly = state_vector[5] - arg_per - raan
            eccentricity = state_vector[1] / (np.sin(arg_per + raan))

            # True anomaly from mean anomaly
            tru_anom = tru_anom_from_mean_anom(mean_anomaly, eccentricity)

            # Convert from Keplarian to Cartesian
            self.state_vector = keplerian_to_rv(StateVector([eccentricity, semimajor_axis,
                                                             inclination, raan, arg_per,
                                                             tru_anom]),
                                                grav_parameter=self.grav_parameter)

    # Some vector quantities
    @property
    def _nodeline(self):
        """The vector node line (defines the longitude of the ascending node in the
        Primary-centered inertial frame)"""

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
        r""" The eccentricity vector :math:`\mathbf{e}`"""

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
        r"""The specific angular momentum, :math:`\mathbf{h}`"""
        return np.cross(self.state_vector[0:3], self.state_vector[3:6], axis=0)

    @property
    def cartesian_state_vector(self):
        r"""The state vector :math:`X_{t_0} = [r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^{T}`
        in 'Primary-Centred' Inertial coordinates, equivalent to ECI in the case of the Earth
        """
        return StateVector(self.state_vector)

    # Some scalar quantities
    @property
    def epoch(self):
        """The epoch, or state timestamp"""
        return self.timestamp

    @property
    def range(self):
        """The distance to object (from gravitational centre of primary)"""
        return np.sqrt(np.dot(self.state_vector[0:3].T,
                              self.state_vector[0:3])).item()

    @property
    def speed(self):
        """The current instantaneous speed (scalar)"""
        return np.sqrt(np.dot(self.state_vector[3:6].T,
                              self.state_vector[3:6]).item())

    @property
    def eccentricity(self):
        r"""The orbital eccentricity, :math:`e \; (0 \le e \le 1)`

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
        """The orbital semi-major axis"""
        return (self.mag_specific_angular_momentum**2 / self.grav_parameter) *\
               (1 / (1 - self.eccentricity**2))

        # Used to be this
        # return 1/((2/self.range) - (self.speed**2)/self.grav_parameter)

    @property
    def inclination(self):
        r"""Orbital inclination, :math:`i \; (0 \le i < \pi)`, [rad]"""
        boldh = self.specific_angular_momentum
        h = self.mag_specific_angular_momentum

        # Note no quadrant ambiguity
        return Inclination(np.arccos(np.clip(boldh[2].item()/h, -1, 1)))

    @property
    def longitude_ascending_node(self):
        r"""The longitude (or right ascension) of ascending node, :math:`\Omega \; (0 \leq \Omega <
        2\pi)`"""
        boldn = self._nodeline
        n = np.sqrt(np.dot(boldn.T, boldn).item())

        # Quadrant ambiguity
        if boldn[1].item() >= 0:
            return EclipticLongitude(np.arccos(np.clip(boldn[0].item()/n, -1, 1)))
        else:  # boldn[1].item() < 0:
            return EclipticLongitude(2*np.pi - np.arccos(np.clip(boldn[0].item()/n, -1, 1)))

    @property
    def argument_periapsis(self):
        r"""The argument of periapsis, :math:`\omega \; (0 \le \omega < 2\pi)` in radians"""
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
            return EclipticLongitude(np.arccos(np.clip(np.dot(boldn.T, bolde).item() /
                                                       (n * self.eccentricity), -1, 1)))
        else:  # bolde[2].item() < 0:
            return EclipticLongitude(2*np.pi - np.arccos(np.clip(np.dot(boldn.T, bolde).item() /
                                                                 (n * self.eccentricity), -1, 1)))

    @property
    def true_anomaly(self):
        r"""The true anomaly, :math:`\theta \; (0 \le \theta < 2\pi)` in radians"""
        # Resolve the quadrant ambiguity.The clip function is required to
        # mitigate against floating-point errors which push the ratio outside
        # the -1,1 region.
        radial_velocity = np.dot(self.state_vector[0:3].T,
                                 self.state_vector[3:6]).item() / self.speed

        if radial_velocity >= 0:
            return EclipticLongitude(np.arccos(np.clip(
                np.dot(self._eccentricity_vector.T / self.eccentricity, self.state_vector[0:3] /
                       self.range).item(), -1, 1)))
        else:  # radial_velocity < 0:
            return EclipticLongitude(2*np.pi - np.arccos(np.clip(
                np.dot(self._eccentricity_vector.T / self.eccentricity, self.state_vector[0:3] /
                       self.range).item(), -1, 1)))

    @property
    def eccentric_anomaly(self):
        r"""The eccentric anomaly, :math:`E \; (0 \le E < 2\pi)` in radians

        Note
        ----
            This computes the quantity exactly via the Keplerian
            eccentricity and true anomaly rather than via the mean
            anomaly using an iterative procedure.

        """
        return EclipticLongitude(np.remainder(2 * np.arctan(np.sqrt((1 - self.eccentricity) /
                                                                    (1 + self.eccentricity)) *
                                                            np.tan(self.true_anomaly / 2)),
                                              2*np.pi))

    @property
    def mean_anomaly(self):
        r"""Mean anomaly, :math:`M \; (0 \le M < 2\pi`), in radians

        Note
        ----
            Uses the eccentric anomaly and Kepler's equation to get
            mean anomaly from true anomaly and eccentricity.

        """

        return EclipticLongitude(self.eccentric_anomaly - self.eccentricity *
                                 np.sin(self.eccentric_anomaly))  # Kepler's equation

    @property
    def period(self):
        """Orbital period, :math:`T` ([time])"""
        return ((2 * np.pi) / np.sqrt(self.grav_parameter)) * \
            np.power(self.semimajor_axis, 3 / 2)

    @property
    def mean_motion(self):
        r"""The mean motion, :math:`\frac{2 \pi}{T}`, where :math:`T` is the period, (rad / [time])
        """
        return 2 * np.pi / self.period

    @property
    def mag_specific_angular_momentum(self):
        """The magnitude of the specific angular momentum, :math:`h`"""
        boldh = self.specific_angular_momentum
        return np.sqrt(np.dot(boldh.T, boldh).item())

        # Alternative via scalars
        # return np.sqrt(self.grav_parameter * self.semimajor_axis *
        # (1 - self.eccentricity ** 2))

    @property
    def specific_orbital_energy(self):
        r"""Specific orbital energy (:math:`\frac{-GM}{2a}`)"""
        return -self.grav_parameter / (2 * self.semimajor_axis)

    @property
    def equinoctial_h(self):
        r"""The horizontal component of the eccentricity in equinoctial coordinates is
        :math:`h = e \sin (\omega + \Omega)`"""

        return self.eccentricity * np.sin(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinoctial_k(self):
        r"""The vertical component of the eccentricity in equinoctial coordinates is
        :math:`k = e \cos (\omega + \Omega)`"""

        return self.eccentricity * np.cos(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinoctial_p(self):
        r"""The horizontal component of the inclination in equinoctial coordinates is
        :math:`p = \tan (i/2) \sin \Omega`"""
        return np.tan(self.inclination/2) * \
            np.sin(self.longitude_ascending_node)

    @property
    def equinoctial_q(self):
        r"""The vertical component of the inclination in equinoctial coordinates is
        :math:`q = \tan (i/2) \cos \Omega`"""
        return np.tan(self.inclination / 2) * np.cos(self.longitude_ascending_node)

    @property
    def mean_longitude(self):
        r"""The mean longitude, defined as :math:`\lambda = M_0 + \omega + \Omega` (rad)"""
        return EclipticLongitude(self.mean_anomaly + self.argument_periapsis +
                                 self.longitude_ascending_node)

    # The following return vectors of complete sets of elements
    @property
    def keplerian_elements(self):
        r"""The vector of Keplerian elements :math:`X = [e, a, i, \Omega, \omega, \theta]^{T}`
        where :math:`e` is the orbital eccentricity (unitless), :math:`a` the semi-major axis
        ([length]), :math:`i` the inclination (radian), :math:`\Omega` is the longitude of the
        ascending node (radian), :math:`\omega` the argument of periapsis (radian), and
        :math:`\theta` the true anomaly (radian)"""
        return StateVector(np.array([[self.eccentricity],
                           [self.semimajor_axis],
                           [self.inclination],
                           [self.longitude_ascending_node],
                           [self.argument_periapsis],
                           [self.true_anomaly]]))

    @property
    def two_line_element(self):
        r"""The Two-Line Element vector :math:`X = [i, \Omega, e, \omega, M_0, n]^{T}` where
        :math:`i` the inclination (radian) :math:`\Omega` is the longitude of the ascending node
        (radian), :math:`e` is the orbital eccentricity (unitless), :math:`\omega` the argument of
        periapsis (radian), :math:`M_0` the mean anomaly (radian) :math:`n` the mean motion
        (rad/[time]) [2]_"""
        return StateVector(np.array([[self.inclination],
                           [self.longitude_ascending_node],
                           [self.eccentricity],
                           [self.argument_periapsis],
                           [self.mean_anomaly],
                           [self.mean_motion]]))

    @property
    def equinoctial_elements(self):
        r"""The equinoctial elements, :math:`X = [a, h, k, p, q, \lambda]^{T}` where :math:`a` the
        semi-major axis ([length]), :math:`h` and :math:`k` are the horizontal and vertical
        components of the eccentricity respectively (unitless), :math:`p` and :math:`q` are the
        horizontal and vertical components of the inclination respectively (radian) and
        :math:`\lambda` is the mean longitude (radian) [3]_
        """
        return StateVector(np.array([[self.semimajor_axis],
                           [self.equinoctial_h],
                           [self.equinoctial_k],
                           [self.equinoctial_p],
                           [self.equinoctial_q],
                           [self.mean_longitude]]))


class GaussianOrbitalState(GaussianState, OrbitalState):
    """An Orbital state for use in Kalman filters (and perhaps elsewhere). Inherits from
    GaussianState so has covariance matrix. As no checks on the validity of the covariance
    matrix are made, care should be exercised in its use. The propagator will generally require
    a particular coordinate reference which must be understood.

    All methods provided by OrbitalState are available.

    """
