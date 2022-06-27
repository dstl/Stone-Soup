# -*- coding: utf-8 -*-
import datetime
from enum import Enum
from typing import Mapping, Any

import numpy as np


from ..base import Property
from ..functions.orbital import keplerian_to_rv, tru_anom_from_mean_anom
from . import Type
from .array import StateVector
from .state import State, GaussianState, ParticleState
from .angle import Inclination, EclipticLongitude
from ..reader.astronomical import TLEDictReader


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


class Orbital(Type):
    r"""The orbital base type. This is the building block of Stone Soup's orbital inference
    routines and follows the principle that you shouldn't have to care which parameterisation you
    use. The class stores relevant information internally and undertakes whatever conversions are
    necessary.

    The gravitational parameter :math:`\mu = GM` can be defined. If left undefined it defaults to
    that of the Earth, :math:`3.986004418 \, (\pm \, 0.000000008) \times 10^{14} \mathrm{m}^3
    \mathrm{s}^{−2}`

    An orbital state is constructed from the input vector :math:`X_{t_{0}}` at epoch
    :attr:`State.timestamp`, :math:`t_0`. The coordinates of :math:`X_{t_{0}}` are Cartesian
    Earth-Centered Inertial (ECI) [m] by default, but may be selected via the "coordinates" keyword
    by passing a :class:`~.CoordinateSystem` object, or an appropriate string. Allowable coordinate
    systems are:

        Coordinates = "Cartesian", the input state vector is

             .. math::

                X_{t_0} = [r_x, r_y, r_z, \dot{r}_x, \dot{r}_y,
                    \dot{r}_z]^{T}

        where :math:`r_x, r_y, r_z` are the Cartesian position coordinates in the Primary-Centered
        Inertial frame and :math:`\dot{r}_x, \dot{r}_y, \dot{r}_z` are the corresponding velocity
        coordinates.

        Coordinates = "Keplerian" (Keplarian elements), construct using input state vector:

            .. math::

                X_{t_0} = [e, a, i, \Omega, \omega, \theta]^{T} \\

        where:
        :math:`e` is the orbital eccentricity (unitless),
        :math:`a` the semi-major axis ([length]),
        :math:`i` the inclination (radian),
        :math:`\Omega` is the longitude of the ascending node (radian),
        :math:`\omega` the argument of periapsis (radian), and
        :math:`\theta` the true anomaly (radian).

        Coordinates = "TLE" (Two-Line Elements [1]_), initiates using input vector

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

        Coordinates = "Equinoctial" (equinoctial elements [2]_),

            .. math::

                X_{t_0} = [a, h, k, p, q, \lambda]^{T} \\

        where :math:`a` the semi-major axis ([length]),
        :math:`h` is the horizontal component of the eccentricity (unitless),
        :math:`k` is the vertical component of the eccentricity (unitless),
        :math:`q` is the horizontal component of the inclination (radian),
        :math:`k` is the vertical component of the inclination (radian),
        :math:`\lambda` is the mean longitude (radian).


    References
    ----------
    .. [1] NASA, Definition of Two-line Element Set Coordinate System, [spaceflight.nasa.gov](
           https://spaceflight.nasa.gov/realdata/sightings/SSapplications/Post/JavaSSOP/
           SSOP_Help/tle_def.html)

    .. [2] Broucke, R. A. & Cefola, P. J. 1972, Celestial Mechanics, Volume 5, Issue 3, pp. 303-310

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

    # The following nine attributes provide support for two-line element representations
    catalogue_number: int = Property(
        default=None, doc="NORAD Catalog Number: a unique identifier for each earth-orbiting "
                          "artificial satellite")

    classification: str = Property(
        default=None, doc="Classification (U=Unclassified, C=Classified, S=Secret")

    international_designator: str = Property(
        default=None, doc="International designator incorporates the year of launch, launch "
                          "number that year and place of launch.")

    ballistic_coefficient: float = Property(
        default=None, doc=r"The ballistic coefficient is the first derivative of the mean "
                          r"motion. (units of :math:`mathrm{rad s}^{-2}`)")

    second_derivative_mean_motion: float = Property(
        default=None, doc=r"The second derivative of the mean motion. "
                          r"(:math:`mathrm{rad s}^{-3}`)")

    bstar: float = Property(
        default=None, doc=r"The TLE drag coefficient. :math:`B* = \frac{B \rho_0}{2}` where "
                          r":math:`\rho_0` is density of a standard atmosphere and "
                          r":math:B = \frac{C_D A}{m}` for coefficient of drag :math:`C_D`, "
                          r"cross-sectional area :math:`A` and mass :math:`m` is the mass.")

    ephemeris_type: int = Property(
        default=None, doc="Ephemeris type (NORAD use). Zero in distributed TLE data.")

    element_set_number: int = Property(
        default=None, doc="Element set number in the TLE. Incremented when a new TLE is "
                          "generated for this object.")

    revolution_number: int = Property(default=None, doc="Number of revolutions at the epoch")

    metadata: dict = Property(default=None, doc="Dictionary containing metadata about orbit")

    def __init__(self, state_vector, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)

        if 'coordinates' in kwargs:
            coordinates = CoordinateSystem(kwargs['coordinates'])
        else:
            coordinates = CoordinateSystem.CARTESIAN

        # Check to see if the initialisation is via metadata
        if coordinates.name == 'TLE' and \
                (self.state_vector is None or len(self.state_vector) == 0):
            if 'metadata' in kwargs and kwargs['metadata'] is not None:
                tle = TLEDictReader({'line_1': kwargs['metadata']['line_1'],
                                     'line_2': kwargs['metadata']['line_2']})

                self.state_vector = StateVector([tle.inclination, tle.longitude_of_ascending_node,
                                            tle.eccentricity, tle.arg_periapsis, tle.mean_anomaly,
                                            tle.mean_motion])
                kwargs['timestamp'] = tle.epoch

                #super().__init__(state_vector, *args, **kwargs)

            else:
                raise TypeError("State vector and metadata cannot both be empty")

        # Otherwise check that the state vector is the right size
        elif len(self.state_vector) != 6:
            raise ValueError(
                "State vector shape should be 6x1 : got {}".format(self.state_vector.shape))

        # Coordinate type checks
        if coordinates.name == 'CARTESIAN':
            pass

        elif coordinates.name == 'KEPLERIAN':

            if np.less(self.state_vector[0], 0.0) | np.greater(self.state_vector[0], 1.0):
                raise ValueError("Eccentricity should be between 0 and 1: got {}"
                                 .format(self.state_vector[0]))

            # Convert Keplerian elements to Cartesian

            # First enforce the correct type
            self.state_vector[2] = Inclination(self.state_vector[2])
            self.state_vector[3] = EclipticLongitude(self.state_vector[3])
            self.state_vector[4] = EclipticLongitude(self.state_vector[4])
            self.state_vector[5] = EclipticLongitude(self.state_vector[5])

            self.state_vector = keplerian_to_rv(self.state_vector,
                                                grav_parameter=self.grav_parameter)

        elif coordinates.name == 'TLE':

            if np.less(self.state_vector[2], 0.0) | np.greater(self.state_vector[2], 1.0):
                raise ValueError("Eccentricity should be between 0 and 1: got {}"
                                 .format(self.state_vector[0]))

            #super().__init__(state_vector, *args, **kwargs)

            if 'metadata' in kwargs and kwargs['metadata']:
                tle = TLEDictReader({'line_1': kwargs['metadata']['line_1'],
                                     'line_2': kwargs['metadata']['line_2']})

                self.catalogue_number = tle.catalogue_number
                self.classification = tle.classification
                self.international_designator = tle.international_designator
                self.ballistic_coefficient = tle.ballistic_coefficient
                self.second_derivative_mean_motion = tle.second_derivative_mean_motion
                self.bstar = tle.bstar
                self.ephemeris_type = tle.ephemeris_type
                self.element_set_number = tle.element_set_number
                self.revolution_number = tle.revolution_number

            # First enforce the correct type
            self.state_vector[0] = Inclination(self.state_vector[0])
            self.state_vector[1] = EclipticLongitude(self.state_vector[1])
            self.state_vector[3] = EclipticLongitude(self.state_vector[3])
            self.state_vector[4] = EclipticLongitude(self.state_vector[4])

            # Get the semi-major axis from the mean motion
            semimajor_axis = np.cbrt(self.grav_parameter / self.state_vector[5] ** 2)

            # True anomaly from mean anomaly
            tru_anom = tru_anom_from_mean_anom(self.state_vector[4], self.state_vector[2])

            # Use given and derived quantities to convert from Keplarian to
            # Cartesian
            self.state_vector = keplerian_to_rv(StateVector([self.state_vector[2], semimajor_axis,
                                                             self.state_vector[0],
                                                             self.state_vector[1],
                                                             self.state_vector[3], tru_anom]),
                                                grav_parameter=self.grav_parameter)

        elif coordinates.name == 'EQUINOCTIAL':

            if np.less(self.state_vector[1], -1.0) | np.greater(self.state_vector[1], 1.0):
                raise ValueError("Horizontal Eccentricity should be between -1 "
                                 "and 1: got {}".format(self.state_vector[1]))
            if np.less(self.state_vector[2], -1.0) | np.greater(self.state_vector[2], 1.0):
                raise ValueError("Vertical Eccentricity should be between -1 and "
                                 "1: got {}".format(self.state_vector[2]))

            # First enforce the correct type for mean longitude
            self.state_vector[5] = EclipticLongitude(self.state_vector[5])

            # Calculate the Keplarian element quantities
            semimajor_axis = self.state_vector[0]
            raan = np.arctan2(self.state_vector[3], self.state_vector[4])
            inclination = 2 * np.arctan(self.state_vector[3] / np.sin(raan))
            arg_per = np.arctan2(self.state_vector[1], self.state_vector[2]) - raan
            mean_anomaly = self.state_vector[5] - arg_per - raan
            eccentricity = self.state_vector[1] / (np.sin(arg_per + raan))

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
                                 self.state_vector[3:6]).item() / rang

        return (1 / self.grav_parameter) * ((speed ** 2 - self.grav_parameter / rang)
                                            * self.state_vector[0:3] - rang *
                                            radial_velocity *
                                            self.state_vector[3:6])

    @property
    def specific_angular_momentum(self):
        r"""The specific angular momentum, :math:`\mathbf{h}`."""
        return np.cross(self.state_vector[0:3], self.state_vector[3:6], axis=0)

    @property
    def cartesian_state_vector(self):
        r"""The state vector :math:`X_{t_0} = [r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^{T}`
        in 'Primary-Centred' Inertial coordinates, equivalent to ECI in the case of the Earth.
        """
        return StateVector(self.state_vector)

    # Some scalar quantities
    @property
    def epoch(self):
        """The epoch, or state timestamp."""
        return self.timestamp

    @property
    def range(self):
        """The distance to object (from gravitational centre of primary)."""
        return np.sqrt(np.dot(self.state_vector[0:3].T,
                              self.state_vector[0:3])).item()

    @property
    def speed(self):
        """The current instantaneous speed (scalar)."""
        return np.sqrt(np.dot(self.state_vector[3:6].T,
                              self.state_vector[3:6]).item())

    @property
    def eccentricity(self):
        r"""The orbital eccentricity, :math:`e \; (0 \le e \le 1)`.

        Note
        ----
        This version of the calculation uses a form dependent only on scalars.

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
        """The orbital semi-major axis."""
        return (self.mag_specific_angular_momentum ** 2 / self.grav_parameter) * \
               (1 / (1 - self.eccentricity ** 2))

        # Used to be this
        # return 1/((2/self.range) - (self.speed**2)/self.grav_parameter)

    @property
    def inclination(self):
        r"""Orbital inclination, :math:`i \; (0 \le i < \pi)`, [rad]."""
        boldh = self.specific_angular_momentum
        h = self.mag_specific_angular_momentum

        # Note no quadrant ambiguity
        return Inclination(np.arccos(np.clip(boldh[2].item() / h, -1, 1)))

    @property
    def longitude_ascending_node(self):
        r"""The longitude (or right ascension) of ascending node, :math:`\Omega \; (0 \leq \Omega <
        2\pi)`."""
        boldn = self._nodeline
        n = np.sqrt(np.dot(boldn.T, boldn).item())

        # Quadrant ambiguity
        if boldn[1].item() >= 0:
            return EclipticLongitude(np.arccos(np.clip(boldn[0].item() / n, -1, 1)))
        else:  # boldn[1].item() < 0:
            return EclipticLongitude(2 * np.pi - np.arccos(np.clip(boldn[0].item() / n, -1, 1)))

    @property
    def argument_periapsis(self):
        r"""The argument of periapsis, :math:`\omega \; (0 \le \omega < 2\pi)` in radians."""
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
            return EclipticLongitude(2 * np.pi - np.arccos(np.clip(
                np.dot(boldn.T, bolde).item() / (n * self.eccentricity), -1, 1)))

    @property
    def true_anomaly(self):
        r"""The true anomaly, :math:`\theta \; (0 \le \theta < 2\pi)` in radians."""
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
            return EclipticLongitude(2 * np.pi - np.arccos(np.clip(
                np.dot(self._eccentricity_vector.T / self.eccentricity, self.state_vector[0:3] /
                       self.range).item(), -1, 1)))

    @property
    def eccentric_anomaly(self):
        r"""The eccentric anomaly, :math:`E \; (0 \le E < 2\pi)` in radians.

        Note
        ----
            This computes the quantity exactly via the Keplerian
            eccentricity and true anomaly rather than via the mean
            anomaly using an iterative procedure.

        """
        return EclipticLongitude(np.remainder(2 * np.arctan(np.sqrt((1 - self.eccentricity) /
                                                                    (1 + self.eccentricity)) *
                                                            np.tan(self.true_anomaly / 2)),
                                              2 * np.pi))

    @property
    def mean_anomaly(self):
        r"""Mean anomaly, :math:`M \; (0 \le M < 2\pi`), in radians.

        Note
        ----
            Uses the eccentric anomaly and Kepler's equation to get
            mean anomaly from true anomaly and eccentricity.

        """

        return EclipticLongitude(self.eccentric_anomaly - self.eccentricity *
                                 np.sin(self.eccentric_anomaly))  # Kepler's equation

    @property
    def period(self):
        """Orbital period, :math:`T` ([time])."""
        return ((2 * np.pi) / np.sqrt(self.grav_parameter)) * \
            np.power(self.semimajor_axis, 3 / 2)

    @property
    def mean_motion(self):
        r"""The mean motion, :math:`\frac{2 \pi}{T}`, where :math:`T` is the period, (rad / [time]).
        """
        return 2 * np.pi / self.period

    @property
    def mag_specific_angular_momentum(self):
        """The magnitude of the specific angular momentum, :math:`h`."""
        boldh = self.specific_angular_momentum
        return np.sqrt(np.dot(boldh.T, boldh).item())

        # Alternative via scalars
        # return np.sqrt(self.grav_parameter * self.semimajor_axis *
        # (1 - self.eccentricity ** 2))

    @property
    def specific_orbital_energy(self):
        r"""Specific orbital energy (:math:`\frac{-GM}{2a}`)."""
        return -self.grav_parameter / (2 * self.semimajor_axis)

    @property
    def equinoctial_h(self):
        r"""The horizontal component of the eccentricity in equinoctial coordinates is
        :math:`h = e \sin (\omega + \Omega)`."""

        return self.eccentricity * np.sin(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinoctial_k(self):
        r"""The vertical component of the eccentricity in equinoctial coordinates is
        :math:`k = e \cos (\omega + \Omega)`."""

        return self.eccentricity * np.cos(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinoctial_p(self):
        r"""The horizontal component of the inclination in equinoctial coordinates is
        :math:`p = \tan (i/2) \sin \Omega`."""
        return np.tan(self.inclination / 2) * \
            np.sin(self.longitude_ascending_node)

    @property
    def equinoctial_q(self):
        r"""The vertical component of the inclination in equinoctial coordinates is
        :math:`q = \tan (i/2) \cos \Omega`."""
        return np.tan(self.inclination / 2) * np.cos(self.longitude_ascending_node)

    @property
    def mean_longitude(self):
        r"""The mean longitude, defined as :math:`\lambda = M_0 + \omega + \Omega` (rad)."""
        return EclipticLongitude(self.mean_anomaly + self.argument_periapsis +
                                 self.longitude_ascending_node)

    # The following return vectors of complete sets of elements
    @property
    def keplerian_elements(self):
        r"""The vector of Keplerian elements :math:`X = [e, a, i, \Omega, \omega, \theta]^{T}`
        where :math:`e` is the orbital eccentricity (unitless), :math:`a` the semi-major axis
        ([length]), :math:`i` the inclination (radian), :math:`\Omega` is the longitude of the
        ascending node (radian), :math:`\omega` the argument of periapsis (radian), and
        :math:`\theta` the true anomaly (radian)."""
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
        (rad/[time]). [2]_"""
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
        :math:`\lambda` is the mean longitude (radian). [3]_
        """
        return StateVector(np.array([[self.semimajor_axis],
                                     [self.equinoctial_h],
                                     [self.equinoctial_k],
                                     [self.equinoctial_p],
                                     [self.equinoctial_q],
                                     [self.mean_longitude]]))

    @property
    def tle_dict(self):
        """Return the two-line elements as metadata. There's considerable variability within
        distributed TLE catalogues with inconsistent leading 0s, leading + signs and signs of
        zero exponents. This method tries to follow the practice in more recent CelesTrak TLEs and
        returns no leading 0, removes leading + signs and fixes the sign of zero exponents as +.
        Note that this means that checksums occasionally don't match.
        """

        def _tlefmt1(number):
            """TLE format for ballistic coefficient. Drops leading 0 but retains decimal point"""
            nstr = f"{number:8.8f}"
            if number < 0:
                return nstr[0] + nstr[2:]
            else:
                return ' ' + nstr[1:]

        def _tlefmt2(number):
            """TLE format for second derivative mean motion and B*"""
            if number == 0:
                return ' 00000+0'
            else:
                nstr = '{:5.5e}'.format(number)
                mantissa, exponent = nstr.split("e")
                outstr = f"{float(mantissa) / 10:5.5f}" + f"{int(exponent) + 1:+1.0f}"
                if number < 0:
                    return outstr[0] + outstr[3:]
                else:
                    return ' ' + outstr[2:]

        tst = self.timestamp
        timest = str(tst.year)[2:4] + str(tst.timetuple().tm_yday + tst.hour / 24 + tst.minute /
                                          (60 * 24) + tst.second / (3600 * 24) + tst.microsecond /
                                          (1e6 * 3600 * 24))

        line1 = "1 " + f"{self.catalogue_number:5}" + self.classification + ' ' + \
                f"{self.international_designator:8}" + ' ' + f"{float(timest):014.8f}" + ' ' + \
                _tlefmt1(self.ballistic_coefficient/(4 * np.pi) * 86400**2) + ' ' + \
                _tlefmt2(self.second_derivative_mean_motion/(6 * np.pi) * 86400**3) + ' ' + \
                _tlefmt2(self.bstar * 6.371e6) + ' ' + f"{self.ephemeris_type:1}" + ' ' + \
                f"{self.element_set_number:4}"

        line2 = "2 " + f"{self.catalogue_number:5}" + ' ' + f"{self.inclination*180/np.pi:8.4f}" \
                + ' ' + f"{self.longitude_ascending_node*180/np.pi:8.4f}" + ' ' \
                + f"{self.eccentricity:7.7f}"[2:] + ' ' \
                + f"{self.argument_periapsis*180/np.pi:8.4f}" + ' ' \
                + f"{self.mean_anomaly*180/np.pi:8.4f}" + ' ' \
                + f"{self.mean_motion/(2*np.pi) *3600*24:11.8f}" + f"{self.revolution_number:5.0f}"

        line1 = line1 + str(TLEDictReader.checksum(line1))
        line2 = line2 + str(TLEDictReader.checksum(line2))

        return {'line_1': line1, 'line_2': line2}


class OrbitalState(Orbital, State):
    r"""The orbital state class which inherits from :class:`~.Orbital` and :class:`~.State`.
    The :attr:`state_vector` is held as :math:`[\mathbf{r}, \dot{\mathbf{r}}]`, the "Orbital State
    Vector" (as traditionally understood in orbital mechanics), where :math:`\mathbf{r}` is the
    (3D) Cartesian position in the primary-centered inertial frame, while :math:`\dot{\mathbf{r}}`
    is the corresponding velocity vector. All methods provided by :class:`~.Orbital` are available.
    Formulae for conversions are generally found in, or derived from [3]_.
    References
    ----------
    .. [3] Curtis, H.D. 2010, Orbital Mechanics for Engineering Students (3rd Ed), Elsevier
           Aerospace Engineering Series
    """


class GaussianOrbitalState(Orbital, GaussianState):
    """An orbital state for use in Kalman filters (and perhaps elsewhere). Inherits from
    GaussianState so has a covariance matrix. As no checks on the validity of the covariance
    matrix are made, care should be exercised in its use. The propagator will generally require
    a particular coordinate reference which must be understood.
    All methods provided by :class:`~.Orbital` are available.
    """


class ParticleOrbitalState(Orbital, ParticleState):
    """An orbital state for use in Particle filters and SMC estimation. Inherits from
    ParticleState. Again, no checks on the validity are made. The propagator will generally require
    a particular coordinate reference which must be understood.

    All methods provided by :class:`~.Orbital` are available.

    """