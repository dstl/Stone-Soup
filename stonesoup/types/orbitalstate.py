# -*- coding: utf-8 -*-
import datetime
import numpy as np

from ..base import Property
from .state import State


class OrbitalState(State):

    r"""The orbital state base type. This is the building block of the
    orbital branch and follows the principle that you shouldn't have to
    worry too much what parameterisation you're using, the object
    stores the relevant information internally and can cope with
    whatever conversions are necessary.

    The :attr:`State.state_vector` is held as :math:`[\mathbf{r},
    \mathbf{v}]`, the "Orbital State Vector" as traditionally
    understood in orbital mechanics, where :math:`\mathbf{r}` is the
    Cartesian position in the primary-centered inertial frame while
    :math:`\mathbf{v}` is the corresponding velocity vector. All other
    parameters are accessed via functions.

    Construct using the appropriate :attr:`State.state_vector`
    :math:`X_{t_{0}}` at epoch :attr:`State.timestamp` :math:`t_0` and
    by way of keywords, via:
        coordinates = "Cartesian" (the orbital state vector),
        coordinates = "Keplerian" (Keplarian elements),
        coordinates = "TLE" (Two-Line elements) or
        coordinates = "Equinoctial" (equinoctial elements).

    The gravitational parameter :math:`GM` can be defined. If left
    undefined it defaults to that of the Earth, :math:`3.986004418
    (\pm 0.000000008) \\times 10^{14} \mathrm{m}^3 \mathrm{s}^{−2}`.



    :reference: Curtis, H.D. 2010, Orbital Mechanics for Engineering
    Students (3rd Ed), Elsevier Aerospace Engineering Series
    """

    grav_parameter = Property(
        float, default=3.986004418e14,
        doc="Standard gravitational parameter :math:`\\mu = G M`")

    metadata = Property(
        {}, default={}, doc="Dictionary containing metadata about orbit")

    def __init__(self, state_vector, coordinates='Cartesian', *args, **kwargs):
        r"""Can be initialised in a number of different ways according to
        preference.

        Parameters
        ----------
        state_vector : :class:`numpy.ndarray`
            The input vector whose elements depend on the parameterisation
            used. See 'Keywords' below.

        Keywords
        --------
        coordinates
            The chosen input coordinate frame. Can be 'Cartesian', 'Keplerian',
            'TLE' or 'Equinoctal'.


        Returns
        -------
        Constructs the class

        """
        if len(state_vector) != 6:
            raise ValueError("State vector shape should be 6x1 : got {}".format(
                state_vector.shape))
        super().__init__(state_vector, *args, **kwargs)

        if coordinates.lower() == 'cartesian':
            r"""The orbital state vector.
            
            Parameters
            ----------
            state_vector : numpy.ndarray
            
                .. math::
            
                    X_{t_0} = [r_x, r_y, r_z, \odot{r}_x, \odot{r}_y, 
                    \odot{r}_z]
        
                where:
                :math:`r_x, r_y, r_z` is the Cartesian position coordinate in 
                Earth Centered Inertial (ECI) coordinates
                :math:`\odot{r}_x, \odot{r}_y, \odot{r}_z` is the velocity 
                coordinate in ECI coordinates
                
            """

            #  No need to do any conversions here
            self.state_vector = state_vector

        elif coordinates.lower() == 'keplerian':
            r"""The state vector input should have the following input: 
            
            Parameters
            ----------
            state_vector : numpy.ndarray
            
                .. math::

                    X_{t_0} = [e, a, i, \Omega, \omega, \\theta]^{T} \\

                where:
                :math:`e` is the orbital eccentricity (unitless),
                :math:`a` the semi-major axis (m),
                :math:`i` the inclination (rad),
                :math:`\Omega` is the longitude of the ascending node (rad),
                :math:`\omega` the argument of periapsis (rad), and
                :math:`\\theta` the true anomaly (rad).
                
            """

            self.state_vector = self._keplerian_to_rv(state_vector)

        elif coordinates.upper() == 'TLE':
            r"""For the TLE state vector, the structure reflects the data 
                format that's been in use since the 1960s:
                
            Parameters
            ----------
            state_vector : numpy.ndarray()
            
                The two line element input vector

                .. math::

                    X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T} \\

                where :math:`i` the inclination (rad),
                :math:`\Omega` is the longitude of the ascending node (rad),
                :math:`e` is the orbital eccentricity (unitless),
                :math:`\omega` the argument of perigee (rad),
                :math:'M_0' the mean anomaly (rad)
                :math:`n` the mean motion (rad)
            
            Reference
            ---------
            https://spaceflight.nasa.gov/realdata/sightings/SSapplications/
            Post/JavaSSOP/SSOP_Help/tle_def.html
            
            """
            # Get the semi-major axis from the mean motion
            semimajor_axis = np.cbrt(self.grav_parameter/state_vector[5][0]**2)

            # True anomaly from mean anomaly
            tru_anom = self._tru_anom_from_mean_anom(state_vector[4][0], state_vector[2][0])

            return self._keplarian_to_rv(np.array[[state_vector[2][0]],
                                                 [semimajor_axis],
                                                 [state_vector[0][0]],
                                                 [state_vector[1][0]],
                                                 [state_vector[3][0]],
                                                 [tru_anom]])


        elif coordinates.lower() == 'equinoctial':
            """A version of the Equinoctial state vector as input:
            
            Parameters
            ----------
            state_vector : numpy.ndarray

                .. math::

                    X_{t_0} = [a, h, k, p, q, \lambda]^{T} \\

                where :math:`a` the semi-major axis (m),
                :math:`h` is the horizontal component of the eccentricity :math:`e`,
                :math:`k` is the vertical component of the eccentricity :math:`e`,
                :math:`p` is the horizontal component of the inclination :math:`i`,
                :math:`q` is the vertical component of the inclination :math:`i` and
                :math:'lambda' is the mean longitude
                
            Returns 
            -------
            Initialises the state vector
                
            Reference
            ---------
            Broucke, R. A. & Cefola, P. J. 1972, Celestial Mechanics, Volume 5, Issue 3, pp.303-310
                
            """

            semimajor_axis = state_vector[0][0]
            raan = np.arctan2(state_vector[3][0], state_vector[4][0])
            inclination = 2*np.arctan(state_vector[3][0]/np.sin(raan))
            arg_per = np.arctan2(state_vector[3][0], state_vector[2][0]) - raan
            mean_anomaly = state_vector[5][0] - arg_per - raan
            eccentricity = state_vector[1][0]/(np.sin(arg_per + raan))

            # True anomaly from mean anomaly
            tru_anom = self._tru_anom_from_mean_anom(mean_anomaly,
                                                     eccentricity)

            return self._keplarian_to_rv(np.array[[eccentricity],
                                                 [semimajor_axis],
                                                 [inclination],
                                                 [raan],
                                                 [arg_per],
                                                 [tru_anom]])

        else:
            raise ValueError("Coordinate keyword not recognised")

    @property
    def _eccentric_anomaly_from_mean_anomaly(self, mean_anomaly, eccentricity,
                                             tolerance=1e-8):
        """Approximately solve the transcendental equation
        :math:`E - e sin E = M_e` for E. This is an iterative process using
        Newton's method.

        Parameters
        ----------
        mean_anomaly : float
            Current mean anomaly
        eccentricity : float
            Orbital eccentricity
        tolerance : float
            Iteration tolerance

        Returns
        -------
        float
            Eccentric anomaly of the orbit
        """

        if mean_anomaly < np.pi:
            ecc_anomaly = mean_anomaly + eccentricity / 2
        else:
            ecc_anomaly = mean_anomaly - eccentricity / 2

        ratio = 1

        while ratio > tolerance:
            f = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly
            fp = 1 - eccentricity * np.cos(ecc_anomaly)
            ratio = f / fp  # Need to check conditioning
            ecc_anomaly = ecc_anomaly - ratio

        return ecc_anomaly

    @property
    def _tru_anom_from_mean_anom(self, mean_anomaly, eccentricity):
        r"""Get the true anomaly from the mean anomaly via the eccentric
        anomaly

        Parameters
        ----------
        mean_anomaly : float
            The mean anomaly
        eccentricity : float
            Eccentricity

        Returns
        -------
        float
            True anomaly

        """
        cos_ecc_anom = np.cos(self._eccentric_anomaly_from_mean_anomaly(
            mean_anomaly, eccentricity))

        return np.arccos((eccentricity - cos_ecc_anom)/
                         (eccentricity*cos_ecc_anom - 1))

    @property
    def _perifocal_position(self, eccentricity, semimajor_axis, true_anomaly):
        r"""The position vector in perifocal coordinates calculated from the
        Keplarian elements

        Parameters
        ----------
        eccentricity : float
            Orbit eccentricity
        semimajor_axis : float
            Orbit semi-major axis
        true_anomaly
            Orbit true anomaly

        Returns
        -------
        numpy.ndarry
            :math:`[r_x, r_y, r_z]` position in perifocal coordinates

        """

        # Cache some trigonometric functions
        c_tran = np.cos(true_anomaly)
        s_tran = np.sin(true_anomaly)

        return semimajor_axis * (1 - eccentricity ** 2) / (1 + eccentricity * c_tran) * \
               np.array([[c_tran],
                         [s_tran],
                         [0]])

    @property
    def _perifocal_velocity(self, eccentricity, semimajor_axis, true_anomaly):
        r"""The velocity vector in perifocal coordinates calculated from the
        Keplarian elements

        Parameters
        ----------
        eccentricity : float
            Orbit eccentricity
        semimajor_axis : float
            Orbit semi-major axis
        true_anomaly
            Orbit true anomaly

        Returns
        -------
        numpy.ndarry
            :math:`[v_x, v_y, v_z]` position in perifocal coordinates

        """

        # Cache some trigonometric functions
        c_tran = np.cos(true_anomaly)
        s_tran = np.sin(true_anomaly)

        return np.sqrt(self.grav_parameter / (semimajor_axis * (1 - eccentricity**2))) * \
               np.array([[-s_tran],
                         [eccentricity + c_tran],
                         [0]])

    @property
    def _perifocal_to_geocentric_matrix(self, inclination, raan, argp):
        r"""Return the matrix which transforms from perifocal to geocentric
        coordinates

        Parameters
        ----------
        inclination : float
            Orbital inclination
        raan : float
            Orbit Right Ascension of the ascending node
        argp : float
            The orbit's argument of periapsis

        Returns
        -------
        numpy.ndarray
            The [3x3] array that transforms from perifocal coordinates to
            geocentric coordinates

        """

        # Cache some trig functions
        s_incl = np.sin(inclination)
        c_incl = np.cos(inclination)

        s_raan = np.sin(raan)
        c_raan = np.cos(raan)

        s_aper = np.sin(argp)
        c_aper = np.cos(argp)

        # Build the matrix
        return np.array([[-s_raan * c_incl * s_aper + c_raan * c_aper,
                          -s_raan * c_incl * c_aper - c_raan * s_aper,
                           s_raan * s_incl],
                          [c_raan * c_incl * s_aper + s_raan * c_aper,
                           c_raan * c_incl * c_aper - s_raan * s_aper,
                           -c_raan * s_incl],
                           [s_incl * s_aper, s_incl * c_aper, c_incl]])

    @property
    def _keplarian_to_rv(self, state_vector):
        r"""Convert the Keplarian orbital elements to position, velocity
        state vector

        Parameters
        ----------
        state_vector : numpy.ndarray()
            defined as

            .. math::

                X = [e, a, i, \Omega, \omega, \\theta]^{T} \\

            where:
            :math:`e` is the orbital eccentricity (unitless),
            :math:`a` the semi-major axis (m),
            :math:`i` the inclination (rad),
            :math:`\Omega` is the longitude of the ascending node (rad),
            :math:`\omega` the argument of periapsis (rad), and
            :math:`\\theta` the true anomaly (rad)

        Returns
        -------
        numpy.ndarray()
            Orbital state vector as :math:`[r_x, r_y, r_z, v_x, v_y, v_z]`

        """

        # Calculate the position vector in perifocal coordinates
        rx = self._perifocal_position(state_vector[0][0],
                                      state_vector[1][0], state_vector[5][0])

        # Calculate the velocity vector in perifocal coordinates
        vx = self._perifocal_velocity(state_vector[0][0],
                                      state_vector[1][0], state_vector[5][0])

        # Transform position (perifocal) and velocity (perifocal)
        # into geocentric
        r = self._perifocal_to_geocentric_matrix(state_vector[2][0],
                                                 state_vector[3][0],
                                                 state_vector[4][0]) * rx
        v = self._perifocal_to_geocentric_matrix(state_vector[2][0],
                                                 state_vector[3][0],
                                                 state_vector[4][0]) * vx

        # And put them into the state vector
        return [r, v]

    def cartesian_state_vector(self):
        r"""Returns the orbital state vector

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray

            .. math::

                X_{t_0} = [r_x, r_y, r_z, \odot{r}_x, \odot{r}_y, \odot{r}_z]

            where:
            :math:`r_x, r_y, r_z` is the Cartesian position coordinate in Earth
            Centered Inertial (ECI) coordinates
            :math:`\odot{r}_x, \odot{r}_y, \odot{r}_z` is the velocity
            coordinate in ECI coordinates

        """

    @property
    def epoch(self):
        """Return the epoch (timestamp)

        Parameters
        ----------
        None

        Returns
        -------
        state.timestamp
            The epoch, or state timestamp

        """

    @property
    def semimajor_axis(self):
        """return the Semi-major axis"""

    @property
    def eccentricity(self):
        """Return the eccentricity"""

    @property
    def inclination(self):
        """Return the orbital inclination"""

    @property
    def longitude_ascending_node(self):
        """Return the longitude (or Right Ascension in the case of the Earth)
        of the ascending node"""

    @property
    def argument_periapsis(self):
        """Return the Argument of Periapsis"""

    @property
    def true_anomaly(self):
        """Return the true anomaly"""

    @property
    def eccentric_anomaly(self):
        """Return the eccentric anomaly. Note that this computes the quantity
        exactly via the Keplerian eccentricity and true anomaly rather than via
        the mean anomaly using an iterative procedure.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The eccentric anomaly, E

        """

        return 2 * np.arctan(np.sqrt((1 - self.eccentricity) /
                                     (1 + self.eccentricity) *
                             np.tan(self.true_anomaly / 2))


    @property
    def mean_anomaly(self):
        """Return the mean anomaly. Uses the eccentric anomaly and Kepler's
        equation to get mean anomaly from true anomaly and eccentricity

        Parameters
        ----------
        None

        Returns
        -------
        float
            Mean anomaly, :math:`M` (radians)

        """

        return self.eccentric_anomaly- self.eccentricity * \
               np.sin(self.eccentric_anomaly)  # Kepler's equation

    @property
    def period(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        float
            Orbital period, :math:`T`

        """
        return ((2 * np.pi) / np.sqrt(self.grav_parameter)) * np.power(self.semimajor_axis, 3 / 2)

    @property
    def specific_angular_momentum(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        float
            Magnitude of the specific angular momentum, :math:`h`

        """
        return np.sqrt(self.grav_parameter * self.semimajor_axis * (1 - self.eccentricity ** 2))

    @property
    def specific_orbital_energy(self):
        """
        Parameters
        ----------

        Returns
        -------
        float
            Specific orbital energy (:math:`\frac{-GM}{2a}`)

        """
        return -self.grav_parameter / (2 * self.semimajor_axis())



class KeplerianOrbitState(OrbitalState):
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


class TLEOrbitState(OrbitalState):
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
