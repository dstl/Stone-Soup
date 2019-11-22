# -*- coding: utf-8 -*-
import datetime
import numpy as np
import string as strg

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

    coordinates = Property(
        strg, default="cartesian",
        doc="The parameterisation used on initiation. Acceptable values "
            "are 'Cartesian', 'Keplerian', 'TLE', or 'Equinoctial'. All"
            "other inputs will return errors"
    )

    grav_parameter = Property(
        float, default=3.986004418e14,
        doc=r"Standard gravitational parameter :math:`\mu = G M` in units of "
            r":math:`\mathrm{m}^3 \mathrm{s}^{-2}`")

    _eanom_precision = Property(
        float, default=1e-8,
        doc="Precision used for the stopping point in determining eccentric "
            "anomaly from mean anomaly"
    )

    metadata = Property(
        {}, default={}, doc="Dictionary containing metadata about orbit")

    def __init__(self, state_vector, *args, **kwargs):
        r"""Can be initialised in a number of different ways according to
        preference.

        Parameters
        ----------
        state_vector : :class:`numpy.array`
            The input vector whose elements depend on the parameterisation
            used. See 'Keywords' below. Must have dimension 6x1.

        Keywords
        --------
        coordinates
            The chosen input coordinate frame. Can be 'Cartesian', 'Keplerian',
            'TLE' or 'Equinoctal'.


        Returns
        -------
        : Constructs the class

        """
        if len(state_vector) != 6:
            raise ValueError("State vector shape should be 6x1 : got {}".format(
                state_vector.shape))

        super().__init__(state_vector, *args, **kwargs)

        # Query the coordinates
        if self.coordinates.lower() == 'cartesian':
            r"""The orbital state vector.
            
            Parameters
            ----------
            state_vector : numpy.array
            
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

        elif self.coordinates.lower() == 'keplerian':
            r"""The state vector input should have the following input: 
            
            Parameters
            ----------
            state_vector : numpy.array
            
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

        elif self.coordinates.upper() == 'TLE':
            r"""For the TLE state vector, the structure reflects the data 
                format that's been in use since the 1960s:
                
            Parameters
            ----------
            state_vector : numpy.array()
            
                The two line element input vector

                .. math::

                    X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T} \\

                where :math:`i` the inclination (rad),
                :math:`\Omega` is the longitude of the ascending node (rad),
                :math:`e` is the orbital eccentricity (unitless),
                :math:`\omega` the argument of perigee (rad),
                :math:'M_0' the mean anomaly (rad)
                :math:`n` the mean motion (rad/unit time)
            
            Reference
            ---------
            https://spaceflight.nasa.gov/realdata/sightings/SSapplications/
            Post/JavaSSOP/SSOP_Help/tle_def.html
            
            """
            # TODO: ensure this works for parabolas and hyperbolas
            # Get the semi-major axis from the mean motion
            semimajor_axis = np.cbrt(self.grav_parameter/state_vector[5][0]**2)

            # True anomaly from mean anomaly
            tru_anom = self._tru_anom_from_mean_anom(state_vector[4][0], state_vector[2][0])

            self.state_vector = self._keplerian_to_rv(
                np.array([[state_vector[2][0]],
                         [semimajor_axis],
                         [state_vector[0][0]],
                         [state_vector[1][0]],
                         [state_vector[3][0]],
                         [tru_anom]]))

        elif self.coordinates.lower() == 'equinoctial':
            r"""A version of the Equinoctial state vector as input:
            
            Parameters
            ----------
            state_vector : numpy.array

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
            arg_per = np.arctan2(state_vector[1][0], state_vector[2][0]) - raan
            mean_anomaly = state_vector[5][0] - arg_per - raan
            eccentricity = state_vector[1][0]/(np.sin(arg_per + raan))

            # True anomaly from mean anomaly
            tru_anom = self._tru_anom_from_mean_anom(mean_anomaly,
                                                     eccentricity)

            self.state_vector = self._keplerian_to_rv(
                np.array([[eccentricity],
                         [semimajor_axis],
                         [inclination],
                         [raan],
                         [arg_per],
                         [tru_anom]]))

        else:
            raise TypeError("Coordinate keyword not recognised")

    '''A few helper functions compute intermediate quantities'''
    def _eccentric_anomaly_from_mean_anomaly(self, mean_anomaly, eccentricity):
        """Approximately solve the transcendental equation
        :math:`E - e sin E = M_e` for E. This is an iterative process using
        Newton's method.

        Parameters
        ----------
        mean_anomaly : float
            Current mean anomaly
        eccentricity : float
            Orbital eccentricity

        Returns
        -------
        : float
            Eccentric anomaly of the orbit
        """

        if mean_anomaly < np.pi:
            ecc_anomaly = mean_anomaly + eccentricity / 2
        else:
            ecc_anomaly = mean_anomaly - eccentricity / 2

        ratio = 1

        while np.abs(ratio) > self._eanom_precision:
            f = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly
            fp = 1 - eccentricity * np.cos(ecc_anomaly)
            ratio = f / fp  # Need to check conditioning
            ecc_anomaly = ecc_anomaly - ratio

        return ecc_anomaly # Check whether this ever goes outside 0 < 2pi

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
        : float
            True anomaly

        """
        cos_ecc_anom = np.cos(self._eccentric_anomaly_from_mean_anomaly(
            mean_anomaly, eccentricity))
        sin_ecc_anom = np.sin(self._eccentric_anomaly_from_mean_anomaly(
            mean_anomaly, eccentricity))

        # This only works for M_e < \pi
        # return np.arccos(np.clip((eccentricity - cos_ecc_anom) /
        #                 (eccentricity*cos_ecc_anom - 1), -1, 1))

        return np.remainder(np.arctan2(np.sqrt(1 - eccentricity**2) *
                                       sin_ecc_anom,
                                       cos_ecc_anom - eccentricity), 2*np.pi)

    @staticmethod
    def _perifocal_position(eccentricity, semimajor_axis, true_anomaly):
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
        : numpy.array
            :math:`[r_x, r_y, r_z]` position in perifocal coordinates

        """

        # Cache some trigonometric functions
        c_tran = np.cos(true_anomaly)
        s_tran = np.sin(true_anomaly)

        return semimajor_axis * (1 - eccentricity ** 2) / (1 + eccentricity * c_tran) * \
               np.array([[c_tran], [s_tran], [0]])

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
        : numpy.narray
            :math:`[v_x, v_y, v_z]` position in perifocal coordinates

        """

        # Cache some trigonometric functions
        c_tran = np.cos(true_anomaly)
        s_tran = np.sin(true_anomaly)

        return np.sqrt(self.grav_parameter / (semimajor_axis * (1 - eccentricity**2))) * \
               np.array([[-s_tran],
                         [eccentricity + c_tran],
                         [0]])

    @staticmethod
    def _perifocal_to_geocentric_matrix(inclination, raan, argp):
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
        : numpy.array
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

    def _keplerian_to_rv(self, state_vector):
        r"""Convert the Keplarian orbital elements to position, velocity
        state vector

        Parameters
        ----------
        state_vector : numpy.array()
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
        : numpy.array
            Orbital state vector as :math:`[r_x, r_y, r_z, v_x, v_y, v_z]`

        Warning
        -------
        No checking. Assumes Keplerian elements rendered correctly as above

        """

        # Calculate the position vector in perifocal coordinates
        rx = self._perifocal_position(state_vector[0][0], state_vector[1][0],
                                      state_vector[5][0])

        # Calculate the velocity vector in perifocal coordinates
        vx = self._perifocal_velocity(state_vector[0][0], state_vector[1][0],
                                      state_vector[5][0])

        # Transform position (perifocal) and velocity (perifocal)
        # into geocentric
        r = self._perifocal_to_geocentric_matrix(state_vector[2][0],
                                                 state_vector[3][0],
                                                 state_vector[4][0]) @ rx
        v = self._perifocal_to_geocentric_matrix(state_vector[2][0],
                                                 state_vector[3][0],
                                                 state_vector[4][0]) @ vx

        # And put them into the state vector
        return np.concatenate((r, v), axis=0) # Don't really need the keyword...

    '''Some vector quantities'''
    @property
    def _nodeline(self):
        """The vector node line (defines the longitude of the ascending node
        in the Primary-centered inertial frame)

        Parameters
        ----------

        Returns
        -------
        : numpy.array
            The node line (defines the longitude of the ascending node)
        """
        k = np.array([0, 0, 1])
        boldh = self.specific_angular_momentum

        return np.cross(k, boldh, axis=0)

    @property
    def _eccentricity_vector(self):
        r""" The eccentricity vector

        Parameters
        ----------

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
        r"""Specific angular momentum

        Parameters
        ----------

        Returns
        -------
        : numpy.array
            The specific angular momentum, :math:`\mathbf{h}`

        """
        return np.cross(self.state_vector[0:3], self.state_vector[3:6], axis=0)

    @property
    def cartesian_state_vector(self):
        r"""Returns the orbital state vector

        Parameters
        ----------

        Returns
        -------
        : numpy.array

            .. math::

                X_{t_0} = [r_x, r_y, r_z, \odot{r}_x, \odot{r}_y, \odot{r}_z]

            where:
            :math:`r_x, r_y, r_z` is the Cartesian position coordinate in Earth
            Centered Inertial (ECI) coordinates
            :math:`\odot{r}_x, \odot{r}_y, \odot{r}_z` is the velocity
            coordinate in ECI coordinates

        """
        return self.state_vector

    '''Some scalar quantities'''
    @property
    def epoch(self):
        """Return the epoch (timestamp)

        Parameters
        ----------

        Returns
        -------
        : :class:`~.datetime.datetime`
            The epoch, or state timestamp

        """
        return self.timestamp

    @property
    def range(self):
        """The current distance

        Parameters
        ----------

        Returns
        -------
        : float
            The current distance to object

        """
        return np.sqrt(np.dot(self.state_vector[0:3].T,
                              self.state_vector[0:3])).item()

    @property
    def speed(self):
        """The current speed (scalar)

        Parameters
        ----------

        Returns
        -------
        : float
            The current instantaneous speed (scalar)

        """
        return np.sqrt(np.dot(self.state_vector[3:6].T,
                              self.state_vector[3:6]).item())

    @property
    def semimajor_axis(self):
        """return the Semi-major axis

        Parameters
        ----------

        Returns
        -------
        : float
            The orbit semi-major axis

        """
        return 1/((2/self.range) - (self.speed**2)/self.grav_parameter)


    @property
    def eccentricity(self):
        r"""Return the eccentricity (uses the form that depends only on scalars

        Parameters
        ----------

        Returns
        -------
        : float
            The orbital eccentricity, :math:`e, \, (0 \le e \le 1)`

        """
        #TODO Check to see which of the following is quicker/better

        # Either
        #return np.sqrt(np.dot(self._eccentricity_vector.T, self._eccentricity_vector).item())
        # or
        return np.sqrt(1 + (self.mag_specific_angular_momentum**2/self.grav_parameter**2) *
                       (self.speed**2 - 2*self.grav_parameter/self.range))

    @property
    def inclination(self):
        r"""Return the orbital inclination

        Parameters
        ----------

        Returns
        -------
        : float
            Orbital inclination, :math:`i, \, (0 \le i \le \pi)`

        """
        boldh = self.specific_angular_momentum
        h = self.mag_specific_angular_momentum

        # Note no quadrant ambiguity
        return np.arccos(np.clip(boldh[2].item()/h, -1, 1))

        # TODO: Will the output limit need to be checked?
        # Logically not, but it might be worth plotting to check

    @property
    def longitude_ascending_node(self):
        r"""Return the longitude (or right ascension in the case of the Earth)
        of the ascending node

        Parameters
        ----------

        Returns
        -------
        : float
            The longitude (or right ascension) of ascending node, :math:`Omega,
            \, (0 \le \Omega \le 2\pi)`

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
        r"""Return the Argument of Periapsis

        Parameters
        ----------

        Returns
        -------
        : float
            The argument of periapsis, :math:`\omega, \, (0 \le \omega \le
            2\pi)`

        """

        boldn = self._nodeline
        n = np.sqrt(np.dot(boldn.T, boldn).item())
        bolde = self._eccentricity_vector

        # Quadrant ambiguity. The clip function is required to mitigate against
        # the occasional floating-point errors which push the ratio outside the
        # -1,1 region.
        if bolde[2].item() >= 0:
            return np.arccos(np.clip(np.dot(boldn.T, bolde).item() /
                             (n * self.eccentricity), -1, 1))
        elif bolde[2].item() < 0:
            aa = np.dot(boldn.T, bolde).item()/(n * self.eccentricity)
            return 2*np.pi - np.arccos(np.clip(np.dot(boldn.T, bolde).item() /
                                       (n * self.eccentricity), -1, 1))
        else:
            raise ValueError("This shouldn't ever happen")

    @property
    def true_anomaly(self):
        r"""Return the true anomaly

        Parameters
        ----------

        Returns
        -------
        : float
            The true anomaly, :math:`\theta, \, (0 \le \theta \le 2\pi)`

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
        r"""Return the eccentric anomaly. Note that this computes the quantity
        exactly via the Keplerian eccentricity and true anomaly rather than via
        the mean anomaly using an iterative procedure.

        Parameters
        ----------

        Returns
        -------
        : float
            The eccentric anomaly, :math:`E, \, \[0 \le E \le 2\pi\]` radians)

        """
        return np.remainder(2 * np.arctan(np.sqrt((1 - self.eccentricity) /
                                                  (1 + self.eccentricity)) *
                                          np.tan(self.true_anomaly / 2)),
                            2*np.pi)

    @property
    def mean_anomaly(self):
        r"""Return the mean anomaly. Uses the eccentric anomaly and Kepler's
        equation to get mean anomaly from true anomaly and eccentricity

        Parameters
        ----------

        Returns
        -------
        : float
            Mean anomaly, :math:`M` (radians; :math:`0 \le M \le 2\pi`)

        """

        return self.eccentric_anomaly - self.eccentricity * \
               np.sin(self.eccentric_anomaly)  # Kepler's equation

    @property
    def period(self):
        """
        Parameters
        ----------

        Returns
        -------
        : float
            Orbital period, :math:`T`

        """
        return ((2 * np.pi) / np.sqrt(self.grav_parameter)) * np.power(self.semimajor_axis, 3 / 2)

    @property
    def mean_motion(self):
        r"""
        Parameters
        ----------

        Returns
        -------
        : float
            The mean motion, :math:`\frac{2 \pi}{T}`, where :math:`T` is the
            period (rad / unit time)

        """
        return 2 * np.pi / self.period

    @property
    def mag_specific_angular_momentum(self):
        """
        Parameters
        ----------

        Returns
        -------
        : float
            The magnitude of the specific angular momentum, :math:'h'

        """
        boldh = self.specific_angular_momentum
        return np.sqrt(np.dot(boldh.T, boldh).item())

        # Alternative via scalars
        # return np.sqrt(self.grav_parameter * self.semimajor_axis * (1 - self.eccentricity ** 2))

    @property
    def specific_orbital_energy(self):
        r"""
        Parameters
        ----------

        Returns
        -------
        : float
            Specific orbital energy (:math:`\frac{-GM}{2a}`)

        """
        return -self.grav_parameter / (2 * self.semimajor_axis)

    @property
    def equinocital_h(self):
        r"""The horizontal component of the eccentricity in the equinoctial
        parameterisation

        Parameters
        ----------

        Returns
        -------
        : float
            The horizontal component of the eccentricity in equinoctial
            coordinates is :math:`h = e \sin (\omega + \Omega)

        """

        return self.eccentricity * np.sin(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinocital_k(self):
        r"""The vertical component of the eccentricity in the equinoctial
        parameterisation

        Parameters
        ----------

        Returns
        -------
        : float
            The vertical component of the eccentricity in equinoctial
            coordinates is :math:`k = e \cos (\omega + \Omega)

        """

        return self.eccentricity * np.cos(self.argument_periapsis +
                                          self.longitude_ascending_node)

    @property
    def equinocital_p(self):
        r"""The horizontal component of the inclination in the equinoctial
        parameterisation

        Parameters
        ----------

        Returns
        -------
        : float
            The horizontal component of the inclination in equinoctial
            coordinates is :math:`p = \tan (i/2) \sin \Omega

        """

        return np.tan(self.inclination/2) * \
               np.sin(self.longitude_ascending_node)

    @property
    def equinocital_q(self):
        r"""The vertical component of the inclination in the equinoctial
        parameterisation

        Parameters
        ----------

        Returns
        -------
        : float
            The vertical component of the inclination in equinoctial
            coordinates is :math:`q = \tan (i/2) \cos \Omega

        """

        return np.tan(self.inclination / 2) * \
               np.cos(self.longitude_ascending_node)

    @property
    def mean_longitude(self):
        r"""The mean longitude

        Parameters
        ----------

        Returns
        -------
        : float
            The mean longitude, defined as :math:`\lambda = M_0 + \omega +
            \Omega`

        """
        return self.mean_anomaly + self.argument_periapsis + \
               self.longitude_ascending_node

    '''The following return vectors of complete sets of elements'''
    @property
    def keplerian_elements(self):
        r"""
        Return the vector of Keplerian elements

        Parameters
        ----------

        Returns
        -------
        : numpy.array

            .. math::

                    X = [e, a, i, \Omega, \omega, \\theta]^{T} \\

                where:
                :math:`e` is the orbital eccentricity (unitless),
                :math:`a` the semi-major axis ([length]),
                :math:`i` the inclination (rad),
                :math:`\Omega` is the longitude of the ascending node (rad),
                :math:`\omega` the argument of periapsis (rad), and
                :math:`\theta` the true anomaly (rad)

        """

        return np.array([[self.eccentricity],
                           [self.semimajor_axis],
                           [self.inclination],
                           [self.longitude_ascending_node],
                           [self.argument_periapsis],
                           [self.true_anomaly]])

    @property
    def two_line_element(self):
        r"""Return the state vector in the form of the NASA Two-Line Element

        Parameters
        ----------

        Returns
        -------
        : numpy.array
            The two line element input vector

                .. math::

                    X_{t_0} = [i, \Omega, e, \omega, M_0, n]^{T} \\

                where :math:`i` the inclination (rad),
                :math:`\Omega` is the longitude of the ascending node (rad),
                :math:`e` is the orbital eccentricity (unitless),
                :math:`\omega` the argument of periapsis (rad),
                :math:'M_0' the mean anomaly (rad)
                :math:`n` the mean motion (rad/[time])
        """
        return np.array([[self.inclination],
                           [self.longitude_ascending_node],
                           [self.eccentricity],
                           [self.argument_periapsis],
                           [self.mean_anomaly],
                           [self.mean_motion]])

    @property
    def equinoctial_elements(self):
        r"""Return the equinoctial element state vector

        Parameters
        ----------

        Returns
        -------
        : numpy.array

            .. math::

                    X_{t_0} = [a, h, k, p, q, \lambda]^{T} \\

                where :math:`a` the semi-major axis ([length]),
                :math:`h` is the horizontal component of the eccentricity
                :math:`e` (unitless),
                :math:`k` is the vertical component of the eccentricity
                :math:`e` (unitless),
                :math:`p` is the horizontal component of the inclination
                :math:`i` (unitless),
                :math:`q` is the vertical component of the inclination
                :math:`i` (unitless) and
                :math:'lambda' is the mean longitude (rad)

        Reference
        ---------
        Broucke, R. A. & Cefola, P. J. 1972, Celestial Mechanics, Volume 5, Issue 3, pp.303-310

        """
        return np.array([[self.semimajor_axis],
                           [self.equinocital_h],
                           [self.equinocital_k],
                           [self.equinocital_p],
                           [self.equinocital_q],
                           [self.mean_longitude]])


class KeplerianOrbitalState(OrbitalState):
    r"""Merely a shell for the OrbitalState(coordinates='Keplerian') class, but
    includes some boundary checking. As a reminder:

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
    def __init__(self, state_vector, *args, **kwargs):
        # Ensure that the coordinates keyword is set to 'Keplerian' and do some
        # additional checks.

        if np.less(state_vector[0][0], 0.0) | np.greater(state_vector[0][0], 1.0):
            raise ValueError("Eccentricity should be between 0 and 1: got {}"
                             .format(state_vector[0][0]))
        if np.less(state_vector[2][0], 0.0) | np.greater(state_vector[2][0], np.pi):
            raise ValueError("Inclination should be between 0 and pi: got {}"
                             .format(state_vector[2][0]))
        if np.less(state_vector[3][0], 0.0) | np.greater(state_vector[3][0], 2 * np.pi):
            raise ValueError("Longitude of Ascending Node should be between 0 and 2*pi: got {}"
                             .format(state_vector[3][0]))
        if np.less(state_vector[4][0], 0.0) | np.greater(state_vector[4][0], 2 * np.pi):
            raise ValueError("Argument of Periapsis should be between 0 and 2*pi: got {}"
                             .format(state_vector[4][0]))
        if np.less(state_vector[5][0], 0.0) | np.greater(state_vector[5][0], 2 * np.pi):
            raise ValueError("True Anomaly should be between 0 and 2*pi: got {}"
                             .format(state_vector[5][0]))

        # And go ahead and initialise as previously
        super().__init__(state_vector, coordinates='keplerian', *args, **kwargs)


class TLEOrbitalState(OrbitalState):
    r"""
    For the TLE state vector:

        .. math::

            X_{t_0} = [i, \Omega, e, \omega, n, M_0]^{T} \\

    where :math:`i` the inclination (rad),
    :math:`\Omega` is the longitude of the ascending node (rad),
    :math:`e` is the orbital eccentricity (unitless),
    :math:`\omega` the argument of perigee (rad),
    :math:'M_0' the mean anomaly (rad) and
    :math:`n` the mean motion (rad/[time]).

    """
    def __init__(self, state_vector, *args, **kwargs):
        if np.less(state_vector[2][0], 0.0) | np.greater(state_vector[2][0], 1.0):
            raise ValueError("Eccentricity should be between 0 and 1: got {}"
                             .format(state_vector[0][0]))
        if np.less(state_vector[0][0], 0.0) | np.greater(state_vector[0][0], np.pi):
            raise ValueError("Inclination should be between 0 and pi: got {}"
                             .format(state_vector[1][0]))
        if np.less(state_vector[1][0], 0.0) | np.greater(state_vector[1][0], 2*np.pi):
            raise ValueError("Longitude of Ascending Node should be between 0 and 2*pi: got {}"
                             .format(state_vector[2][0]))
        if np.less(state_vector[3][0], 0.0) | np.greater(state_vector[3][0], 2*np.pi):
            raise ValueError("Argument of Periapsis should be between 0 and 2*pi: got {}"
                             .format(state_vector[3][0]))
        if np.less(state_vector[4][0], 0.0) | np.greater(state_vector[4][0], 2*np.pi):
            raise ValueError("Mean Anomaly should be between 0 and 2*pi: got {}"
                             .format(state_vector[5][0]))

        super().__init__(state_vector, coordinates='TLE', *args, **kwargs)


class EquinoctialOrbitalState(OrbitalState):
    r"""
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
        """Don't know where these next few lines came from. They're wrong.
        #if np.less(state_vector[3][0], -1.0) | np.greater(state_vector[3][0], 1.0):
            raise ValueError("Horizontal Inclination should be between -1 and 1: got {}"
                             .format(state_vector[3][0]))
        if np.less(state_vector[4][0], -1.0) | np.greater(state_vector[4][0], 1.0):
            raise ValueError("Vertical Inclination should be between -1 and -1: got {}"
                             .format(state_vector[4][0]))"""
        if np.less(state_vector[5][0], 0.0) | np.greater(state_vector[5][0], 2*np.pi):
            raise ValueError("Mean Longitude should be between 0 and 2*pi: got {}"
                             .format(state_vector[5][0]))

        super().__init__(state_vector, coordinates='Equinoctial', *args, **kwargs)


'''Retained in case those who wrote them want to recover. If not, delete...
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

    :reference: Curtis, H.D. 2010, Orbital Mechanics for Engineering Students
    (3rd Ed), Elsevier Aerospace Engineering Series
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

'''