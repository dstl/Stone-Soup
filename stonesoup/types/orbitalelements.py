# -*- coding: utf-8 -*-
import datetime
import numpy as np

from ..base import Property
from .state import State

class OrbitalElements(State):

    r"""

    A multipurpose state with some added routines to facilitate coordinate conversion.

    Create using a :attr:`State` with :attr:`State.state_vector` :math:`X_{t_{0}}` at Epoch
    :attr:`State.timestamp` :math:`t_0`

        .. math::

            X_{t_0} = [e, a, i, \Omega, \omega, \\theta]^{T} \\

    where :math:`e` is the orbital eccentricity (unitless), :math:`a` the semi-major axis (m), :math:`i` the
    inclination (rad), :math:`\Omega` is the longitude of the ascending node (rad), :math:`\omega` the argument of
    periapsis (rad), and :math:`\\theta` the true anomaly (rad).


    The gravitational parameter :math:`GM` can be defined. If left undefined it defaults to that of the Earth,
    :math:`3.986004418 (\pm 0.000000008) \\times 10^{14} \mathrm{m}^3 \mathrm{s}^{−2}`.


    :reference: Curtis, H.D. 2010, Orbital Mechanics for Engineering Students (3rd Ed), Elsevier Aerospace Engineering
                Series

    """

    grav_parameter = Property(
        float, default=3.986004418e14,
        doc="Standard gravitational parameter :math:`\\mu = G M`")

    def __init__(self, state_vector, *args, **kwargs):
        super().__init__(state_vector, *args, **kwargs)

    # Could replace with getters and setters
    def semimajor_axis(self):
        """

        :return: Orbit semimajor axis, :math:`a` (m)

        Note that positive number indicates an elliptical (or circular) orbit, negative a hyperbolic orbit and
        0 is parabolic

        """
        return self.state_vector[1][0]

    def eccentricity(self):
        """

        :return: Orbit eccentricity, :math:`e` (unitless)

        :math:`e = 0` is a circle
        :math:`0 < e < 1`, an elliptical orbit
        :math:`e = 1` is a parabola, and
        :math:`e > 1' denotes a hyperbolic orbit

        """
        return self.state_vector[0][0]

    def inclination(self):
        """

        :return: Orbit inclination (radians)

        Usually defined as the angle between the orbital plane and that defined by the rotational axis of the parent
        body


        """
        # Ensure that what's returned is between 0 and PI/2 (required by some os the astro libraries)
        return np.remainder(self.state_vector[2][0], np.pi/2)


    def long_asc_node(self):
        """

        :return: Longitude of the ascending node :math:`\Omega` (radians)

        For geocentric orbits, often known as the right ascension of the ascending node

        The longitude at which the orbit crosses the parent's rotational plane from negative to positive

        """
        return self.state_vector[3][0]

    def arg_periapsis(self):
        """

        :return: Argument of periapsis, :math:`\omega` (radians)

        Or the argument of perigee for geocentric orbits, argument of perihelion for heliocentric orbits.

        The angle between the orbital semi-major axis and the longitude of the ascending node (measured anticlockwise
        from the ascending node to the semi-major axis). Note that this is undefined for circular orbits, as circles
        don't have a unique semi-major axis.



        """
        return self.state_vector[4][0]

    def true_anomaly(self):
        """

        :return: True anomaly (radians)

        The current angle of the orbiting body measured anticlockwise from the argument of periapsis.

        """
        return self.state_vector[5][0]

    def period(self):
        """

        :return: Orbital period, :math:`T`

        """
        return ((2*np.pi)/np.sqrt(self.grav_parameter))*np.power(self.semimajor_axis(), 3/2)

    def spec_angular_mom(self):
        """

        :return: Magnitude of the specific angular momentum, :math:`h`

        """
        return np.sqrt(self.grav_parameter * self.semimajor_axis() * (1 - self.eccentricity()**2))

    def spec_orb_ener(self):
        """

        :return: Specific orbital energy (:math:`\frac{-GM}{2a}`)

        """
        return -self.grav_parameter/(2 * self.semimajor_axis())

    def mean_anomaly(self):
        """

        :return: Mean anomaly, :math:`M` (radians)

        Uses the eccentric anomaly and Kepler's equation to get mean anomaly from true anomaly and eccentricity

        """

        ecc_anom = 2 * np.arctan(np.sqrt((1-self.eccentricity())/(1+self.eccentricity())) *
                                 np.tan(self.true_anomaly()/2))
        return ecc_anom - self.eccentricity() * np.sin(ecc_anom) # Kepler's equation

    def position_vector(self):
        """

        :return: Position vector in the parent-fixed inertial frame, r (sometimes confusingly called the state vector)

        """

        # Calculate position vector in perifocal coords.
        c_propo = (self.spec_angular_mom()**2 / self.grav_parameter) * \
                  (1/(1 + self.eccentricity() * np.cos(self.true_anomaly())))
        r_peri = c_propo * np.array([[np.cos(self.true_anomaly())],
                                     [np.sin(self.true_anomaly())],
                                     [0]])

        # And transform to geocentric coordinates by means of matrix rotation
        return self.matrix_perifocal_to_geocentric() @ r_peri

    def velocity_vector(self):

        """
        :return: Velocity vector in the parent-centred inertial frame, :math:`v` or :math:`r^/dot`

        """

        # First calculate velocity vector in perifocal coordinates
        c_propo = (self.grav_parameter / self.spec_angular_mom())
        v_peri = c_propo * np.array([[-np.sin(self.true_anomaly())],
                                     [self.eccentricity() + np.cos(self.true_anomaly())],
                                     [0]])

        # then transform to geocentric coordinates by means of matrix rotation
        return self.matrix_perifocal_to_geocentric() @ v_peri

        pass

    def matrix_perifocal_to_geocentric(self):

        """
        :return: Matrix to Rotate vectors from perifocal plane to the plane defined by the parent body

        """

        # Pre-compute some quantities
        s_lascn = np.sin(self.long_asc_node())
        c_lascn = np.cos(self.long_asc_node())

        s_inc = np.sin(self.inclination())
        c_inc = np.cos(self.inclination())

        s_argp = np.sin(self.arg_periapsis())
        c_argp = np.cos(self.arg_periapsis())

        # Populate the matrix
        return np.array([[-s_lascn * c_inc * s_argp + c_lascn * c_argp, -s_lascn * c_inc * c_argp - c_lascn * s_argp,
                       s_lascn * s_inc],
                      [c_lascn * c_inc * s_argp + s_lascn * c_argp, c_lascn * c_inc * c_argp - s_lascn * s_argp,
                       -c_lascn * s_inc],
                      [s_inc * s_argp, s_inc * c_argp, c_inc]])
