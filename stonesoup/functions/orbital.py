"""
Orbital functions
-----------------

Functions used within multiple orbital classes in Stone Soup

"""
import numpy as np

from . import dotproduct
from ..types.array import StateVector, StateVectors, Matrix


def stumpff_s(z):
    r"""The Stumpff S function

    .. math::

        S(z) = \begin{cases}\frac{\sqrt(z) - \sin{\sqrt(z)}}{(\sqrt(z))^{3}}, & (z > 0)\\
                     \frac{\sinh(\sqrt(-z)) - \sqrt(-z)}{(\sqrt(-z))^{3}}, & (z < 0) \\
                     \frac{1}{6}, & (z = 0)\end{cases}

    Parameters
    ----------
    z : float, array-like
        input parameter, :math:`z` or :math:`[z]`

    Returns
    -------
    : float, array-like
        Output value, :math:`S(z)` in the same format and same size as input.

    """
    gti = z > 0
    lti = z < 0
    eqi = z == 0

    out = np.zeros(np.shape(z)).view(type(z))

    if not np.shape(z):
        if gti:
            sqz = np.sqrt(z)
            out = (sqz - np.sin(sqz)) / sqz ** 3
        elif lti:
            sqz = np.sqrt(-z)
            out = (np.sinh(sqz) - sqz) / sqz ** 3
        else:
            out = 1 / 6
    else:
        out[gti] = (np.sqrt(z[gti]) - np.sin(np.sqrt(z[gti]))) / np.sqrt(z[gti]) ** 3
        out[lti] = (np.sinh(np.sqrt(-z[lti])) - np.sqrt(-z[lti])) / np.sqrt(-z[lti]) ** 3
        out[eqi] = 1 / 6

    return out


def stumpff_c(z):
    r"""The Stumpff C function

    .. math::

        C(z) = \begin{cases}\frac{1 - \cos{\sqrt(z)}}{z}, & (z > 0)\\
                     \frac{\cosh{\sqrt(-z)} - 1}{-z}, & (z < 0) \\
                     \frac{1}{2}, & (z = 0)\end{cases}

    Parameters
    ----------
    z : float
        input parameter, :math:`z`

    Returns
    -------
    : float
        Output value, :math:`C(z)`

    """
    gti = z > 0
    lti = z < 0
    eqi = z == 0

    out = np.zeros(np.shape(z)).view(type(z))

    if not np.shape(z):
        if gti:
            out = (1 - np.cos(np.sqrt(z))) / np.sqrt(z) ** 2
        elif lti:
            out = (np.cosh(np.sqrt(-z)) - 1) / np.sqrt(-z) ** 2
        else:
            out = 1 / 2
    else:
        out[gti] = (1 - np.cos(np.sqrt(z[gti]))) / np.sqrt(z[gti]) ** 2
        out[lti] = (np.cosh(np.sqrt(-z[lti])) - 1) / np.sqrt(-z[lti]) ** 2
        out[eqi] = 1 / 2

    return out


def universal_anomaly_newton(o_state_vector, delta_t,
                             grav_parameter=3.986004418e14, precision=1e-8, max_iterations=1e5):
    r"""Calculate the universal anomaly via Newton's method. Algorithm 3.3 in [1]_.

    Parameters
    ----------
    o_state_vector : :class:`~StateVector, ~StateVectors`
        The orbital state vector formed as
        :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^T`
    delta_t : timedelta
        The time interval over which to estimate the universal anomaly
    grav_parameter : float, optional
        The universal gravitational parameter. Defaults to that of the
        Earth, :math:`3.986004418 \times 10^{14} \ \mathrm{m}^{3} \
        \mathrm{s}^{-2}`
    precision : float, optional
        For Newton's method, the difference between new and old estimates of the universal anomaly
        below which the iteration stops and the answer is returned, (default = 1e-8)
    max_iterations : float, optional
        Maximum number of iterations allowed in while loop (default = 1e5)

    Returns
    -------
    : float
        The universal anomaly, :math:`\chi`

    References
    ----------
    .. [1] Curtis H.D. 2010, Orbital Mechanics for Engineering Students, 3rd Ed., Elsevier

    """

    # For loop across StateVectors
    #out = Matrix(np.zeros((1, np.shape(o_state_vector)[1])))

    # This should really have the calculation abstracted out and then do
    # if statevector do code, else do iteration over code
    #if type(o_state_vector) != StateVectors:
    #    o_state_vector = StateVectors([o_state_vector])

    mag_r_0 = np.sqrt(dotproduct(o_state_vector[0:3, :], o_state_vector[0:3, :]))
    mag_v_0 = np.sqrt(dotproduct(o_state_vector[3:6, :], o_state_vector[3:6, :]))
    v_rad_0 = dotproduct(o_state_vector[3:6, :], o_state_vector[0:3, :]) / mag_r_0
    root_mu = np.sqrt(grav_parameter)
    inv_sma = 2 / mag_r_0 - (mag_v_0 ** 2) / grav_parameter
    chi_i = root_mu * np.abs(inv_sma) * delta_t.total_seconds()

    out = []
    for iinv_sma, cchi_i, mmag_r_0, mmag_v_0, vv_rad_0 in \
            zip(inv_sma.ravel(), chi_i.ravel(), mag_r_0.ravel(), mag_v_0.ravel(), v_rad_0.ravel()):
        ratio = 1
        count = 0
        # Do Newton's method
        while np.abs(ratio) > precision and count <= max_iterations:
            z_i = iinv_sma * cchi_i ** 2
            f_chi_i = mmag_r_0 * vv_rad_0 / root_mu * cchi_i ** 2 * stumpff_c(z_i) + \
                      (1 - iinv_sma * mmag_r_0) * cchi_i ** 3 * stumpff_s(z_i) + \
                      mmag_r_0 * cchi_i - root_mu * delta_t.total_seconds()
            fp_chi_i = mmag_r_0 * vv_rad_0 / root_mu * cchi_i * \
                       (1 - iinv_sma * cchi_i ** 2 * stumpff_s(z_i)) + \
                       (1 - iinv_sma * mmag_r_0) * cchi_i ** 2 * stumpff_c(z_i) + \
                       mmag_r_0
            ratio = f_chi_i / fp_chi_i
            cchi_i = cchi_i - ratio
            count += 1

        out.append(cchi_i)

    return np.reshape(out, np.shape(np.atleast_2d(o_state_vector[0, :])))


def lagrange_coefficients_from_universal_anomaly(o_state_vector, delta_t,
                                                 grav_parameter=3.986004418e14,
                                                 precision=1e-8, max_iterations=1e5):
    r""" Calculate the Lagrangian coefficients, f and g, and their time derivatives, by way of the
    universal anomaly and the Stumpff functions [2]_.

    Parameters
    ----------
    o_state_vector : StateVector
        The (Cartesian) orbital state vector,
        :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]^T`
    delta_t : timedelta
        The time interval over which to calculate
    grav_parameter : float, optional
        The universal gravitational parameter. Defaults to that of the
        Earth, :math:`3.986004418 \times 10^{14} \ \mathrm{m}^{3} \
        \mathrm{s}^{-2}`. Note that the units of time must be seconds.
    precision : float, optional
        Precision to which to calculate the :meth:`universal anomaly` (default = 1e-8). See the doc
        section for that function
    max_iterations : float, optional
        Maximum number of iterations in determining universal anomaly (default = 1e5)

    Returns
    -------
    : float, float, float, float
        The Lagrange coefficients, :math:`f, g, \dot{f}, \dot{g}`, in that order.

    References
    ----------
    .. [2] Bond V.R., Allman M.C. 1996, Modern Astrodynamics: Fundamentals and Perturbation
            Methods, Princeton University Press

    """
    # First get the universal anomaly using Newton's method
    chii = universal_anomaly_newton(o_state_vector, delta_t,
                                    grav_parameter=grav_parameter,
                                    precision=precision, max_iterations=max_iterations)

    # Get the position and velocity vectors
    bold_r_0 = o_state_vector[0:3, :]
    bold_v_0 = o_state_vector[3:6, :]

    # Calculate the magnitude of the position and velocity vectors
    r_0 = np.sqrt(dotproduct(bold_r_0, bold_r_0))
    v_0 = np.sqrt(dotproduct(bold_v_0, bold_v_0))

    # For convenience
    root_mu = np.sqrt(grav_parameter)
    inv_sma = 2 / r_0 - (v_0 ** 2) / grav_parameter
    z = inv_sma * chii ** 2

    # Get the Lagrange coefficients using Stumpf
    f = 1 - chii ** 2 / r_0 * stumpff_c(z)
    g = delta_t.total_seconds() - 1 / root_mu * chii ** 3 * \
        stumpff_s(z)

    # Get the position vector and magnitude of that vector
    bold_r = f * bold_r_0 + g * bold_v_0
    r = np.sqrt(dotproduct(bold_r, bold_r))

    # and the Lagrange (time) derivatives also using Stumpf
    fdot = root_mu / (r * r_0) * (inv_sma * chii ** 3 * stumpff_s(z) - chii)
    gdot = 1 - (chii ** 2 / r) * stumpff_c(z)

    return f, g, fdot, gdot


def eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity,
                                        precision=1e-8, max_iterations=1e5):
    r"""Approximately solve the transcendental equation :math:`E - e sin E = M_e` for E. This is
    an iterative process using Newton's method.

    Parameters
    ----------
    mean_anomaly : float
        Current mean anomaly
    eccentricity : float
        Orbital eccentricity
    precision : float, optional
        Precision used for the stopping point in determining eccentric anomaly from mean anomaly,
        (default = 1e-8)
    max_iterations : float, optional
        Maximum number of iterations for the while loop, (default = 1e5)

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
    count = 0
    while np.abs(ratio) > precision and count <= max_iterations:
        f = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly
        fp = 1 - eccentricity * np.cos(ecc_anomaly)
        ratio = f / fp  # Need to check conditioning
        ecc_anomaly = ecc_anomaly - ratio
        count += 1

    return ecc_anomaly  # Check whether this ever goes outside 0 < 2pi


def tru_anom_from_mean_anom(mean_anomaly, eccentricity, precision=1e-8, max_iterations=1e5):
    r"""Get the true anomaly from the mean anomaly via the eccentric anomaly

    Parameters
    ----------
    mean_anomaly : float
        The mean anomaly
    eccentricity : float
        Eccentricity
    precision : float, optional
        Precision used for the stopping point in determining eccentric anomaly from mean anomaly,
        (default = 1e-8)
    max_iterations : float, optional
        Maximum number of iterations in determining eccentric anomaly, (default = 1e5)

    Returns
    -------
    : float
        True anomaly

    """
    cos_ecc_anom = np.cos(eccentric_anomaly_from_mean_anomaly(
        mean_anomaly, eccentricity, precision=precision, max_iterations=max_iterations))
    sin_ecc_anom = np.sin(eccentric_anomaly_from_mean_anomaly(
        mean_anomaly, eccentricity, precision=precision, max_iterations=max_iterations))

    # This only works for M_e < \pi
    # return np.arccos(np.clip((eccentricity - cos_ecc_anom) /
    #                 (eccentricity*cos_ecc_anom - 1), -1, 1))

    return np.remainder(np.arctan2(np.sqrt(1 - eccentricity ** 2) *
                                   sin_ecc_anom,
                                   cos_ecc_anom - eccentricity), 2 * np.pi)


def perifocal_position(eccentricity, semimajor_axis, true_anomaly):
    r"""The position vector in perifocal coordinates calculated from the Keplerian elements

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

    return semimajor_axis * (1 - eccentricity ** 2) / \
           (1 + eccentricity * c_tran) * np.array([[c_tran], [s_tran],
                                                   [0]])


def perifocal_velocity(eccentricity, semimajor_axis, true_anomaly,
                       grav_parameter=3.986004418e14):
    r"""The velocity vector in perifocal coordinates calculated from the Keplerian elements

    Parameters
    ----------
    eccentricity : float
        Orbit eccentricity
    semimajor_axis : float
        Orbit semi-major axis
    true_anomaly : float
        Orbit true anomaly
    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = G M`. Default is
        :math:`3.986004418 \times 10^{14} \mathrm{m}^3 \mathrm{s}^{-2}`

    Returns
    -------
    : numpy.narray
        :math:`[\dot{r}_x, \dot{r}_y, \dot{r}_z]` velocity in perifocal coordinates

    """

    # Cache some trigonometric functions
    c_tran = np.cos(true_anomaly)
    s_tran = np.sin(true_anomaly)

    return np.sqrt(grav_parameter / (semimajor_axis * (1 - eccentricity ** 2))) \
           * np.array([[-s_tran], [eccentricity + c_tran], [0]])


def perifocal_to_geocentric_matrix(inclination, raan, argp):
    r"""Return the matrix which transforms from perifocal to geocentric coordinates

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
        The :math:`3 \times 3` array that transforms from perifocal coordinates to geocentric
        coordinates

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


def keplerian_to_rv(state_vector, grav_parameter=3.986004418e14):
    r"""Convert the Keplerian orbital elements to position, velocity state vector

    Parameters
    ----------
    state_vector : :class:`~.StateVector`, :class:`~.StateVectors`
        The Keplerian orbital state vector is defined as

        .. math::

            X = [e, a, i, \Omega, \omega, \theta]^{T} \\

        where:
        :math:`e` is the orbital eccentricity (unitless),
        :math:`a` the semi-major axis (m),
        :math:`i` the inclination (rad),
        :math:`\Omega` is the longitude of the ascending node (rad),
        :math:`\omega` the argument of periapsis (rad), and
        :math:`\theta` the true anomaly (rad)

    grav_parameter : float, optional
        Standard gravitational parameter :math:`\mu = G M`. The default is :math:`3.986004418
        \times 10^{14} \mathrm{m}^3 \mathrm{s}^{-2}`

    Returns
    -------
    : :class:`~.StateVector`
        Orbital state vector as :math:`[r_x, r_y, r_z, \dot{r}_x, \dot{r}_y, \dot{r}_z]`

    Warning
    -------
    No checking undertaken. Assumes Keplerian elements rendered correctly as above

    """

    # Calculate the position vector in perifocal coordinates
    rx = perifocal_position(state_vector[0], state_vector[1], state_vector[5])

    # Calculate the velocity vector in perifocal coordinates
    vx = perifocal_velocity(state_vector[0], state_vector[1], state_vector[5],
                            grav_parameter=grav_parameter)

    # Transform position (perifocal) and velocity (perifocal)
    # into geocentric
    r = perifocal_to_geocentric_matrix(state_vector[2], state_vector[3], state_vector[4]) @ rx

    v = perifocal_to_geocentric_matrix(state_vector[2], state_vector[3], state_vector[4]) @ vx

    # And put them into the state vector
    return StateVector(np.concatenate((r, v), axis=0))


def mod_inclination(x):
    r"""Calculates the modulus of an inclination. Inclination angles are within the range :math:`0`
    to :math:`\pi`.

    Parameters
    ----------
    x: float
        inclination angle in radians

    Returns
    -------
    float
        Angle in radians in the range :math:`0` to :math:`+\pi`
    """

    x = x % np.pi

    return x


def mod_elongitude(x):
    r"""Calculates the modulus of an ecliptic longitude in which angles are within the range
    :math:`0` to :math:`2 \pi`.

    Parameters
    ----------
    x: float
        longitudinal angle in radians

    Returns
    -------
    float
        Angle in radians in the range :math:`0` to :math:`+2 \pi`
    """

    x = x % (2 * np.pi)

    return x
