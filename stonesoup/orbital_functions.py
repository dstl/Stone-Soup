# -*- coding: utf-8 -*-
"""Functions used within multiple orbital classes in Stone Soup

"""
import numpy as np


def stumpf_s(z):
    """The Stumpf S function"""
    if z > 0:
        sqz = np.sqrt(z)
        return (sqz - np.sin(sqz)) / sqz ** 3
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.sinh(sqz) - sqz) / sqz ** 3
    elif z == 0:
        return 1 / 6
    else:
        raise ValueError("Shouldn't get to this point")


def stumpf_c(z):
    """The Stumpf C function"""
    if z > 0:
        sqz = np.sqrt(z)
        return (1 - np.cos(sqz)) / sqz ** 2
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.cosh(sqz) - 1) / sqz ** 2
    elif z == 0:
        return 1 / 2
    else:
        raise ValueError("Shouldn't get to this point")


def eccentric_anomaly_from_mean_anomaly(mean_anomaly, eccentricity,
                                        precision=1e-8):
    r"""Approximately solve the transcendental equation
    :math:`E - e sin E = M_e` for E. This is an iterative process using
    Newton's method.

    Parameters
    ----------
    mean_anomaly : float
        Current mean anomaly
    eccentricity : float
        Orbital eccentricity
    precision : float (default = 1e-8)
        Precision used for the stopping point in determining eccentric
        anomaly from mean anomaly

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

    while np.abs(ratio) > precision:
        f = ecc_anomaly - eccentricity * np.sin(ecc_anomaly) - mean_anomaly
        fp = 1 - eccentricity * np.cos(ecc_anomaly)
        ratio = f / fp  # Need to check conditioning
        ecc_anomaly = ecc_anomaly - ratio

    return ecc_anomaly  # Check whether this ever goes outside 0 < 2pi


def tru_anom_from_mean_anom(mean_anomaly, eccentricity):
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
    cos_ecc_anom = np.cos(eccentric_anomaly_from_mean_anomaly(
        mean_anomaly, eccentricity))
    sin_ecc_anom = np.sin(eccentric_anomaly_from_mean_anomaly(
        mean_anomaly, eccentricity))

    # This only works for M_e < \pi
    # return np.arccos(np.clip((eccentricity - cos_ecc_anom) /
    #                 (eccentricity*cos_ecc_anom - 1), -1, 1))

    return np.remainder(np.arctan2(np.sqrt(1 - eccentricity**2) *
                                   sin_ecc_anom,
                                   cos_ecc_anom - eccentricity), 2*np.pi)


def perifocal_position(eccentricity, semimajor_axis, true_anomaly):
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

    return semimajor_axis * (1 - eccentricity ** 2) / \
        (1 + eccentricity * c_tran) * np.array([[c_tran], [s_tran],
                                                [0]])


def perifocal_velocity(eccentricity, semimajor_axis, true_anomaly,
                       grav_parameter=3.986004418e14):
    r"""The velocity vector in perifocal coordinates calculated from the
    Keplarian elements

    Parameters
    ----------
    eccentricity : float
        Orbit eccentricity
    semimajor_axis : float
        Orbit semi-major axis
    true_anomaly : float
        Orbit true anomaly
    grav_parameter : float (default is :math:`3.986004418 \times 10^{14}
    \mathrm{m}^3 \mathrm{s}^{-2}`)
        Standard gravitational parameter :math:`\mu = G M`

    Returns
    -------
    : numpy.narray
        :math:`[v_x, v_y, v_z]` position in perifocal coordinates

    """

    # Cache some trigonometric functions
    c_tran = np.cos(true_anomaly)
    s_tran = np.sin(true_anomaly)

    return np.sqrt(grav_parameter / (semimajor_axis * (1 - eccentricity**2)))\
        * np.array([[-s_tran], [eccentricity + c_tran], [0]])


def perifocal_to_geocentric_matrix(inclination, raan, argp):
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


def keplerian_to_rv(state_vector, grav_parameter=3.986004418e14):
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
    grav_parameter : float (default is :math:`3.986004418 \times 10^{14}
    \mathrm{m}^3 \mathrm{s}^{-2}`)
        Standard gravitational parameter :math:`\mu = G M`
    Returns
    -------
    : numpy.array
        Orbital state vector as :math:`[r_x, r_y, r_z, v_x, v_y, v_z]`

    Warning
    -------
    No checking. Assumes Keplerian elements rendered correctly as above

    """

    # Calculate the position vector in perifocal coordinates
    rx = perifocal_position(state_vector[0][0], state_vector[1][0],
                            state_vector[5][0])

    # Calculate the velocity vector in perifocal coordinates
    vx = perifocal_velocity(state_vector[0][0], state_vector[1][0],
                            state_vector[5][0], grav_parameter=grav_parameter)

    # Transform position (perifocal) and velocity (perifocal)
    # into geocentric
    r = perifocal_to_geocentric_matrix(state_vector[2][0], state_vector[3][0],
                                       state_vector[4][0]) @ rx

    v = perifocal_to_geocentric_matrix(state_vector[2][0], state_vector[3][0],
                                       state_vector[4][0]) @ vx

    # And put them into the state vector
    return np.concatenate((r, v), axis=0)