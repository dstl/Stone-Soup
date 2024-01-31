"""
Navigation functions
--------------------
"""

import numpy as np
import pymap3d
from . import localSphere2GCS, build_rotation_matrix


def earthSpeedFlatSq(dx, dy):
    r"""Calculate the Earth speed flat vector respect to the reference
    frame, from 2D Cartesian coordinates.

    Parameters
    ----------
    dx : float, array like
        :math:`dx` Earth velocity component

    dy : float, array like
        :math:`dy` Earth velocity component

    Returns
    -------
    np.array: sum of the powers of dx and dy
    """

    return np.power(dx, 2) + np.power(dy, 2)


def earthSpeedSq(dx, dy, dz):
    r"""Calculate the Earth speed respect to the reference frame
    from 3D Cartesian coordinates.

    Parameters
    ----------
    dx : float, array like
        :math:`dx` Earth velocity component

    dy : float, array like
        :math:`dy` Earth velocity component

    dz : float, array like
        :math:`dz` Earth velocity component

    Returns
    -------
    np.array: sum of the powers of dx, dy and dz
    """
    return np.power(dx, 2) + np.power(dy, 2) + np.power(dz, 2)


def earthSpeedFlat(dx, dy):
    r"""Same as :class:`~.EarthSpeedFlatSq` but squared.

    Parameters
    ----------
    dx : float, array like
        :math:`dx` Earth velocity component

    dy : float, array like
        :math:`dy` Earth velocity component

    Returns
    -------
    np.array: square root of the sum of the powers of dx and dy
    """
    return np.sqrt(earthSpeedFlatSq(dx, dy))


def getEulersAngles(earthSpeed, earthAcceleration):
    r"""Function to obtain the Euler angles from the
        speed of the aeroplane. The Euler angles are:

        - :math:`\theta` : pitch (elevation)
        - :math:`\phi` : roll  (bank)
        - :math:`\psi` : heading (yaw or Azimuth)

    the Euler angles are converted in radians to uniform with Stone Soup codebase

    Parameters
    ----------
    earthSpeed : np.array, float
                Earth velocity components (dx, dy, dz)
                in local Earth coordinates (m/s)

    earthAcceleration : np.array, float
                  Earth acceleration components (ddx, ddy, ddz)
                  in local Earth coordinates (m/s^2)

    Returns
    -------
    np.array, float
            a 3xN matrix of Euler angles (roll, pitch, heading) (in radians)

    np.array, float
            a 3xN matrix of time derivatives of Euler angles (roll, pitch, heading) (radians/s)

    """

    dx, dy, dz = earthSpeed
    ddx, ddy, ddz = earthAcceleration

    # Calculate the earth speed
    Esfq = earthSpeedFlatSq(dx, dy)
    Esf = earthSpeedFlat(dx, dy)
    Ess = earthSpeedSq(dx, dy, dz)

    Phi = np.arctan2(dy, dx)
    Theta = np.arctan2(-dz, Esf)
    Psi = np.radians(0.)

    composite_euler_angles = np.array([Psi, Theta, Phi])

    composite_euler_acc_angles = np.array([0, 0, 0])

    # in case of acceleration different from 0
    if earthAcceleration.any() > 0:

        num_dphi = dx*ddy - dy*ddx

        dPhi = num_dphi / Esfq
        num_dtheta = (dx*ddz + dy*ddy)*dz/Esf - ddz*Esf

        dTheta = (num_dtheta / Ess)
        dPsi = np.radians(0.)

        composite_euler_acc_angles = np.array([dPsi, dTheta, dPhi])

    return (composite_euler_angles, composite_euler_acc_angles)


def euler2rotationVector(psiThetaPhi, dpsiThetaPhi):
    r""" Function to obtain the rotation vector for given Euler angles
        and their time derivative. The Euler angles are
        the heading of the plane, the pitch and roll.
        The angles are in radians.
        This function is taken from [#]_

    Parameters
    ----------
    psiThetaPhi: np.array, float
                    array containing the three Euler angles (radians)

    dpsiThetaPhi: np.array, float
                    array containing the time derivative of the
                    three Euler angles (radians/s)

    Returns
    -------
    np.array, float
            :math:`\omega_{deg}', array of the rotation vectors (radians/s)

    Reference
    ---------
    .. [#] P. Groves, Principles of GNSS, Inertial,
           and Multisensor Integrated
           Navigation Systems (Second Edition), Artech House, 2013.
    """

    phi_rad = psiThetaPhi[0, :]
    theta_rad = psiThetaPhi[1, :]

    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    sin_phi = np.sin(phi_rad)
    cos_phi = np.cos(phi_rad)

    R = np.array([[np.ones_like(theta_rad), np.zeros_like(theta_rad), -sin_theta],
                  [np.zeros_like(theta_rad), cos_phi, sin_phi * cos_theta],
                  [np.zeros_like(theta_rad), -sin_phi, cos_phi * cos_theta]
                  ])

    return np.einsum('ijh, jh-> ih', R, dpsiThetaPhi)


def getAngularRotationVector(states, latLonAlt0):
    r"""Function to obtain the rotation vector measured by
        the gyroscope instrument.

    Parameters
    ----------
    states: :class:`~.State`
        target state containing the positions, velocities,
        acceleration and the Euler angles
    latLonAlt0: np.array
        reference frame in latitude, longitude and altitude

    Returns
    -------
    np.array, float
                :math:`\omega_{deg}' array of the rotation vectors (radians/s)
    """

    # specify the mapping - may fail if we are not doing things in 3D
    position_mapping = (0, 3, 6)
    angles_mapping = (9, 11, 13)
    dangles_mapping = (10, 12, 14)

    # Coordinates in navigation reference frame
    localpos = states[position_mapping, :]  # use the indices of the position
    psiThetaPhi = states[angles_mapping, :]  # use the indices of the Euler angles
    dpsiThetaPhi = states[dangles_mapping, :]  # use the indices of the dEuler

    # Convert the local reference point to latitude and longitude
    lat_deg, _, _ = localSphere2GCS(localpos[0, :],
                                    localpos[1, :],
                                    localpos[2, :],
                                    latLonAlt0)

    omega_nb_b = euler2rotationVector(psiThetaPhi, dpsiThetaPhi)

    omega_ie_n = earthTurnRateVector(lat_deg[:])

    Rbn = build_rotation_matrix(psiThetaPhi)

    omega_ib_b = ((Rbn @ omega_ie_n) + omega_nb_b)

    return omega_ib_b


def earthTurnRateVector(Lat_degs):
    r"""Function to obtain the Earth turn rate vector
        given the latitude in degrees. The Earth turn rate
        is equal to :math:`7.29\times10^5` radians per seconds.
        The equations are from [#]_.

    Parameters
    ----------
    Lat_degs : float
               latitude in degrees

    Returns
    -------
    np.array
        :math:`~\omega_{ie,i}` (radians/s)

    Reference
    ---------
    .. [#] M. Kok, J. Hol and T. Sch\"{o}n,
           “Using Inertial Sensors for Position and Orientation Estimation”,
           Foundations 456 and Trends in Signal Processing, Vol. 11, No. 1–2, pp 1–153, 2017.
    """

    # Earth turn rate (radians/s)
    turn_rate = 7.292115e-5

    # convert the latitude in radians for math operations
    Lat_rads = np.radians(Lat_degs[:])

    # Turn Rate of the Earth (omega_ie_n)
    omega_ie_n = turn_rate * np.array([np.cos(Lat_rads),
                                       np.zeros_like(Lat_rads),
                                       -np.sin(Lat_rads)])
    return omega_ie_n


def getForceVector(state, latLonAlt0):
    """ Measure the force measured by the accelerometer.
        The final product is a matrix that measure
        the forces as explained in [#]_.

    Parameters
    ----------
    state : :class:`~.State`
        Target states in 15-dimension space

    latLonAlt0: np.array
        reference frame array in latitude,
        longitude and altitude

    Returns
    -------
    np.array
        3D array containing the velocity components
        of the Euler angles.

    Reference
    ---------
    .. [#] M. Kok, J. Hol and T. Sch\"{o}n,
           “Using Inertial Sensors for Position and Orientation Estimation”,
           Foundations 456 and Trends in Signal Processing, Vol. 11, No. 1–2, pp 1–153, 2017.
    """

    # mapping
    position_mapping = (0, 3, 6)
    velocity_mapping = (1, 4, 7)
    acceleration_mapping = (2, 5, 8)
    angles_mapping = (9, 11, 13)

    # coordinates in local navigation frame
    localpos = state[position_mapping, :]
    localvel = state[velocity_mapping, :]
    localacc = state[acceleration_mapping, :]
    psiThetaPhi = state[angles_mapping, :]  # turn it to radians to fix the calculations

    lat_deg, _, alt = localSphere2GCS(localpos[0, :],
                                      localpos[1, :],
                                      localpos[2, :],
                                      latLonAlt0)

    # get the 3D local position, acceleration velocity
    p_n = localpos
    a_nn_n = localacc
    v_n_n = localvel

    # omega rotation
    omega_ie_n = earthTurnRateVector(lat_deg)

    a_ii_n = a_nn_n + 2 * np.cross(omega_ie_n, v_n_n, axis=0) + \
        np.cross(omega_ie_n, np.cross(omega_ie_n, p_n, axis=0), axis=0)

    grav_n = getGravityVector(lat_deg[:], alt[:])

    Rbn = build_rotation_matrix(psiThetaPhi)

    fb = Rbn @ (a_ii_n - grav_n)

    return fb


def getGravityVector(lat_degs, alt):
    """Obtain the gravity vector for a particular
        latitude and altitude.
        This code is adapted from [#]_.

    Parameters
    ----------
    lat_degs : degrees
                latitude in degrees
    alt : float
          altitude in meters

    Returns
    -------
    g_vector : np.array, float
                array containing the gravity
                components on the specific latitude, altitude
                location.

    Reference:
    ----------
    .. [#] P. Groves, Principles of GNSS, Inertial,
           and Multisensor Integrated Navigation Systems (Second Edition), Artech House, 2013.
    """

    # Define some WGS 84 ellipsoid
    # earth_major_axis = pymap3d.Ellipsoid.from_name("wgs84").semimajor_axis  # meters
    earth_minor_axis = pymap3d.Ellipsoid.from_name("wgs84").semiminor_axis  # meters

    # Get the Earth flattening and eccentricity
    # flattening = (earth_major_axis - earth_minor_axis) / earth_major_axis
    flattening = pymap3d.Ellipsoid.from_name("wgs84").flattening

    # eccentricity = np.sqrt(flattening * (2 - flattening))
    eccentricity = pymap3d.Ellipsoid.from_name("wgs84").eccentricity

    # Define the equatorial radius (adapted from Groves)
    earth_radius = 6378137.0  # meters

    # Define the angular rate (omega_ie, WGS_84)
    omega_i_e = 7.292115e-5

    # Define the gravitational constant
    mu = 3.9860044e14

    # Convert the latitude in radians from degrees
    lat_rads = np.radians(lat_degs[:])

    # gravity model
    g0_L = 9.7803253359 * ((1. + 0.001931853 * (np.sin(lat_rads)) ** 2) /
                           np.sqrt(1 - eccentricity ** 2 * (np.sin(lat_rads)) ** 2))

    # gravity component north
    g_north = -8.08e-9 * alt * np.sin(2 * lat_rads)
    # gravity component low
    g_down = g0_L * (1. - (2. / earth_radius) *
                     (1. + flattening * (1. - 2. * ((np.sin(lat_rads)) ** 2)) +
                      ((omega_i_e ** 2 * earth_radius ** 2 * earth_minor_axis) / mu))
                     * alt + (3 / (earth_radius ** 2)) * alt ** 2)

    g_vector = np.array([g_north, np.zeros_like(g_down), g_down])

    return g_vector
