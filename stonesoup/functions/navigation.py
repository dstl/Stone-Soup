"""
Navigation functions
--------------------
"""

import numpy as np
from math import pi
from . import localSphere2GCS


def angle_wrap(angle):
    """ Fix the angle if it gets over the threshold"""

    return np.remainder(angle+pi, 2.*pi) - pi


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
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))


def getEulersAngles(earthSpeed, earthAcceleration):
    r"""Function to obtain the Euler angles from the
        speed of the aeroplane. The Euler angles are:

        - :math:`\psi` : heading
        - :math:`\theta` : pitch
        - :math:`\phi` : roll

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
            a 3xN matrix of Euler angles (heading, pitch, roll) (in degrees)

    np.array, float
            a 3xN matrix of time derivatives of Euler angles (heading, pitch, roll) (deg/s)

    """

    dx, dy, dz = earthSpeed
    ddx, ddy, ddz = earthAcceleration

    # Calculate the earth speed
    Esfq = earthSpeedFlatSq(dx, dy)
    Esf = earthSpeedFlat(dx, dy)
    Ess = earthSpeedSq(dx, dy, dz)

    psi = np.degrees(np.arctan2(dy, dx))
    Theta = np.degrees(np.arctan2(-dz, Esf))
    Phi = np.degrees(0.)

    composite_euler_angles = np.array([psi, Theta, Phi])

    composite_euler_acc_angles = np.array([0, 0, 0])

    # in case of acceleration different from 0
    if earthAcceleration.any() > 0:

        num_dpsi = np.multiply(dx, ddy) - np.multiply(dy, ddx)
        dPsi = np.degrees(np.divide(num_dpsi, Esfq))

        num_dtheta = np.divide(np.multiply(dz,
                                           (np.multiply(dx, ddz) +
                                            np.multiply(dy, ddy))
                                           ), Esf) -\
            np.multiply(ddz, Esf)

        dTheta = np.degrees(np.divide(num_dtheta, Ess))
        dPhi = np.degrees(0)
        composite_euler_acc_angles = np.array([dPsi, dTheta, dPhi])

    return (composite_euler_angles, composite_euler_acc_angles)


def euler2rotationVector(psiThetaPhi_deg, dpsiThetaPhi_deg):
    r""" Function to obtain the rotation vector for given Euler angles
        and their time derivative. The Euler angles are
        the heading of the plane, the pitch and roll.
        This function is taken from [#]_

    Parameters
    ----------
    psiThetaPhi_deg: np.array, float
                    array containing the three Euler angles (deg)

    dpsiThetaPhi_deg: np.array, float
                    array containing the time derivative of the
                    three Euler angles (deg/s)

    Returns
    -------
    np.array, float
            :math:`\omega_{deg}', array of the rotation vectors

    Reference
    ---------
    [#]_ :  P. Groves, Principles of GNSS, Inertial,
            and Multisensor Integrated
            Navigation Systems (Second Edition),
            Artech House, 2013.
    """

    # Number of points
    npts = psiThetaPhi_deg.shape[1]

    # Create an array of 3x Number of points
    omega_deg = np.zeros((3, npts))

    # loop on the various points
    for ipoint in range(npts):
        # convert the angles into radians for maths calculations
        theta_rad = np.radians(psiThetaPhi_deg[1, ipoint])
        phi_rad = np.radians(psiThetaPhi_deg[2, ipoint])

        R = np.array([[1, 0, -np.sin(theta_rad)],
                      [0, np.cos(phi_rad), np.sin(phi_rad) * np.cos(theta_rad)],
                      [0, -np.sin(phi_rad), np.cos(phi_rad) * np.cos(theta_rad)]
                      ])

        omega_deg[:, ipoint] = np.matmul(R, dpsiThetaPhi_deg[:, ipoint].reshape(-1, 1))[:, 0]
    return omega_deg


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
                :math:`\omega_{deg}' array of the rotation vectors
    """

    # specify the mapping - may fail if we are not doing things in 3D
    position_mapping = (0, 3, 6)
    velocity_mapping = (1, 4, 7)
    acceleration_mapping = (2, 5, 8)
    angles_mapping = (9, 11, 13)
    dangles_mapping = (10, 12, 14)

    # create a series of points
    npts = states.shape[1]

    # Coordinates in navigation reference frame
    localpos = states[position_mapping, :]  # use the indices of the position
    psiThetaPhi_deg = states[angles_mapping, :]  # use the indices of the Euler angles
    dpsiThetaPhi_deg = states[dangles_mapping, :]  # use the indices of the dEuler

    # Convert the local reference point to latitude and longitude
    lat_deg, _, _ = localSphere2GCS(localpos[0, :],
                                    localpos[1, :],
                                    localpos[2, :],
                                    latLonAlt0)

    omega_ib_b = np.zeros((3, npts))

    # loop over the data points
    for i in range(npts):

        omega_nb_b = euler2rotationVector(psiThetaPhi_deg[:, i].reshape(-1, 1),
                                          dpsiThetaPhi_deg[:, i].reshape(-1, 1))

        # evaluate the rotation of the earth in the specific point
        omega_ie_n = earthTurnRateVector(lat_deg[i])

        Rbn = rotate3Ddeg(psiThetaPhi_deg[0, i],
                          psiThetaPhi_deg[1, i],
                          psiThetaPhi_deg[2, i])

        # assume zero transport rate
        omega_ib_b[:, i] = (np.matmul(Rbn, omega_ie_n.reshape(-1, 1)) + omega_nb_b)[:, 0]

    return omega_ib_b


def rotate3Ddeg(psi, theta, phi):
    """Get a 3D rotation matrix corresponding
        to the Euler angles (in degrees). The
        3D rotation matrix (ground to airframe).

    Parameters
    ----------
    psi : degrees
        Heading angle in degrees
    theta : degrees
        pitch angle in degrees
    phi : degrees
        roll angle in degrees

    Returns
    -------
    np.array
        3D rotation matrix

    """

    # convert the angles in radians
    psi_rad = np.radians(psi)
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # create matrix components
    rot_0 = np.array([(np.cos(psi_rad), np.sin(psi_rad), 0),
                      (-np.sin(psi_rad), np.cos(psi_rad), 0),
                      (0, 0, 1)])

    rot_1 = np.array([(np.cos(theta_rad), 0, -np.sin(theta_rad)),
                      (0, 1, 0),
                      (np.sin(theta_rad), 0, np.cos(theta_rad))
                      ])

    R1 = rot_1 @ rot_0

    rot_2 = np.array([(1, 0, 0),
                      (0, np.cos(phi_rad), np.sin(phi_rad)),
                      (0, -np.sin(phi_rad), np.cos(phi_rad))
                      ])

    return rot_2 @ R1


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
        :math:`~\omega_{ie,i}`

    Reference
    ---------
    [#] : M. Kok, J. Hol and T. Sch\"{o}n,
          “Using Inertial Sensors for Position and Orientation Estimation”,
          Foundations 456 and Trends in Signal Processing, Vol. 11, No. 1–2, pp 1–153, 2017.
    """

    # Earth turn rate (rads/s)
    turn_rate = 7.292115e-5

    # convert the latitude in radians for math operations
    Lat_rads = np.radians(Lat_degs)

    # Turn Rate of the Earth (omega_ie_n)
    omega_ie_n = turn_rate * np.array([np.cos(Lat_rads),
                                       0,
                                       -np.sin(Lat_rads)]).reshape(1, -1)

    return omega_ie_n


def getForceVector(state, latLonAlt0):
    """Measure the force measured by the accelerometer.
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
    [#] = M. Kok, J. Hol and T. Sch\"{o}n,
          “Using Inertial Sensors for Position and Orientation Estimation”,
          Foundations 456 and Trends in Signal Processing, Vol. 11, No. 1–2, pp 1–153, 2017.
    """

    # mapping
    position_mapping = (0, 3, 6)
    velocity_mapping = (1, 4, 7)
    acceleration_mapping = (2, 5, 8)
    angles_mapping = (9, 11, 13)
    dangles_mapping = (10, 12, 14)

    # get the points
    npts = state.shape[1]

    # coordinates in local navigation frame
    localpos = state[position_mapping, :]
    localvel = state[velocity_mapping, :]
    localacc = state[acceleration_mapping, :]
    psiThetaPhi_deg = state[dangles_mapping, :]

    lat_deg, _, alt = localSphere2GCS(localpos[0, :],
                                      localpos[1, :],
                                      localpos[2, :],
                                      latLonAlt0)

    # create a force matrix
    fb = np.zeros((3, npts))

    # loop over the points
    for ipoint in range(npts):
        # get the 3D local position, acceleration velocity
        p_n = localpos[:, ipoint]
        a_nn_n = localacc[:, ipoint]
        v_n_n = localvel[:, ipoint]

        # omega rotation
        omega_ie_n = earthTurnRateVector(lat_deg[ipoint])

        a_ii_n = a_nn_n + 2 * np.cross(omega_ie_n, v_n_n) + \
            np.cross(omega_ie_n, np.cross(omega_ie_n, p_n))

        grav_n = getGravityVector(lat_deg[ipoint], alt[ipoint])

        Rbn = rotate3Ddeg(psiThetaPhi_deg[0, ipoint],
                          psiThetaPhi_deg[1, ipoint],
                          psiThetaPhi_deg[2, ipoint])

        fb[:, ipoint] = np.matmul(Rbn, (a_ii_n - grav_n).reshape(-1, 1))[:, 0]

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
    [#] : P. Groves, Principles of GNSS, Inertial,
          and Multisensor Integrated Navigation Systems (Second Edition),
          Artech House, 2013.
    """

    # Define some WGS 84 ellipsoid
    earth_major_axis = 6378137.0  # meters
    earth_minor_axis = 6356752.314245  # meters

    # compute the Earth flattening and eccentricity
    flattening = (earth_major_axis - earth_minor_axis) / earth_major_axis
    eccentricity = np.sqrt(flattening * (2 - flattening))

    # Define the equatorial radius (adapted from Groves)
    earth_radius = 6378137.0  # meters
    # Define the angular rate (omega_ie, WGS_84)
    omega_i_e = 7.292115e-5
    # Define the gravitational constant
    mu = 3.9860044e14

    # Convert the latitude in radians from degrees
    lat_rads = np.radians(lat_degs)

    # gravity model
    g0_L = 9.7803253359 * ((1. + 0.001931853 *
                            (np.sin(lat_rads)) ** 2) /
                           np.sqrt(1 - eccentricity ** 2 *
                                   (np.sin(lat_rads)) ** 2))

    # gravity component north
    g_north = -8.08e-9 * alt * np.sin(2 * lat_rads)
    # gravity component low
    g_down = g0_L * (1. - (2. / earth_radius) *
                     (1. + flattening * (1. - 2. *
                                         ((np.sin(lat_rads)) ** 2))
                      + ((omega_i_e ** 2 * earth_radius ** 2 * earth_minor_axis) / mu))
                     * alt + (3 / (earth_radius ** 2)) * alt ** 2)

    g_vector = np.array([g_north, 0, g_down])

    return g_vector
