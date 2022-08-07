# -*- coding: utf-8 -*-
"""Coordinate conversions and related functions for use in Astrometric
calculations

"""
import numpy as np
from datetime import datetime


def local_sidereal_time(longitude, timestamp=None):
    """Find the sidereal time for a given longitude and time

    Parameters
    ----------
    longitude : float
        The longitude of the point of interest, in radians East of
        the prime meridian
    timestamp : datetime.datetime (default, datetime.utcnow())
        The time in UT at which to calculate. Defaults to the time at
        which the calculation is made in UTC.

    Returns
    -------
    float :
        The sidereal time (in units of radians)

    Reference
    ---------
    1. Boulet D.L. 1991, Methods of orbit determination for the
    microcomputer, Willmann-Bell

    """

    if timestamp is None:
        ldt = datetime.utcnow()
    else:
        # Correct to utc. If no timezone given then assume it's given in utc.
        if timestamp.utcoffset() is not None:
            ldt = timestamp - timestamp.utcoffset()
        else:
            ldt = timestamp  # assumes time is already given in utc

    # J0 is the Julian day at 0 UT which can be found via
    j0 = ldt.toordinal() + 1721424.5

    # Alternatively use the formula from Boulet (1991) which only works in the
    # 20th and 21st centuries, apparently
    # y = ldt.year
    # m = ldt.month
    # d = ldt.day
    # j0 = 367*y - int(7*(y + int((m+9)/12))/4) + int(275*m/9) + d + 1721013.5

    # The UT (decimal hours)
    ut = ldt.hour + ldt.minute / 60 + ldt.second / 3600 + \
        ldt.microsecond / (3600 * 1e6)

    # Julian centuries between J0 and J2000
    t0 = (j0 - 2451545)/36525

    # The Greenwich sidereal time at 0 UT (in degrees)
    theta_g0 = (100.4606184 + 36000.77004*t0 + 0.000387933*t0**2 -
                2.583e-8*t0**3) % 360

    # The Greenwich sidereal time at this UT
    theta_g = theta_g0 + 360.98564724*ut/24
    theta_g = theta_g*np.pi/180  # Convert to radians

    return (theta_g + longitude) % (2*np.pi)


def topocentric_to_geocentric(latitude, longitude, height, timestamp=None,
                              radius_p=6378137, flattening=0.00335281):
    r"""Compute the geocentric position of a position given in topocentric
    coordinates.

    Parameters
    ----------
    latitude : float (radians)
        Geodetic latitude
    longitude : float (radians)
        Geodetic longitiude
    height : float (m)
        Altitude above 'sea level', i.e. the surface of the oblate
        spheriod that is the the primary body
    timestamp : datetime.datetime
        The datetime object representing the time when the
        transformation is to be made. Default is datetime.utcnow(). If timezone information via
        tzinfo is included then :meth:`local_sidereal_time()` will correct to UT. Otherwise UT is
        assumed.
    radius_p : float (m)
        The equatorial radius of the primary body. This defaults to the
        approximate average value for the Earth: 6,378,137 m
    flattening : float (unitless)
        The flattening factor of the oblate spheroid defined as
        :math:`\frac{R_{eq} - R_{po}}{R_{eq}}` where :math:`R_{eq}` and
        :math:`R_{po}` define the equatorial and polar radii
        respectively. Defaults to that of the Earth, (0.00335281)


    Returns
    -------
    np.array :
        The geocentric position vector, :math:`[r_x r_y r_z]^T`

    """
    # Get the local sidereal time
    lst = local_sidereal_time(longitude, timestamp=timestamp)

    coeff1 = (radius_p/(np.sqrt(1 - (2*flattening - flattening**2)*np.sin(latitude)**2)) + height)\
        * np.cos(latitude)

    coeff2 = ((radius_p * (1 - flattening)**2)/(np.sqrt(1 - (2*flattening - flattening**2) *
                                                        (np.sin(latitude)**2))) + height) * \
        np.sin(latitude)

    return np.array([[coeff1*np.cos(lst)], [coeff1*np.sin(lst)], [coeff2]])


def topocentric_altaz_to_radec(altitude, azimuth, latitude, longitude,
                               timestamp=None):
    """Convert the topocentric altitude and azimuth of a target observed
    from a particular location (specified by the latitude and longitude,
    and time) into the (absolute) right ascension and declination

    Parameters
    ----------
    altitude : float (radians)
        Topocentric altitude above horizon
    azimuth : float (radians)
        Topocentric azimuth East of North
    latitude : float (radians)
        Geodetic latitude
    longitude : float (radians)
        Geodetic longitiude
    timestamp : datetime.datetime
        The datetime object representing the time when the transformation is to be made. Default
        is datetime.utcnow(). If timezone information via tzinfo is included then
        :meth:`local_sidereal_time()` will correct to UT. Otherwise UT is assumed.

    Returns
    -------
    RA, Dec : (radians, radians)
        The right ascension and declination of the target

    """
    # Ensure the azimuth sits between 0 and 2pi
    azimuth = azimuth % (2*np.pi)

    # Some trigonometric pre-calculations
    clat = np.cos(latitude)
    slat = np.sin(latitude)
    caz = np.cos(azimuth)
    calt = np.cos(altitude)
    salt = np.sin(altitude)

    # Calculate dec
    declination = np.arcsin(clat * caz * calt + slat * salt)

    # Calculate hour angle
    if azimuth < np.pi:
        hourangle = 2*np.pi - np.arccos((clat * salt - slat * caz * calt) /
                                        np.cos(declination))
    else:
        hourangle = np.arccos((clat * salt - slat * caz * calt) /
                              np.cos(declination))

    # calculate RA
    rightascension = local_sidereal_time(longitude, timestamp=timestamp) - hourangle

    return rightascension, declination


def topocentric_altaz_to_radecrate(altitude, azimuth, altituderate,
                                   azimuthrate, latitude, longitude,
                                   timestamp=None,
                                   inertial_angular_velocity=7.292115e-5):
    """Convert the topocentric rates of change of altitude and azimuth of
    a target observed from a particular location (specified by the
    latitude and longitude, and time) into the (absolute) rates of change
    of right ascension and declination

    Parameters
    ----------
    altitude : float (radians)
        Topocentric altitude above horizon
    azimuth : float (radians)
        Topocentric azimuth East of North
    altituderate : float (radians/s)
        Rate of change in altitude
    azimuthrate : float (radians/s)
        Rate of change in azimuth
    latitude : float (radians)
        Geodetic latitude
    longitude : float (radians)
        Geodetic longitiude
    timestamp : datetime.datetime
        The datetime object representing the time when the transformation is to be made. Default
        is datetime.utcnow(). If timezone information via tzinfo is included then
        :meth:`local_sidereal_time()` will correct to UT. Otherwise UT is assumed.
    inertial_angular_velocity : float (radians/s),
        The angular velocity of the primary body in its inertial frame.
        Defaults to the value of the Earth, 7.292115e-5 rad s^{-1}

    Returns
    -------
    dRA/dt, dDec/dt : (radians/s, radians/s)
        The rates of change of right ascension and declination of the
        target

    """
    ra, dec = topocentric_altaz_to_radec(altitude, azimuth, latitude,
                                         longitude, timestamp=timestamp)

    # Caching some trig
    slat = np.sin(latitude)
    clat = np.cos(latitude)
    salt = np.sin(altitude)
    calt = np.cos(altitude)
    saz = np.sin(azimuth)
    caz = np.cos(azimuth)

    # The rates of change of ra and dec are
    decdot = (1/np.cos(dec)) * (-azimuthrate * clat * saz * calt +
                                altituderate * (slat*calt - clat*caz*salt))

    radot = inertial_angular_velocity + (azimuthrate * caz * calt -
                                         altituderate * saz * salt + decdot *
                                         saz * calt * np.tan(dec)) / \
        (clat*salt - slat*caz*calt)

    return radot, decdot


def direction_cosine_unit_vector(ra, dec):
    """Calculate the direction cosine unit vector from the Right
    Ascension and Declination

    Parameters
    ----------
    ra : float  # TODO - change this to bearing class?
        The target Right Ascension
    dec : float
        The target's Declination

    Returns
    -------
    : np.array
        3x1 vector of direction cosines

    """

    # Cache some trig results
    sdec = np.sin(dec)
    cdec = np.cos(dec)
    cra = np.cos(ra)
    sra = np.sin(ra)

    # Unit vector direction
    return np.array([[cdec*cra], [cdec*sra], [sdec]])


def direction_rate_cosine_unit_vector(ra, dec, radot, decdot):
    r"""Calculate the direction rate cosine unit vector from the Right
    Ascension, Declination, and their rates of change

    Parameters
    ----------
    ra : float  # TODO - change this to bearing class?
        The target Right Ascension (radian)
    dec : float
        The target's Declination (radian)
    radot : float
        The rate of change in Right Ascension (rad s^{-1})
    decdot : float
        The rate of change in Declination (rad s^{-1})

    Returns
    -------
    : np.array
        3x1 vector of direction cosines
    """
    # Cache some trig results
    sdec = np.sin(dec)
    cdec = np.cos(dec)
    cra = np.cos(ra)
    sra = np.sin(ra)

    # Direction cosine rates vector
    return np.array([[-radot * sra * cdec - decdot * cra * sdec],
                     [radot * cra * cdec - decdot * sra * sdec],
                     [decdot * cdec]])
def ecliptical2equatorial(x_ec, y_ec, z_ec, eps):
    """
        Convert equatorial coordinates to ecliptical

        Parameters
        ----------
        x_ec: X equatorial position
        y_ec: Y equatorial position
        z_ec: Z equatorial position
        eps: Epsilon

        Returns
        -------
        tuple
            set of coordinates in eccliptical position

    """
    return np.matmul([[1,0,0],[0, np.cos(eps), -1 * np.sin(eps)], [0, np.sin(eps), np.cos(eps)]], [x_ec,y_ec, z_ec])


def equatorial2ecliptical(x_eq, y_eq, z_eq, eps):
    """
        Convert equatorial coordinates to ecliptical

        Parameters
        ----------
        x_eq: X equatorial position
        y_eq: Y equatorial position
        z_eq: Z equatorial position
        eps: Epsilon

        Returns
        -------
        tuple
            set of coordinates in eccliptical position

    """
    return np.matmul([[1,0,0],[0, np.cos(eps), np.sin(eps)], [0, -1* np.sin(eps), np.cos(eps)]],[x_eq,y_eq, z_eq] )



def geodetic_to_cartesian(lng, lat):
    """
        Geodetic coordinates into cartesian

        Parameters
        ----------
        lat: X equatorial position
        long: Y equatorial position


        Returns
        -------
        tuple
            set of coordinates in cartesian

    """
    R = 6371
    return R * np.cos(lat) * np.cos(long), R * np.cos(lat) * np.sin(lng), R*np.sin(lng)



def cartesian_to_geodetic(x,y,z):
    """
        Cartesian coordinates into geodetic

        Parameters
        ----------
        x: x coordinate
        y: y coordinate
        z: z coordinate

        Returns
        -------
        tuple
            set of coordinates in geodetic

    """
    R = 6371
    return np.arcsin(z/R), np.arctan(y,x)

#TODO add cylindrical to geodetic and vice versa
