# -*- coding: utf-8 -*-
"""Coordinate conversions and related functions for use in Astrometric
calculations

"""
import numpy as np
from datetime import datetime


def local_sidereal_time(longitude, datetime_ut=None):
    """Find the sidereal time for a given longitude and time

    Parameters
    ----------
    longitude : float
        The longitude of the point of interest, in radians East of
        the prime meridian
    datetime_ut : datetime.datetime (default, datetime.now())
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

    # TODO: Work out how to take the true local time and convert to UT
    # TODO: using timezones

    if datetime_ut is None:
        ldt = datetime.utcnow()
    else:
        ldt = datetime_ut

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
    theta_g0 = (100.4606184 + 36000.77004*t0 + 0.000387933*t0**2 - 2.583e-8*t0**3) % 360

    # The Greenwich sidereal time at this UT
    theta_g = theta_g0 + 360.98564724*ut/24
    theta_g = theta_g*np.pi/180  # Convert to radians

    return (theta_g + longitude) % (2*np.pi)


def topocentric_to_geocentric(latitude, longitude, height, datetime_ut=None,
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
    datetime_ut : datetime.datetime
        The datetime object representing the time in UT when the
        transformation is to be made. Default is datetime.utcnow()
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

    # If the date time isn't specified use now().
    if datetime_ut is None:
        ldt = datetime.now()
    else:
        ldt = datetime_ut

    # Get the local sidereal time
    lst = local_sidereal_time(longitude, datetime_ut=ldt)

    coeff1 = (radius_p /(np.sqrt(1 - (2*flattening - flattening**2) *
                                 np.sin(latitude)**2)) + height) * \
        np.cos(latitude)

    coeff2 = ((radius_p * (1 - flattening)**2)/(np.sqrt(1 - (2*flattening -
                                                             flattening**2) *
                                                        (np.sin(latitude)**2)))
              + height) * np.sin(latitude)

    return np.array([[coeff1*np.cos(lst)], [coeff1*np.sin(lst)], [coeff2]])


def topocentric_altaz_to_radec(altitude, azimuth, latitude, longitude,
                               datetime_ut=None):
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
    datetime_ut : datetime.datetime
        The datetime object representing the time in UT when the
        calculation is to be made. Default is datetime.utcnow()

    Returns
    -------

    RA, Dec : (radians, radians)
        The right ascension and declination of the target

    """

    # If the date time isn't specified use now().
    if datetime_ut is None:
        ldt = datetime.now()
    else:
        ldt = datetime_ut

    # Ensure the azimuth sits between 0 and 2pi
    azimuth = azimuth % 2*np.pi

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
    rightascension = local_sidereal_time(longitude, datetime_ut=ldt) - \
                     hourangle

    return rightascension, declination
