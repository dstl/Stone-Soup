# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime

from ...base import Property
from ...types.array import StateVector
from ...types.angle import Bearing, Elevation
from .nonlinear import CartesianToElevationBearing
from ...astronomical_conversions import local_sidereal_time


class ECItoAltAz(CartesianToElevationBearing):
    """
    A measurement model which converts a target with coordinates in Earth-centered inertial (ECI)
    coordinates to local altitude and azimuth. Inherits from :class:`~.CartesianToElevationBearing`
    and overwrites :meth:`function()`.

    The measurement is made by converting sensor and target into Earth-centred
    inertial (ECI) coordinates and then working out the relative position and
    velocity vectors and finally converting back to desired quantities.
    Coordinate transforms are done using Astropy. Astropy has several language
    conventions that are worth knowing.

    """
    ndim_state = Property(int, default=6, doc="The ECI coordinate is usually provided as ")
    mapping = Property(np.ndarray, default=[0, 1, 2], doc="The positional variables in the target"
                                                          "state vector")

    timestamp = Property(datetime, default=946728000.0,
                         doc="Timestamp, defaults to 2000-01-01 12:00:00. Can be aware or naive. "
                             "If aware, will try to correct for timezone.")

    # Position from which the measurement is taken
    latitude = Property(float, default=0.0,
                        doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0,
                         doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.0,
                         doc="Observatory elevation (m)")

    def __init__(self, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)

    def position_vector(self, timestamp=datetime(2000, 1, 1, 12, 0, 0)):
        """ Returns the ECI position at the time the measurement is made.

        Parameters
        ----------
        timestamp: datetime
            The time at which the position vector is calculated. defaults to 2000-01-01 12:00:00.
            Can be aware or naive. If aware, will try to correct for timezone.

        Returns
        -------
         : np.ndarray
            Cartesian position vector of the sensor

        """

        fl = 0.003353  # The oblateness or flattening of the Earth.
        r_e = 6378137  # Earth's equatorial radius

        ssqlat = np.sin(self.latitude)**2
        clat = np.cos(self.latitude)
        slat = np.sin(self.latitude)

        sit = local_sidereal_time(self.longitude, timestamp=timestamp)
        csit = np.cos(sit)  # cos of the sidereal time
        ssit = np.sin(sit)  # sin of the sidereal time

        r_c = r_e/(np.sqrt(1 - (2*fl - fl**2)*ssqlat)) + self.elevation
        r_s = r_e*(1-fl)**2/(np.sqrt(1 - (2*fl - fl**2)*ssqlat)) + self.\
            elevation

        return np.array([[r_c * clat * csit],
                         [r_c * clat * ssit],
                         [r_s * slat]])

    def matrix_eci_to_topoh(self, timestamp=datetime(2000, 1, 1, 12, 0, 0)):
        """ This matrix rotates the ECI coordinate to the topocentric horizon frame

        Parameters
        ----------
        timestamp: datetime
            The time at which the matrix is calculated. defaults to 2000-01-01 12:00:00.
            Can be aware or naive. If aware, will try to correct for timezone.

        Returns
        -------
         : np.ndarray
            the matrix that rotates the ECI coordinate to the topocentric horizon coordinate
        """

        theta = local_sidereal_time(self.longitude, timestamp=timestamp)
        phi = self.latitude

        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        return np.array([[-stheta, ctheta, 0],
                         [-sphi*ctheta, -sphi*stheta, cphi],
                         [cphi*ctheta, cphi*stheta, sphi]])

    def function(self, state, noise=False, timestamp=datetime(2000, 1, 1, 12, 0, 0), **kwargs) \
            -> StateVector:
        r"""The function which returns az, alt given an target with ECI coordinates and a
        measurement
        position and time.

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)
        timestamp: datetime
            The time at which the observation is made. defaults to 2000-01-01 12:00:00.
            Can be aware or naive. If aware, will try to correct for timezone.

        Returns
        -------
        :class:`~.StateVector` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Compute the relative position in ECI
        rp = state.state_vector[self.mapping, :] - self.position_vector(timestamp)

        # To get it into topocentric horizon coordinates we need the
        # transformation matrix
        rt = self.matrix_eci_to_topoh(timestamp) @ rp
        rmag = np.sqrt(rt[0] ** 2 + rt[1] ** 2 + rt[2] ** 2)
        nrt = rt / rmag

        # Turn this into an altitude and azimuth
        alt = np.arcsin(nrt[2])
        azi = np.arccos(nrt[1] / np.cos(alt))

        # Bit of jiggery required to get azimuth quadrant
        if nrt[0] / np.cos(alt) < 0:
            azi = 2 * np.pi - azi

        return StateVector([[Elevation(alt)], [Bearing(azi)]]) + noise
