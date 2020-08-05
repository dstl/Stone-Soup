import numpy as np

from datetime import datetime
from typing import Union


from astropy import constants as const
from astropy import units as u
from astropy import time as tim

from ...base import Property
from ...types.array import StateVector, StateVectors
from ...types.angle import Bearing, Elevation
from .base import MeasurementModel
from .nonlinear import CartesianToElevationBearing


class ECItoAzAlt(CartesianToElevationBearing):
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
    mapping = Property(np.ndarray, default=[0, 1, 2], doc="The positional variables in the target"
                                                          "state vector")

    timestamp = Property(datetime, default=946728000.0,
                         doc="Timestamp, defaults to 2000-01-01 12:00:00")

    # Position from which the measurement is taken
    latitude = Property(float, default=0.0,
                        doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0,
                         doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.9,
                         doc="Observatory elevation (m)")

    def sidtime_rad(self):
        """
        Local mean sidereal time at the measurement in units of radians

        Returns
        -------
         : float
            local mean sidereal time (radians)
        """
        time = tim.Time(self.timestamp)
        sid_t = time.sidereal_time('mean',
                                   longitude=self.longitude)  # The sidereal
        # time - rendered in units of hour angle
        return sid_t.to(u.rad)  # local mean sidereal time in radian

    @property
    def position_vector(self):
        """ Returns the ECI position of the sensor

        Returns
        -------
         : np.ndarray
            Cartesian position vector of the sensor

        """

        fl = 0.0335  # The oblateness or flattening of the Earth.
        r_e = const.R_earth.to(u.m).value  # Earth radius

        ssqlat = np.sin(self.latitude)**2
        clat = np.cos(self.latitude)
        slat = np.sin(self.latitude)

        sit = self.sidtime_rad().value
        csit = np.cos(sit)  # cos of the sidereal time
        ssit = np.sin(sit)  # sin of the sidereal time

        r_c = r_e/(np.sqrt(1 - (2*fl - fl**2)*ssqlat)) + self.elevation
        r_s = r_e*(1-fl)**2/(np.sqrt(1 - (2*fl - fl**2)*ssqlat)) + self.\
            elevation

        return np.array([[r_c * clat * csit],
                         [r_c * clat * ssit],
                         [r_s * slat]])

    def matrix_eci_to_topoh(self):
        """ This matrix rotates the ECI ECI coordinate to the topocentric horizon frame

        Returns
        -------
         : np.ndarray
            the matrix that rotates the ECI coordinate to the topocentric horizon coordinate
        """

        theta = self.sidtime_rad().value
        phi = self.latitude

        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        return np.array([[-stheta, ctheta, 0],
                         [-sphi*ctheta, -sphi*stheta, cphi],
                         [cphi*ctheta, cphi*stheta, sphi]])

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""The function which returns az, alt given an target with ECI coordinates and a measurement
        position and time.

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs()
            else:
                noise = 0

        # Compute the relative position in ECI
        rp = state.state_vector[self.mapping, :] - self.position_vector

        # To get it into topocentric horizon coordinates we need the
        # transformation matrix
        rt = self.matrix_eci_to_topoh() @ rp
        rmag = np.sqrt(rt[0] ** 2 + rt[1] ** 2 + rt[2] ** 2)
        nrt = rt / rmag

        # Turn this into an altitude and azimuth
        alt = np.arcsin(nrt[2])
        azi = np.arccos(nrt[1] / np.cos(alt))

        # Bit of jiggery required to get azimuth quadrant
        if nrt[0] / np.cos(alt) < 0:
            azi = 2 * np.pi - azi

        return StateVector([[Bearing(azi)], [Elevation(alt)]]) + noise

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[Bearing(0.)], [Elevation(0.)], [0.]]) + out
        return out
