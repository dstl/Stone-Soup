import numpy as np

from datetime import datetime

from astropy import constants as const
from astropy import units as u
from astropy import time as tim

from ...base import Property
from .base import MeasurementModel


class SimpleObservatory(MeasurementModel):
    """
    An instance of an 'Observatory'. This is a sensor with a location
    (longitude, latitude, elevation) capable of observing astronomical objects.
    This instance is 'simple' in that it returns independent measurements of
    the azimuth, altitude and range.

    The estimate of the uncertainty is left to be defined by the user.

    The measurement is made by converting sensor and target into Earth-centred
    inertial (ECI) coordinates and then working out the relative position and
    velocity vectors and finally converting back to desired quantities.
    Coordinate transforms are done using Astropy. Astropy has several language
    conventions that are worth knowing.

    """
    latitude = Property(float, default=0.0,
                        doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0,
                         doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.9,
                         doc="Observatory elevation (m)")

    p_fa = Property(float, default=1e-10, doc="Probability of false alarm")
    p_d = Property(float, default=0.999999,
                   doc="Probability of target detection")

    noise = Property(np.array, default=None, doc="Covariance of the noise")

    ndim_state = 3

    latitude = Property(float, default=0.0,
                        doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0,
                         doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.0, doc="Observatory elevation (m)")
    timestamp = Property(datetime, default=946728000.0,
                         doc="Timestamp, defaults to 2000-01-01 12:00:00")

    p_fa = Property(float, default=1e-10,
                    doc="False alarm rate (expected number per observation")
    p_d = Property(float, default=0.999999,
                   doc="Probability of target detection")
    r_fa = Property(float, default=300000,
                    doc="Expected false alarm range (m)")

    noise = Property(np.array, default=np.diag(np.array([np.pi/100,
                                                         np.pi/100, 1e3])),
                     doc="Covariance of the noise")

    minAlt = Property(float, default=np.pi/8,
                      doc="Minimum altitude above the horizon for an "
                          "observation")

    @property
    def ndim_meas(self):
        return 3

    def function(self, target):
        print("Using the much more descriptive function, 'observe'")
        self.observe(target)

    def rvs(self):
        pass

    def pdf(self):
        pass

    def observe(self, target):
        """Make an observation

        Parameters
        ----------
        target : :class:~`Platform`
            a target - with a location defined in interpretable coordinates
            that may be in the field of view

        Returns
        -------
        : list of :class:~`State`
            a set of observations math:`z = [Azimuth, Altitude, range]^T` (rad,
            rad, m), or an empty list

        Notes
        -----
        The list will be empty or comprise a combination of one true detection
        and any number of false alarms (depending on the false alarm rate).
        Only one true detection per observation is permitted.

        """

        z = []  # Initialise empty list

        # Decide whether a false alarm is to be returned
        for ii in range(np.random.poisson(self.p_fa)):
            # Uniform in altitude, azimuth is simplest
            # And then Poissonian with parameter noise[2][2]?
            z = [z, np.array([[np.random.uniform(0, 2*np.pi)],
                              [np.random.uniform(self.minAlt, np.pi)],
                              [np.random.poisson(self.r_fa)]])]

        # Compute the relative position in ECI
        rp = target.position_vector() - self.position_vector()

        # To get it into topocentric horizon coordinates we need the
        # transformation matrix
        rt = self.matrix_eci_to_topoh() @ rp
        rmag = np.sqrt(rt[0]**2 + rt[1]**2 + rt[2]**2)
        nrt = rt/rmag

        # Turn this into an altitude and azimuth
        alt = np.arcsin(nrt[2])
        azi = np.arccos(nrt[1] / np.cos(alt))

        # Bit of jiggery required to get azimuth quadrant
        if nrt[0]/np.cos(alt) < 0:
            azi = 2*np.pi - azi

        # Check to see if the target is above minimum altitude for observation
        if alt > self.minAlt:
            # If so, detect the target with a probability pd
            if np.random.uniform() < self.p_d:
                #
                zz = np.random.multivariate_normal(np.array([azi[0], alt[0],
                                                             rmag[0]]),
                                                   self.noise)
                print(zz)
                z = np.append(z, zz.transpose())

        return z

    def position_vector(self):
        """

        :return: the position vector of the sensor

        """

        fl = 0.0335  # The oblateness or flattening of the Earth.
        r_e = const.R_earth.to(u.m).value  # Earth radius (is this the only
        # reason to invoke astropy?

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
        """

        :return: the matrix that rotates the ECI coordinate to the topocentric
        horizon coordinate
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

    def sidtime_rad(self):
        """

        :return: local mean sidereal time in units of radians
        """
        time = tim.Time(self.timestamp)
        sid_t = time.sidereal_time('mean',
                                   longitude=self.longitude)  # The sidereal
        # time - rendered in units of hour angle
        return sid_t.to(u.rad)  # local mean sidereal time in radian
