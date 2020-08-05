# -*- coding: utf-8 -*-
from abc import ABC

import numpy as np
from datetime import datetime

from ...base import Property
from ...types.detection import Detection
from ..sensor import Sensor
from ...models.measurement.astronomical import ECItoAzAlt


class AllSkyTelescope(Sensor, ABC):
    """
    An simple instance of a telescope. This is a sensor with a location
    (longitude, latitude, elevation) capable of observing astronomical objects.
    This instance is simple in that it returns independent measurements of
    the azimuth, altitude and range. There is no field of view (hence the all-sky)
    but there is a minimum altitude.

    The uncertainty is Gaussian in azimuth/altitude.

    """

    timestamp = Property(datetime, default=946728000.0,
                         doc="Timestamp, defaults to 2000-01-01 12:00:00")

    # Position of the observatory
    latitude = Property(float, default=0.0,
                        doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0,
                         doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.9,
                         doc="Observatory elevation (m)")

    # Properties of the detector
    p_fa = Property(float, default=1e-10, doc="Probability of false alarm")
    p_d = Property(float, default=0.999999,
                   doc="Probability of target detection")
    r_fa = Property(float, default=300000,
                    doc="Expected false alarm range (m)")

    minAlt = Property(float, default=np.pi/8,
                      doc="Minimum altitude above the horizon for an "
                          "observation")
    # The noise covariance
    noise = Property(np.array, default=None, doc="Covariance of the noise")

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
            a set of observations math:`z = [Azimuth, Altitude]^T` (rad,
            rad), or an empty list

        Notes
        -----
        The list will be empty or comprise a combination of one true detection
        and any number of false alarms (depending on the false alarm rate).
        Only one true detection per observation is permitted.

        """

        measurement_vector = []  # Initialise empty list

        # Decide whether a false alarm is to be returned
        for ii in range(np.random.poisson(self.p_fa)):
            # Uniform in altitude, azimuth is simplest
            # And then Poissonian with parameter noise[2][2]?
            measurement_vector = [measurement_vector,
                                  np.array([[np.random.uniform(0, 2*np.pi)],
                                            [np.random.uniform(self.minAlt, np.pi)]])]

        measurement_model = ECItoAzAlt(timestamp=self.timestamp, latitude=self.latitude,
                                       longitude=self.longitude, elevation=self.elevation)

        azal = measurement_model.function(target, noise=self.noise)

        # Check to see if the target is above minimum altitude for observation
        if azal[1] > self.minAlt:
            # If so, detect the target with a probability pd
            if np.random.uniform() < self.p_d:
                measurement_vector = np.append(measurement_vector, azal)

        return Detection(measurement_vector, measurement_model=measurement_model,
                         timestamp=self.timestamp)
