# -*- coding: utf-8 -*-
from abc import ABC

import numpy as np
from datetime import datetime

from ...base import Property
from ...types.detection import Detection
from ...types.array import StateVector
from ...types.angle import Elevation, Bearing
from ..passive import PassiveElevationBearing
from ...models.measurement.astronomical import ECItoAzAlt


class AllSkyTelescope(PassiveElevationBearing):
    """
    An simple instance of a telescope. This is a sensor with a location
    (longitude, latitude, elevation) capable of observing astronomical objects.
    This instance is simple in that it returns independent measurements of
    the azimuth, altitude. There is no field of view (hence the all-sky)
    but there is a minimum altitude.

    The uncertainty is Gaussian in azimuth/altitude.

    """
    # Position of the observatory
    latitude = Property(float, default=0.0,
                        doc="Observatory latitude (radians)")
    longitude = Property(float, default=0.0,
                         doc="Observatory longitude (radians)")
    elevation = Property(float, default=0.0,
                         doc="Observatory elevation (m)")

    # Properties of the detector
    e_fa = Property(float, default=1e-10, doc="Expected number of false alarms per observation "
                                              "(Poisson distributed)")
    p_d = Property(float, default=0.999999,
                   doc="Probability of target detection")

    minAlt = Property(float, default=np.pi/8,
                      doc="Minimum altitude above the horizon for an "
                          "observation (rad)")

    def observe(self, target, timestamp=datetime(2000, 1, 1, 12, 0, 0)):
        """Make an observation

        Parameters
        ----------
        target : :class:`~.Platform`
            a target - with a location defined in interpretable coordinates
            that may be in the field of view
        timestamp : :class:`~.datetime.datetime`
            The time and date as a datetime object. Assumed to be utm if naive. If tzinfo is
            present, then local time is assumed and corrected for. Default is 2000-01-01, 12:00:00.

        Returns
        -------
        : list of :class:~`Detection`
            a set of observations math:`z = [:class:`~.Elevation`, :class:`~.Bearing`]^T` (rad,
            rad), or an empty list

        Notes
        -----
        The list will be empty or comprise a combination of one true detection
        and any number of false alarms (depending on the false alarm rate).
        Only one true detection per observation is permitted.

        """

        # initialise the model
        measurement_model = ECItoAzAlt(self.noise_covar, latitude=self.latitude,
                                       longitude=self.longitude, elevation=self.elevation)

        measurement_vector = []  # Initialise empty list

        # Decide whether a false alarm is to be returned
        for ii in range(np.random.poisson(self.e_fa)):
            # Uniform in altitude, azimuth is simplest
            # And then Poissonian with parameter noise[2][2]?
            measurement_vector = np.append(
                measurement_vector,
                Detection(StateVector([[Elevation(np.random.uniform(self.minAlt, np.pi/2)),
                                        Bearing(np.random.uniform(-np.pi, np.pi))]]),
                          measurement_model=measurement_model, timestamp=timestamp))

        alaz = measurement_model.function(target, noise=True, timestamp=timestamp)

        # Check to see if the target is above minimum altitude for observation
        # if so, detect the target with a probability pd
        if alaz[0] > self.minAlt and np.random.uniform() < self.p_d:
            measurement_vector = np.append(measurement_vector, Detection(alaz))

        return measurement_vector
