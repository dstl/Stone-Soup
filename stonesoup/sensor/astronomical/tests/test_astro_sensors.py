# -*- coding: utf-8 -*-
from datetime import datetime
from pytest import approx
import numpy as np

from stonesoup.sensor.astronomical.electrooptic import AllSkyTelescope
from stonesoup.types.orbitalstate import OrbitalState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.angle import Elevation, Bearing


def test_electrooptic():
    """

    """
    time = datetime(2001, 10, 12, 12, 0, 30)  # Assume this is utc.

    # The ECI position of the target is [-2032.4, 4591.2, -4544.8] (km)
    # Obviously the velocity component is wrong (and ignored)
    ostate = OrbitalState(np.array([[-2032400], [4591200], [-4544800], [0], [0], [0]]),
                          coordinates="Cartesian", timestamp=time)
    # The answer is az = 129.8 degrees, alt = 41.41 degrees
    z_gt = StateVector([Elevation(41.41 * np.pi / 180), Bearing(129.8 * np.pi / 180)])

    # Set up model
    cov = CovarianceMatrix([[0.01, 0], [0, 0.01]])  # 0.1 rad uncertainty in alt and az
    # This returns a local sidereal time of 110.0 radians

    latitude = -40*np.pi/180
    longitude = -91.25*np.pi/180
    elevation = 0

    false_alarm_rate = 1
    prob_det = 0.7

    sensor = AllSkyTelescope(6, np.array([0, 1, 2]), cov, latitude=latitude, longitude=longitude,
                             elevation=elevation, e_fa=false_alarm_rate, p_d=prob_det, min_alt=0)

    measurements = sensor.observe(ostate, timestamp=time)

    assert len(measurements) >= 0
