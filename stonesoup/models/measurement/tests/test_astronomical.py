# coding: utf-8
import numpy as np
from datetime import datetime
from stonesoup.models.measurement.astronomical import ECItoAltAz
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.orbitalstate import OrbitalState
from stonesoup.types.angle import  Bearing, Elevation
from stonesoup.types.detection import Detection


def test_ecitoaltaz():
    """This is example 5.9 in [1]

    References
    ----------
    1. Curtis, H.D. 2010, Orbital Mechanics for Engineering Students 3rd Ed., Elsevier Aerospace
       Engineering Series
    """

    time = datetime(2001, 10, 12, 12, 0, 30)  # Assume this is utc.

    # The ECI position of the target is [-2032.4, 4591.2, -4544.8] (km)
    # Obviously the velocity component is wrong (and ignored)
    ostate = OrbitalState(np.array([[-2032400], [4591200], [-4544800], [0], [0], [0]]),
                          coordinates="Cartesian", timestamp=time)
    # The answer is az = 129.8 degrees, alt = 41.41 degrees
    z_gt = StateVector([Elevation(41.41*np.pi/180), Bearing(129.8*np.pi/180)])

    # Set up model
    cov = CovarianceMatrix([[0.01, 0], [0, 0.01]])  # 0.1 rad uncertainty in alt and az
    measurement_model = ECItoAltAz(cov, latitude=-40*np.pi/180, longitude=-91.25*np.pi/180)
    # This returns a local sidereal time of 110.0 radians

    # Execute measurement
    z = measurement_model.function(ostate, timestamp=time)

    assert np.allclose(z, z_gt, rtol=0.001)

    print(measurement_model.pdf(Detection(z), ostate))