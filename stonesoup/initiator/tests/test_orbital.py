import numpy as np
import pytest
from datetime import datetime, timedelta

from ...types.detection import Detection
from ...astronomical_conversions import local_sidereal_time
from ..preliminaryorbitdetermination import GibbsInitiator, LambertInitiator, \
    RangeAltAzInitiator, GaussInitiator


def test_gibbsinitiator():
    """Example 5.1 from [1]

    Using three position vectors initialise a track using the Gibbs
    initiator. Then check with the answer in the book. Note that [1]
    has rounding issues.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering Students,
        Third Edition, Elsevier"""

    # Define three position vectors
    r1 = Detection(np.array([[-294.32], [4265.1], [5986.7]]))
    r2 = Detection(np.array([[-1365.5], [3637.6], [6346.8]]))
    r3 = Detection(np.array([[-2940.3], [2473.7], [6555.8]]))

    # The solution given in [1] is:
    v2 = np.array([[-6.2174], [-4.0122], [1.599]])
    out_cart = np.concatenate((r2.state_vector, v2))

    # set up the input triple
    inlist = [(r1, r2, r3)]
    # and then initiator
    ginitiator = GibbsInitiator(grav_parameter=3.986004418e5)

    # Track initialise
    out_tracks = ginitiator.initiate(inlist)
    out_track = out_tracks.pop()

    assert(np.allclose(out_cart, out_track[1].state_vector, rtol=1e-5))


def test_lambert_initiator():
    """Examples 5.2 and 5.3 in [1]

    Using two timestamped position vectors to initialise a track using the
    Lambert/Laplace-based initiator. Then check with the answer in the book.
    Note that [1] has rounding issues.

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering Students,
        Third Edition, Elsevier

    """
    """Example 5.2"""
    # Create the initiator
    linitiator = LambertInitiator(grav_parameter=3.986004418e5)

    # Set a time interval
    time1 = datetime.now()
    time2 = time1 + timedelta(hours=1)

    # Two position vectors
    r1 = Detection(np.array([[5000], [10000], [2100]]), timestamp=time1)
    r2 = Detection(np.array([[-14600], [2500], [7000]]), timestamp=time2)

    # These are the answers we aim to arrive at
    v1 = np.array([[-5.9925], [1.9254], [3.2456]])
    v2 = np.array([[-3.3125], [-4.1966], [-0.38529]])
    oc1 = np.concatenate((r1.state_vector, v1))
    oc2 = np.concatenate((r2.state_vector, v2))

    # Do the calculation
    otracks = linitiator.initiate([(r1, r2)], directions=["prograde"])

    # Check that we got the answer right
    otrack = otracks.pop()  # Pop the only track off the set
    assert np.allclose(otrack[0].state_vector, oc1, rtol=1e-4)
    assert np.allclose(otrack[1].state_vector, oc2, rtol=1e-4)

    """Now Example 5.3"""
    # A new time interval
    time2 = time1 + timedelta(hours=13.5)

    # Two new position vectors
    r1 = Detection(np.array([[273378], [0], [0]]), timestamp=time1)
    r2 = Detection(np.array([[145820], [12758], [0]]), timestamp=time2)

    # This time there's a measured angular deviation
    dtrue_anomaly = 5/180 * np.pi # That's 5 degrees.

    # Set the answers to what the book says
    v1 = np.array([[-2.4356], [0.26741], [0]])
    oc1 = np.concatenate((r1.state_vector, v1))

    # Do the calculation
    otracks = linitiator.initiate([(r1, r2)], true_anomalies=[dtrue_anomaly])

    # And check the answer is right
    otrack = otracks.pop()
    assert np.allclose(otrack[0].state_vector, oc1, rtol=1e-4)


def test_ranaltaz_initiatior():
    """Example 5.10 in [1]

    Simulate an observation which measures range, altitude and azimuth
    and their rates

    Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering
        Students, Third Edition, Elsevier

    """

    # Set up the problem
    # Location of sensor
    latitude = 60*np.pi/180
    height = 0
    # Together the following give a local sidereal time of 300 degrees
    longitude = 199.8782*np.pi/180
    local_time = datetime(2020, 1, 1, 0, 0, 0)

    # The measurement
    range = 2551e3
    azimuth = np.pi/2
    altitude = np.pi/6
    drange = 0
    dazimuth = 1.973e-3
    daltitude = 9.864e-4

    # The answer will be
    out_state = np.array([[3831e3], [-2216e3], [6605e3],
                          [1504], [-4562], [-292]])

    # Create a detection
    detection = Detection(np.array([[range],
                                    [altitude],
                                    [azimuth],
                                    [drange],
                                    [daltitude],
                                    [dazimuth]]), timestamp=local_time)
    # Create the initiator
    rinitiator = RangeAltAzInitiator()

    # Initiate tracks
    otracks = rinitiator.initiate([detection], latitude, longitude, height)

    # Check
    otrack = otracks.pop()
    assert np.allclose(otrack[0].state_vector, out_state, rtol=1e-3)


def test_gauss_initiator():
    """ Example 5.11 in [1]

    Test initiator based on angles-only measurements

     Reference
    ---------
        1. Curtis, H. D. 2010, Orbital Mechanics for Engineering
        Students, Third Edition, Elsevier

    """

    # Set up the problem
    # Location of sensor
    latitude = 40 * np.pi/180
    height = 1000  # (m)
    # Together the following give the correct local sidereal times
    longitude = 304.3846 * np.pi / 180
    local_time = datetime(2020, 1, 1, 0, 0, 0)

    # Set up the measurements
    detections = (Detection(np.pi/180*np.array([[43.537], [-8.7833]]),
                            timestamp=local_time),
                  Detection(np.pi/180*np.array([[54.420], [-12.074]]),
                            timestamp=local_time + timedelta(seconds=118.1)),
                  Detection(np.pi/180*np.array([[64.318], [-15.105]]),
                            timestamp=local_time + timedelta(seconds=237.58))
                  )

    # The answer will be
    out_state = np.array([[5659.1e3], [6533.8e3], [3270.1e3],
                          [-3880], [5115.6], [-2238.7]])

    # Create the initiator
    ginitiator = GaussInitiator()

    # Initiate tracks
    otracks = ginitiator.initiate([detections], latitude, longitude, height)

    # Check
    otrack = otracks.pop()
    assert np.allclose(otrack[0].state_vector, out_state, rtol=1e-2)