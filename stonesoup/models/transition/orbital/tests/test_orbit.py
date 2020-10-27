from .....types.array import StateVector
from .....types.orbitalstate import TLEOrbitalState
from ..orbit import SGP4TransitionModel

# https://pypi.org/project/sgp4/
from sgp4.api import Satrec
from sgp4.api import jday
import numpy as np
from datetime import datetime


def test_SGP4TransitionModel():
    # Propagate a TLE, evaluate position at an initial time
    initial_date = datetime(2008, 9, 20, 12, 25, 40, 104192)

    # https://en.wikipedia.org/wiki/Two-line_element_set
    line_1 = \
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line_2 = \
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

    # TLE representations
    # 1:SGP4 object form from external library
    tle_ext = Satrec.twoline2rv(line_1, line_2)
    # 2: String format
    tle_dict = {'line_1': line_1, 'line_2': line_2}

    # Initialise the state
    test_tle = TLEOrbitalState(None, metadata=tle_dict)

    # Evaluate position at initial date using external SGP4
    jd, fr = jday(initial_date.year, initial_date.month, initial_date.day,
                  initial_date.hour, initial_date.minute, initial_date.second)
    e, r, v = tle_ext.sgp4(jd, fr)

    # Evaluate position at initial date using wrapped SGP4 library
    testSGP4 = SGP4TransitionModel()
    outrv = testSGP4.transition(test_tle)

    # Check position vectors are equal
    assert (np.array_equal(StateVector(r), outrv[0:3]))
