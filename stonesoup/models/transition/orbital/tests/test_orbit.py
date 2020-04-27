from .....types.orbitalstate import TLEOrbitalState
from ..orbit import SGP4TransitionModel

# https://pypi.org/project/sgp4/
from sgp4.api import Satrec
from sgp4.api import jday
import numpy as np
from datetime import datetime


def test_SGP4TransitionModel():
    # Propagate a TLE, evaluate position at an initial time
    initial_date = datetime(2019, 12, 9, 12, 0, 0)

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

    # A state vector in TLE format
    state_vector = np.zeros(shape=(6, 1))
    state_vector[0, 0] = tle_ext.inclo
    state_vector[1, 0] = tle_ext.nodeo
    state_vector[2, 0] = tle_ext.ecco
    state_vector[3, 0] = tle_ext.argpo
    state_vector[4, 0] = tle_ext.no
    state_vector[5, 0] = tle_ext.no_kozai

    # Initialise the state
    test_tle = TLEOrbitalState(state_vector, timestamp=initial_date,
                               metadata=tle_dict)

    # Evaluate position at initial date using external SGP4
    jd, fr = jday(initial_date.year, initial_date.month, initial_date.day,
                  initial_date.hour, initial_date.minute, initial_date.second)
    e, r, v = tle_ext.sgp4(jd, fr)

    # Evaluate position at initial date using wrapped SGP4 library
    testSGP4 = SGP4TransitionModel()
    outrv = testSGP4.transition(test_tle)

    # Note that if we use outrv  - which is a StateVector to update the State,
    # we're going to have to ensure consistency of the metadata

    # Check position vectors are equal
    assert (np.array_equal(r, outrv[0:3]))
