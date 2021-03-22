# coding: utf-8

from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np

# https://pypi.org/project/sgp4/
from sgp4.api import Satrec
from sgp4.api import jday

from .....types.orbitalstate import OrbitalState
from .....types.array import StateVector, CovarianceMatrix
from ..orbit import CartesianKeplerianTransitionModel, TLEKeplerianTransitionModel, \
    SGP4TransitionModel


# Define some orbital states to test
ini_cart = np.array([[7000], [-12124], [0], [2.6679], [4.621], [0]])
out_tle = StateVector([2.674, 4.456, 0.1712, 0.3503, 0.3504, 0.0007662])

# Set the times
time1 = datetime(2011, 11, 12, 13, 45, 31)  # Just a time
time2 = datetime(2011, 11, 12, 14, 45, 31)  # An hour later
deltat = time2-time1

# Different types of propagator
propagator_c = CartesianKeplerianTransitionModel()
propagator_sm = TLEKeplerianTransitionModel()


def test_cartesiantransitionmodel():
    """Tests the CartesianKeplerianTransitionModel():

    Example 3.7 in [1]

    """
    # The book tells me the answer is:
    fin_cart = np.array([[-3296.8], [7413.9], [0], [-8.2977], [-0.96309], [0]])
    # But that appears only to be accurate to 1 part in 1000. (Due to rounding
    # in the examples)
    # I think the answer is more like (but not sure how rounding done in book!)
    # fin_cart = np.array([[-3297.2], [7414.2], [0], [-8.2974], [-0.96295],
    # [0]])

    initial_state = OrbitalState(ini_cart, coordinates="Cartesian",
                                 timestamp=time1, grav_parameter=398600)

    final_state = OrbitalState(propagator_c.transition(initial_state, time_interval=deltat),
                               timestamp=time1+deltat, grav_parameter=398600)

    # Check something's happened
    assert not np.all(initial_state.keplerian_elements ==
                      final_state.keplerian_elements)

    # Check the elements match the book. But beware rounding
    assert np.allclose(final_state.cartesian_state_vector, fin_cart, rtol=1e-3)


# Test the class TLEKeplerianTransitionModel class
def test_meanmotion_transition():
    """Tests SimpleMeanMotionTransitionModel()"""

    initial_state = OrbitalState(out_tle, timestamp=time1, coordinates='TLE')
    final_state = OrbitalState(
        propagator_sm.transition(initial_state, noise=False, time_interval=deltat),
        timestamp=time1 + deltat, coordinates='TLE')

    # Check something's happened
    assert not np.all(initial_state.two_line_element == final_state.two_line_element)

    # Check the final state is correct
    final_meananomaly = np.remainder(out_tle[4] + out_tle[5]*deltat.total_seconds(), 2*np.pi)

    assert np.isclose(float(initial_state.mean_anomaly), float(out_tle[4]), rtol=1e-8)
    assert np.isclose(float(final_state.mean_anomaly), float(final_meananomaly), rtol=1e-8)


def test_meanm_cart_transition():
    """Test two Keplerian transition models against each other"""

    # Set the times up
    time = time1
    dt = timedelta(minutes=3)

    # Initialise the state vector
    initial_state = OrbitalState(ini_cart, coordinates="Cartesian",
                                 timestamp=time, grav_parameter=398600)

    # Make state copies to recurse
    state1 = initial_state
    state2 = deepcopy(state1)

    assert (np.allclose(state1.cartesian_state_vector,
                        state2.cartesian_state_vector, rtol=1e-8))

    # Dumb way to do things
    while time < datetime(2011, 11, 12, 13, 45, 31) + timedelta(minutes=6):
        state1 = OrbitalState(propagator_c.transition(state1, time_interval=dt),
                              timestamp=time, grav_parameter=398600)
        state2 = OrbitalState(propagator_sm.transition(state2, noise=False, time_interval=dt),
                              timestamp=time, coordinates='TLE', grav_parameter=398600)
        assert (np.allclose(state1.state_vector, state2.state_vector,
                            rtol=1e-8))
        time = time + dt


def test_sampling():
    """test the sampling functions work"""

    # Initialise the state vector
    initial_state = OrbitalState(ini_cart, coordinates="Cartesian", timestamp=time1)
    initial_state.grav_parameter = 398600

    # noise it up
    propagator_sm.process_noise = \
        CovarianceMatrix(np.diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-2, 1e-10]))
    # take some samples
    samp_states_sm = propagator_sm.rvs(num_samples=10, time_interval=deltat)

    # Do it in Cartesian
    propagator_c.process_noise = CovarianceMatrix(np.diag([10, 10, 10, 10e-3, 10e-3, 10e-3]))
    samp_states_c = propagator_c.rvs(num_samples=100, time_interval=deltat)

    for state in samp_states_c.T:
        assert(propagator_c.pdf(OrbitalState(state,
                                             timestamp=initial_state.timestamp
                                             + deltat,
                                             grav_parameter=initial_state.
                                             grav_parameter),
                                initial_state, time_interval=deltat) >= 0)
        # PDF must be positive

    for state in samp_states_sm.T:
        assert(propagator_sm.pdf(OrbitalState(state,
                                              timestamp=initial_state.timestamp
                                              + deltat,
                                              grav_parameter=initial_state.
                                              grav_parameter),
                                 initial_state, time_interval=deltat) >= 0)


def test_SGP4TransitionModel():
    # Propagate a TLE, evaluate position at an initial time
    initial_date = datetime(2008, 9, 20, 12, 25, 40, 104192)
    final_date = initial_date + timedelta(hours=1)

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

    # Initialise the state (Grav parameter in m)
    test_tle = OrbitalState(None, coordinates='TLE', timestamp=initial_date, metadata=tle_dict)

    # Evaluate position at initial date using external SGP4
    jd, fr = jday(initial_date.year, initial_date.month, initial_date.day,
                  initial_date.hour, initial_date.minute, initial_date.second)
    e, r, v = tle_ext.sgp4(jd, fr)

    # Evaluate position at initial date using wrapped SGP4 library
    testSGP4 = SGP4TransitionModel()
    outrv = testSGP4.transition(test_tle)

    # Check position vectors are equal (convert to m)
    assert np.allclose(1000*StateVector(r), outrv[0:3])

    # Evaluate position at final date using external SGP4
    jd, fr = jday(final_date.year, final_date.month, final_date.day,
                  final_date.hour, final_date.minute, final_date.second)
    e, r, v = tle_ext.sgp4(jd, fr)

    # Evaluate position at final date using wrapped SGP4 library
    outrv = testSGP4.transition(test_tle, time_interval=timedelta(hours=1))

    assert np.allclose(1000*StateVector(r), outrv[0:3])
