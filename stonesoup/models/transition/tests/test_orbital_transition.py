# coding: utf-8

import numpy as np
from copy import copy, deepcopy
from datetime import datetime, timedelta
from ....types.orbitalstate import TLEOrbitalState, OrbitalState
from ..orbital.orbit import SimpleMeanMotionTransitionModel, CartesianTransitionModel

# Test the class SimpleMeanMotionTransitionModel class
def test_meanmotion_transition():
    """Tests SimpleMeanMotionTransitionModel()"""

    # Define the initial orbital state
    out_tle = np.array([[2.674], [4.456], [0.1712], [0.3503], [0.3504],
                        [0.0007662]])

    time1 = datetime(2011, 11, 12, 13, 45, 31) # Just a time
    time2 = datetime(2011, 11, 12, 14, 46, 32) # An hour, one minute, one second later
    deltat = time2-time1

    initial_state = TLEOrbitalState(out_tle, timestamp=time1)
    propagator = SimpleMeanMotionTransitionModel()
    final_state = propagator.transition(initial_state, deltat)

    # Check something's happened
    assert not np.all(initial_state.two_line_element == final_state.two_line_element)

    final_meananomaly = np.remainder(out_tle[4][0] + 3661*out_tle[5][0],
                                     2*np.pi)

    assert np.isclose(np.remainder(initial_state.mean_anomaly, 2*np.pi),
                      np.remainder(out_tle[4][0], 2*np.pi), rtol=1e-8)
    assert np.isclose(np.remainder(final_state.mean_anomaly, 2*np.pi),
                      np.remainder(final_meananomaly, 2*np.pi), rtol=1e-8)


# Need to remember how to do this without repeated code
def test_cartesiantransitionmodel():
    """Tests the CartesianTransitionModel():

    Example 3.7 in [1]

    """

    # Define the initial orbital state
    ini_cart = np.array([[7000], [-12124], [0], [2.6679], [4.621], [0]])

    # The book tells me the answer is:
    fin_cart = np.array([[-3296.8], [7413.9], [0], [-8.2977], [-0.96309], [0]])
    # But that appears only to be accurate to 1 part in 1000. (Due to rounding in
    # the examples)
    # I think the answer is more like (but not sure how rounding done in book!)
    #fin_cart = np.array([[-3297.2], [7414.2], [0], [-8.2974], [-0.96295], [0]])

    time1 = datetime(2011, 11, 12, 13, 45, 31)  # Just a time
    time2 = datetime(2011, 11, 12, 14, 45, 31)  # An hour later
    deltat = time2 - time1

    initial_state = OrbitalState(ini_cart, coordinates="Cartesian", timestamp=time1)
    initial_state.grav_parameter = 398600

    propagator = CartesianTransitionModel()
    final_state = propagator.transition(initial_state, deltat)

    # Check something's happened
    assert not np.all(initial_state.keplerian_elements == final_state.keplerian_elements)

    # Check the elements match the book. But beware rounding
    assert np.allclose(final_state.cartesian_state_vector, fin_cart, rtol=1e-3)


def test_meanm_cart_ransition():
    """Test two transition models against each other"""

    # Set the times up
    time = datetime(2011, 11, 12, 13, 45, 31)  # Just a time
    deltat = timedelta(minutes=3)

    # Initialise the state vector
    ini_cart = np.array([[7000], [-12124], [0], [2.6679], [4.621], [0]])
    initial_state = OrbitalState(ini_cart, coordinates="Cartesian", timestamp=time)
    initial_state.grav_parameter = 398600

    # Define some propagators
    propagator1 = CartesianTransitionModel()
    propagator2 = SimpleMeanMotionTransitionModel()
    state1 = initial_state
    state2 = deepcopy(state1)

    assert (np.allclose(state1.cartesian_state_vector, state2.cartesian_state_vector, rtol=1e-8))

    # Dumb way to do things
    while time < datetime(2011, 11, 12, 13, 45, 31) + timedelta(hours=10):
        state1 = propagator1.transition(state1, deltat)
        state2 = propagator2.transition(state2, deltat)
        assert (np.allclose(state1.cartesian_state_vector, state2.cartesian_state_vector, rtol=1e-8))
        time = time + deltat
