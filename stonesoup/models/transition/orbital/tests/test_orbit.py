# coding: utf-8

from datetime import datetime

import numpy as np

from .....types.orbitalstate import OrbitalState
from ..orbit import CartesianKeplerianTransitionModel


# Define some orbital states to test
ini_cart = np.array([[7000], [-12124], [0], [2.6679], [4.621], [0]])

# Set the times
time1 = datetime(2011, 11, 12, 13, 45, 31)  # Just a time
time2 = datetime(2011, 11, 12, 14, 45, 31)  # An hour later
deltat = time2-time1

propagator_c = CartesianKeplerianTransitionModel()


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
