# coding: utf-8

import numpy as np
from datetime import datetime, timedelta
from ....types.orbitalstate import TLEOrbitalState
from ..orbital.orbit import SimpleMeanMotionTransitionModel

# Test the class SimpleMeanMotionTransitionModel class
def test_meanmotion_transition():
    """"""

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
