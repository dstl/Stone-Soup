# -*- coding: utf-8 -*-
import datetime

import numpy as np

from stonesoup.types import Track, State


def test_track():
    """ Track Type test """

    state_vec = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    timestamp = datetime.datetime.now()
    state = State(state_vec, timestamp)

    # Track initialisation without initial state
    track = Track()
    assert(not track.states)

    # Track initialisation with initial state
    track = Track(state)
    assert(len(track.states) == 1)
    assert(track.state == state)
