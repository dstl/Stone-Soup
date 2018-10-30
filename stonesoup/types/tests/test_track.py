# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ..particle import Particle
from ..state import State, GaussianState, ParticleState
from ..track import Track


def test_track_empty():
    # Track initialisation without initial state
    track = Track()
    assert len(track) == 0


@pytest.mark.parametrize('state', [
    State(np.array([[0]]), datetime.datetime.now()),
    GaussianState(np.array([[0]]), np.array([[0]]), datetime.datetime.now()),
    ParticleState([Particle(np.array([[0]]), 1)], datetime.datetime.now()),
    ],
    ids=['State', 'GaussianState', 'ParticleState'])
def test_track_state(state):
    # Track initialisation with initial state
    track = Track([state])
    assert len(track) == 1
    assert track.state is state

    # All
    assert np.array_equal(state.state_vector, state.state_vector)
    assert track.timestamp == state.timestamp

    # Gaussian and Particle
    if hasattr(state, 'covar'):
        assert np.array_equal(track.covar, state.covar)
    else:
        with pytest.raises(AttributeError):
            track.covar

    # Particle
    if hasattr(state, 'particles'):
        assert track.particles == state.particles
    else:
        with pytest.raises(AttributeError):
            track.particles


def test_track_id():
    state = State(np.array([[0]]), datetime.datetime.now())
    track = Track([state])
    assert isinstance(track.id, str)

    track = Track([state], 'abc')
    assert isinstance(track.id, str)
    assert track.id == 'abc'
