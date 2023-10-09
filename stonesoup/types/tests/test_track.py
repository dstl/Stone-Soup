import copy
import datetime

import numpy as np
import pytest

from ..detection import Detection
from ..hypothesis import SingleHypothesis
from ..numeric import Probability
from ..state import State, GaussianState, ParticleState
from ..track import Track
from ..update import Update


def test_track_empty():
    # Track initialisation without initial state
    track = Track()
    assert len(track) == 0


@pytest.mark.parametrize('state', [
    State(np.array([[0]]), datetime.datetime.now()),
    GaussianState(np.array([[0]]), np.array([[0]]), datetime.datetime.now()),
    ParticleState([[0]], datetime.datetime.now(), [Probability(1)]),
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
        for track_particle, state_particle in zip(track.particles, state.particles):
            assert track_particle.state_vector == state_particle.state_vector
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

    track = Track(id=None)
    assert isinstance(track.id, str)


def test_track_metadata():
    track = Track()
    assert track.metadata == {}
    assert not track.metadatas

    track = Track(init_metadata={'colour': 'blue'})

    assert track.metadata == {'colour': 'blue'}
    assert not track.metadatas

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'side': 'ally'}))
    )
    track.append(state)
    assert track.metadata == {'colour': 'blue', 'side': 'ally'}
    assert len(track.metadatas) == 1
    assert track.metadata == track.metadatas[-1]

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'side': 'enemy'}))
    )
    track.append(state)
    assert track.metadata == {'colour': 'blue', 'side': 'enemy'}
    assert len(track.metadatas) == 2

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'colour': 'red'}))
    )
    track[0] = state
    assert track.metadata == track.metadatas[-1] == {'colour': 'red', 'side': 'enemy'}
    assert len(track.metadatas) == 2
    assert track.metadatas[0] == {'colour': 'red'}

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'speed': 'fast'}))
    )
    track.insert(1, state)
    assert track.metadata == {'colour': 'red', 'side': 'enemy', 'speed': 'fast'}
    assert len(track.metadatas) == 3
    assert track.metadatas[0] == {'colour': 'red'}
    assert track.metadatas[1] == {'colour': 'red', 'speed': 'fast'}
    assert track.metadatas[2] == {'colour': 'red', 'side': 'enemy', 'speed': 'fast'}

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'size': 'small'}))
    )
    track.insert(-1, state)
    assert track.metadata == {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert len(track.metadatas) == 4
    assert track.metadatas[0] == {'colour': 'red'}
    assert track.metadatas[1] == {'colour': 'red', 'speed': 'fast'}
    assert track.metadatas[2] == {'colour': 'red', 'speed': 'fast', 'size': 'small'}
    assert track.metadatas[3] == \
           {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'colour': 'black'}))
    )
    track.insert(-100, state)
    assert track.metadata == {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert len(track.metadatas) == 5
    assert track.metadatas[0] == {'colour': 'black'}
    assert track.metadatas[1] == {'colour': 'red'}
    assert track.metadatas[2] == {'colour': 'red', 'speed': 'fast'}
    assert track.metadatas[3] == {'colour': 'red', 'size': 'small', 'speed': 'fast'}
    assert track.metadatas[4] == \
           {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'colour': 'black'}))
    )
    track.insert(100, state)
    assert track.metadata == {'colour': 'black', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert len(track.metadatas) == 6
    assert track.metadatas[0] == {'colour': 'black'}
    assert track.metadatas[1] == {'colour': 'red'}
    assert track.metadatas[2] == {'colour': 'red', 'speed': 'fast'}
    assert track.metadatas[3] == {'colour': 'red', 'size': 'small', 'speed': 'fast'}
    assert track.metadatas[4] == \
           {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert track.metadatas[5] == \
           {'colour': 'black', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'colour': 'green'}))
    )
    track.append(state)
    assert track.metadata == {'colour': 'green', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert len(track.metadatas) == 7
    assert track.metadatas[0] == {'colour': 'black'}
    assert track.metadatas[1] == {'colour': 'red'}
    assert track.metadatas[2] == {'colour': 'red', 'speed': 'fast'}
    assert track.metadatas[3] == {'colour': 'red', 'size': 'small', 'speed': 'fast'}
    assert track.metadatas[4] == \
           {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert track.metadatas[5] == \
           {'colour': 'black', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert track.metadatas[6] == \
           {'colour': 'green', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}

    state = Update(
        hypothesis=SingleHypothesis(None, Detection(np.array([[0]]), metadata={'colour': 'white'}))
    )
    track[-2] = state
    assert track.metadata == {'colour': 'green', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert len(track.metadatas) == 7
    assert track.metadatas[0] == {'colour': 'black'}
    assert track.metadatas[1] == {'colour': 'red'}
    assert track.metadatas[2] == {'colour': 'red', 'speed': 'fast'}
    assert track.metadatas[3] == {'colour': 'red', 'size': 'small', 'speed': 'fast'}
    assert track.metadatas[4] == \
           {'colour': 'red', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert track.metadatas[5] == \
           {'colour': 'white', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}
    assert track.metadatas[6] == \
           {'colour': 'green', 'side': 'enemy', 'speed': 'fast', 'size': 'small'}


def test_copy():
    metadatas = [{'update_number': i} for i in range(3)]

    track = Track([State([0, 1]),
                   State([1, 2]),
                   State([3, 4])],
                  init_metadata=metadatas[0])
    track.metadatas = metadatas

    copied_track = copy.copy(track)

    assert track is not copied_track
    assert track.states is not copied_track.states
    assert track.init_metadata is copied_track.init_metadata
    assert track.metadatas is not copied_track.metadatas

    for original_state, copied_state in zip(track.states, copied_track.states):
        assert original_state is copied_state

    for original_metadata, copied_metadata in zip(track.metadatas, copied_track.metadatas):
        assert original_metadata is copied_metadata

    assert len(track) == 3
    assert len(copied_track) == 3
    copied_track.append(State([4, 5]))

    assert len(track) == 3
    assert len(copied_track) == 4

    assert len(track.metadatas) == 3
    assert len(copied_track.metadatas) == 4
