# -*- coding: utf-8 -*-
import datetime

import numpy as np
import scipy.linalg
import pytest

from stonesoup.base import Property
from ..angle import Bearing
from ..array import StateVector, CovarianceMatrix
from ..numeric import Probability
from ..particle import Particle
from ..state import State, GaussianState, ParticleState, StateMutableSequence, \
    WeightedGaussianState, SqrtGaussianState, CompositeState


def test_state():
    with pytest.raises(TypeError):
        State()

    # Test state initiation without timestamp
    state_vector = StateVector([[0], [1]])
    state = State(state_vector)
    assert np.array_equal(state.state_vector, state_vector)

    # Test state initiation with timestamp
    timestamp = datetime.datetime.now()
    state = State(state_vector, timestamp=timestamp)
    assert state.timestamp == timestamp


def test_state_invalid_vector():
    with pytest.raises(ValueError):
        State(StateVector([[[1, 2, 3, 4]]]))


def test_gaussianstate():
    """ GaussianState Type test """

    with pytest.raises(TypeError):
        GaussianState()

    mean = StateVector([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = CovarianceMatrix([[2.2128, 0, 0, 0],
                              [0.0002, 2.2130, 0, 0],
                              [0.3897, -0.00004, 0.0128, 0],
                              [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    timestamp = datetime.datetime.now()

    # Test state initiation without timestamp
    state = GaussianState(mean, covar)
    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))
    assert(state.ndim == mean.shape[0])
    assert(state.timestamp is None)

    # Test state initiation with timestamp
    state = GaussianState(mean, covar, timestamp)
    assert(np.array_equal(mean, state.mean))
    assert(np.array_equal(covar, state.covar))
    assert(state.ndim == mean.shape[0])
    assert(state.timestamp == timestamp)


def test_gaussianstate_invalid_covar():
    mean = StateVector([[1], [2], [3], [4]])  # 4D
    covar = CovarianceMatrix(np.diag([1, 2, 3]))  # 3D
    with pytest.raises(ValueError):
        GaussianState(mean, covar)


def test_sqrtgaussianstate():
    """Test the square root Gaussian Type"""

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0.1, 0.03, 0.01],
                      [0.1, 2.2130, 0.03, 0.02],
                      [0.03, 0.03, 2.123, 0.01],
                      [0.01, 0.02, 0.01, 2.012]]) * 1e3
    timestamp = datetime.datetime.now()

    # Test that a lower triangular matrix returned when 'full' covar is passed
    lower_covar = np.linalg.cholesky(covar)
    state = SqrtGaussianState(mean, lower_covar, timestamp=timestamp)
    assert np.array_equal(state.sqrt_covar, lower_covar)
    assert np.allclose(state.covar, covar, 0, atol=1e-10)
    assert np.allclose(state.sqrt_covar @ state.sqrt_covar.T, covar, 0, atol=1e-10)
    assert np.allclose(state.sqrt_covar @ state.sqrt_covar.T, lower_covar @ lower_covar.T, 0,
                       atol=1e-10)

    # Test that a general square root matrix is also a solution
    general_covar = scipy.linalg.sqrtm(covar)
    another_state = SqrtGaussianState(mean, general_covar, timestamp=timestamp)
    assert np.array_equal(another_state.sqrt_covar, general_covar)
    assert np.allclose(state.covar, covar, 0, atol=1e-10)
    assert not np.allclose(another_state.sqrt_covar, lower_covar, 0, atol=1e-10)


def test_weighted_gaussian_state():
    mean = StateVector([[1], [2], [3], [4]])  # 4D
    covar = CovarianceMatrix(np.diag([1, 2, 3]))  # 3D
    weight = 0.3
    timestamp = datetime.datetime.now()
    with pytest.raises(ValueError):
        WeightedGaussianState(mean, covar, timestamp, weight)

    # Test initialization using a GuassianState
    mean = StateVector([[1], [2], [3], [4]])  # 4D
    covar = CovarianceMatrix(np.diag([1, 2, 3, 4]))
    weight = 0.3
    gs = GaussianState(mean, covar, timestamp=timestamp)
    wgs = WeightedGaussianState.from_gaussian_state(gaussian_state=gs, weight=weight)
    assert np.array_equal(gs.state_vector, wgs.state_vector)
    assert np.array_equal(gs.covar, wgs.covar)
    assert gs.timestamp == wgs.timestamp
    assert weight == wgs.weight
    assert wgs.state_vector is not gs.state_vector
    assert wgs.covar is not gs.covar

    # Test copy flag
    wgs = WeightedGaussianState.from_gaussian_state(gaussian_state=gs, copy=False)
    assert wgs.state_vector is gs.state_vector
    assert wgs.covar is gs.covar

    # Test gaussian_state property
    gs2 = wgs.gaussian_state
    assert np.array_equal(gs.state_vector, gs2.state_vector)
    assert np.array_equal(gs.covar, gs2.covar)
    assert gs.timestamp == gs2.timestamp


def test_particlestate():
    with pytest.raises(TypeError):
        ParticleState()

    # 1D
    num_particles = 10
    state_vector1 = StateVector([[0.]])
    state_vector2 = StateVector([[100.]])
    weight = Probability(1/num_particles)
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    # Test state without timestamp
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, StateVector([[50]]))
    assert np.allclose(state.covar, CovarianceMatrix([[2500]]))

    # Test state with timestamp
    timestamp = datetime.datetime.now()
    state = ParticleState(particles, timestamp=timestamp)
    assert np.allclose(state.state_vector, StateVector([[50]]))
    assert np.allclose(state.covar, CovarianceMatrix([[2500]]))
    assert state.timestamp == timestamp

    # 2D
    state_vector1 = StateVector([[0.], [0.]])
    state_vector2 = StateVector([[100.], [200.]])
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    state = ParticleState(particles)
    assert isinstance(state, State)
    assert ParticleState in State.subclasses
    assert np.allclose(state.state_vector, StateVector([[50], [100]]))
    assert np.allclose(state.covar, CovarianceMatrix([[2500, 5000], [5000, 10000]]))


def test_particlestate_weighted():
    num_particles = 10

    # Half particles at high weight at 0
    state_vector1 = StateVector([[0.]])
    weight1 = Probability(0.75 / (num_particles / 2))

    # Other half of particles low weight at 100
    state_vector2 = StateVector([[100]])
    weight2 = Probability(0.25 / (num_particles / 2))

    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight1) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight2) for _ in range(num_particles//2))

    # Check particles sum to 1 still
    assert pytest.approx(1) == sum(particle.weight for particle in particles)

    # Test state vector is now weighted towards 0 from 50 (non-weighted mean)
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, StateVector([[25]]))
    assert np.allclose(state.covar, CovarianceMatrix([[1875]]))


def test_particlestate_angle():
    num_particles = 10

    state_vector1 = StateVector([[Bearing(np.pi + 0.1)], [-10.]])
    state_vector2 = StateVector([[Bearing(np.pi - 0.1)], [20.]])
    weight = Probability(1/num_particles)
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    # Test state without timestamp
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, StateVector([[np.pi], [5.]]))
    assert np.allclose(state.covar, CovarianceMatrix([[0.01, -1.5], [-1.5, 225]]))


def test_state_mutable_sequence_state():
    state_vector = StateVector([[0]])
    timestamp = datetime.datetime(2018, 1, 1, 14)
    delta = datetime.timedelta(minutes=1)
    sequence = StateMutableSequence(
        [State(state_vector, timestamp=timestamp+delta*n)
         for n in range(10)])

    assert sequence.state is sequence.states[-1]
    assert np.array_equal(sequence.state_vector, state_vector)
    assert sequence.timestamp == timestamp + delta*9

    del sequence[-1]
    assert sequence.timestamp == timestamp + delta*8


def test_state_mutable_sequence_slice():
    state_vector = StateVector([[0]])
    timestamp = datetime.datetime(2018, 1, 1, 14)
    delta = datetime.timedelta(minutes=1)
    sequence = StateMutableSequence(
        [State(state_vector, timestamp=timestamp+delta*n)
         for n in range(10)])

    assert isinstance(sequence[timestamp:], StateMutableSequence)
    assert isinstance(sequence[5:], StateMutableSequence)
    assert isinstance(sequence[timestamp], State)
    assert isinstance(sequence[5], State)

    assert len(sequence[timestamp:]) == 10
    assert len(sequence[:timestamp]) == 0
    assert len(sequence[timestamp+delta*5:]) == 5
    assert len(sequence[:timestamp+delta*5]) == 5
    assert len(sequence[timestamp+delta*4:timestamp+delta*6]) == 2
    assert len(sequence[timestamp+delta*2:timestamp+delta*8:3]) == 2
    assert len(sequence[timestamp+delta*1:][:timestamp+delta*2]) == 1

    assert sequence[timestamp] == sequence.states[0]

    end_timestamp = sequence.timestamp
    assert sequence[end_timestamp] == sequence.states[-1]
    assert sequence[end_timestamp].state_vector == StateVector([[0]])

    # Add state at same time
    sequence.append(State(state_vector + 1, timestamp=end_timestamp))
    assert sequence[end_timestamp]
    assert sequence[end_timestamp].state_vector == StateVector([[1]])

    assert len(sequence) == 11
    assert len(list(sequence.last_timestamp_generator())) == 10

    with pytest.raises(TypeError):
        sequence[timestamp:1]

    with pytest.raises(IndexError):
        sequence[timestamp-delta]


def test_state_mutable_sequence_sequence_init():
    """Test initialising with an existing sequence"""
    state_vector = StateVector([[0]])
    timestamp = datetime.datetime(2018, 1, 1, 14)
    delta = datetime.timedelta(minutes=1)
    sequence = StateMutableSequence(
        StateMutableSequence([State(state_vector, timestamp=timestamp + delta * n)
                              for n in range(10)]))

    assert not isinstance(sequence.states, list)

    assert sequence.state is sequence.states[-1]
    assert np.array_equal(sequence.state_vector, state_vector)
    assert sequence.timestamp == timestamp + delta * 9

    del sequence[-1]
    assert sequence.timestamp == timestamp + delta * 8


def test_state_mutable_sequence_error_message():
    """Test that __getattr__ doesn't incorrectly identify the source of a missing attribute"""

    class TestSMS(StateMutableSequence):
        test_property: int = Property(default=3)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.test_variable = 5

        def test_method(self):
            pass

        @property
        def complicated_attribute(self):
            if self.test_property == 3:
                return self.test_property
            else:
                raise AttributeError('Custom error message')

    timestamp = datetime.datetime.now()
    test_obj = TestSMS(states=State(state_vector=StateVector([1, 2, 3]), timestamp=timestamp))

    # First check no errors on assigned vars
    test_obj.test_method()
    assert test_obj.test_property == 3
    test_obj.test_property = 6
    assert test_obj.test_property == 6
    assert test_obj.test_variable == 5

    # Now check that state variables are proxied correctly
    assert np.array_equal(test_obj.state_vector, StateVector([1, 2, 3]))
    assert test_obj.timestamp == timestamp

    # Now check that the right error messages are raised on missing attributes
    with pytest.raises(AttributeError, match="'TestSMS' object has no attribute 'missing_method'"):
        test_obj.missing_method()

    with pytest.raises(AttributeError, match="'TestSMS' object has no attribute "
                                             "'missing_variable'"):
        _ = test_obj.missing_variable

    # And check custom error messages are not swallowed
    # in the default case (test_property == 3), complicated_attribute works
    test_obj.test_property = 3
    assert test_obj.complicated_attribute == 3

    # when test_property != 3  it raises a custom error.
    test_obj.test_property = 5
    with pytest.raises(AttributeError, match="Custom error message"):
        _ = test_obj.complicated_attribute


def test_composite_state_timestamp_errors():
    with pytest.raises(AttributeError,
                       match="CompositeState must either have component states to define its "
                             "timestamp or a default timestamp"):
        CompositeState()
    with pytest.raises(AttributeError,
                       match="Component-states must share the same timestamp"):
        CompositeState([State([0], timestamp=1), State([0], timestamp=2)])
    with pytest.raises(AttributeError,
                       match="If a default timestamp is defined alongside component states, "
                             "these states must share the same timestamp as the default "
                             "timestamp"):
        CompositeState([State([0], timestamp=1)], default_timestamp=2)
    with pytest.raises(AttributeError,
                       match="If a default timestamp is defined alongside component states, "
                             "these states must share the same timestamp as the default "
                             "timestamp"):
        CompositeState([State([0], timestamp=1), State([0], timestamp=1)], default_timestamp=2)


def test_composite_state_timestamp():

    # Test timestamp
    for i in range(1, 4):
        assert CompositeState(i*[State([0], timestamp=1)]).timestamp == 1

    assert CompositeState(default_timestamp=1).timestamp == 1

    for i in range(4):
        assert CompositeState(i*[State([0], timestamp=1)], default_timestamp=1).timestamp == 1

    a = State([0, 1], timestamp=1)
    b = State([2], timestamp=1)
    c = State([3, 4], timestamp=1)
    state = CompositeState([a, b, c])

    # Test state vectors
    assert all((state.state_vectors[i] == state.inner_states[i].state_vector).all()
               for i in range(len(state.inner_states)))

    # Test state vector
    assert (state.state_vector == StateVector([0, 1, 2, 3, 4])).all()

    # Test get
    assert state[0] == a
    assert state[1] == b
    assert state[2] == c

    d = State([5, 6], timestamp=1)

    # Test set
    state[1] = d
    assert state[1] == d
    assert (state.state_vector == StateVector([0, 1, 5, 6, 3, 4])).all()

    # Test del
    del state[1]
    assert state[1] == c
    assert (state.state_vector == StateVector([0, 1, 3, 4])).all()

    states = [a, b, c]
    state = CompositeState(states)

    # Test iter
    for i, sub_state in enumerate(state):
        assert sub_state == states[i]

    # Test len
    assert len(state) == 3

    # Test insert
    state.insert(1, d)
    assert state.inner_states == [a, d, b, c]
    assert (state.state_vector == StateVector([0, 1, 5, 6, 2, 3, 4])).all()

    # Test append
    e = State([7], timestamp=1)
    state.append(e)
    assert state.inner_states == [a, d, b, c, e]
    assert (state.state_vector == StateVector([0, 1, 5, 6, 2, 3, 4, 7])).all()
