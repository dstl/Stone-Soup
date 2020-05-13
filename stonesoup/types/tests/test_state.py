# -*- coding: utf-8 -*-
import datetime

import numpy as np
import scipy.linalg
import pytest

from ..numeric import Probability
from ..particle import Particle
from ..state import State, GaussianState, ParticleState, \
    StateMutableSequence, WeightedGaussianState, SqrtGaussianState


def test_state():
    with pytest.raises(TypeError):
        State()

    # Test state initiation without timestamp
    state_vector = np.array([[0], [1]])
    state = State(state_vector)
    assert np.array_equal(state.state_vector, state_vector)

    # Test state initiation with timestamp
    timestamp = datetime.datetime.now()
    state = State(state_vector, timestamp=timestamp)
    assert state.timestamp == timestamp


def test_state_invalid_vector():
    with pytest.raises(ValueError):
        State(np.array([[[1, 2, 3, 4]]]))


def test_gaussianstate():
    """ GaussianState Type test """

    with pytest.raises(TypeError):
        GaussianState()

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
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
    mean = np.array([[1], [2], [3], [4]])  # 4D
    covar = np.diag([1, 2, 3])  # 3D
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
    state = SqrtGaussianState(mean, covar, timestamp=timestamp, sqrt_form=False)
    assert (np.array_equal(state.covar, lower_covar))

    # Test that a general square root matrix is also a solution
    general_covar = scipy.linalg.sqrtm(covar)
    another_state = SqrtGaussianState(mean, general_covar, timestamp=timestamp)
    assert (np.array_equal(another_state.covar, general_covar))
    assert(not np.allclose(another_state.covar, lower_covar, 0, atol=1e-10))
    assert(np.allclose(state.covar@state.covar.T, lower_covar@lower_covar.T, 0, atol=1e-10))


def test_weighted_gaussian_state():
    mean = np.array([[1], [2], [3], [4]])  # 4D
    covar = np.diag([1, 2, 3])  # 3D
    weight = 0.3
    with pytest.raises(ValueError):
        a = WeightedGaussianState(mean, covar, weight)
        assert a.weight == weight


def test_particlestate():
    with pytest.raises(TypeError):
        ParticleState()

    # 1D
    num_particles = 10
    state_vector1 = np.array([[0]])
    state_vector2 = np.array([[100]])
    weight = Probability(1/num_particles)
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    # Test state without timestamp
    state = ParticleState(particles)
    assert np.allclose(state.state_vector, np.array([[50]]))
    assert np.allclose(state.covar, np.array([[2500]]))

    # Test state with timestamp
    timestamp = datetime.datetime.now()
    state = ParticleState(particles, timestamp=timestamp)
    assert np.allclose(state.state_vector, np.array([[50]]))
    assert np.allclose(state.covar, np.array([[2500]]))
    assert state.timestamp == timestamp

    # 2D
    state_vector1 = np.array([[0], [0]])
    state_vector2 = np.array([[100], [200]])
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    state = ParticleState(particles)
    assert np.allclose(state.state_vector, np.array([[50], [100]]))
    assert np.allclose(state.covar, np.array([[2500, 5000], [5000, 10000]]))


def test_particlestate_weighted():
    num_particles = 10

    # Half particles at high weight at 0
    state_vector1 = np.array([[0]])
    weight1 = Probability(0.75 / (num_particles / 2))

    # Other half of particles low weight at 100
    state_vector2 = np.array([[100]])
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
    assert np.allclose(state.state_vector, np.array([[25]]))
    assert np.allclose(state.covar, np.array([[1875]]))


def test_state_mutable_sequence_state():
    state_vector = np.array([[0]])
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
    state_vector = np.array([[0]])
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

    with pytest.raises(TypeError):
        sequence[timestamp:1]

    with pytest.raises(IndexError):
        sequence[timestamp-delta]
