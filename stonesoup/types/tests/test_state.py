# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ..particle import Particle
from ..state import State, GaussianState, ParticleState


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
        State(np.array([1, 2, 3, 4]))


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


def test_particlestate():
    with pytest.raises(TypeError):
        ParticleState()

    # 1D
    num_particles = 10
    state_vector1 = np.array([[0]])
    state_vector2 = np.array([[100]])
    weight = 1/num_particles
    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight) for _ in range(num_particles//2))

    # Test state without timestamp
    state = ParticleState(particles)
    assert np.array_equal(state.state_vector, np.array([[50]]))
    assert np.array_equal(state.covar, np.array([[2500]]))

    # Test state with timestamp
    timestamp = datetime.datetime.now()
    state = ParticleState(particles, timestamp=timestamp)
    assert np.array_equal(state.state_vector, np.array([[50]]))
    assert np.array_equal(state.covar, np.array([[2500]]))
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
    assert np.array_equal(state.state_vector, np.array([[50], [100]]))
    assert np.array_equal(state.covar, np.array([[2500, 5000], [5000, 10000]]))


def test_particlestate_weighted():
    num_particles = 10

    # Half particles at high weight at 0
    state_vector1 = np.array([[0]])
    weight1 = 0.75 / (num_particles / 2)

    # Other half of particles low weight at 100
    state_vector2 = np.array([[100]])
    weight2 = 0.25 / (num_particles / 2)

    particles = []
    particles.extend(Particle(
        state_vector1, weight=weight1) for _ in range(num_particles//2))
    particles.extend(Particle(
        state_vector2, weight=weight2) for _ in range(num_particles//2))

    # Check particles sum to 1 still
    assert sum(particle.weight for particle in particles) == pytest.approx(1)

    # Test state vector is now weighted towards 0 from 50 (non-weighted mean)
    state = ParticleState(particles)
    assert np.array_equal(state.state_vector, np.array([[25]]))
    assert np.array_equal(state.covar, np.array([[1875]]))
