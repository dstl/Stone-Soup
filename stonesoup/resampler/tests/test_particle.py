# -*- coding: utf-8 -*-
import numpy as np

from ...types.particle import Particle
from ..particle import SystematicResampler


def test_systematic_equal():
    particles = [Particle(np.array([[i]]), weight=1/20) for i in range(20)]

    resamlper = SystematicResampler()

    new_particles = resamlper.resample(particles)

    # Particles equal weight, so should end up with similar particles.
    assert all(np.array_equal(np.array([[i]]), new_particle.state_vector)
               for i, new_particle in enumerate(new_particles))


def test_systematic_single():
    particles = [Particle(np.array([[i]]), weight=1 if i == 10 else 0)
                 for i in range(20)]

    resamlper = SystematicResampler()

    new_particles = resamlper.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    assert all(np.array_equal(np.array([[10]]), new_particle.state_vector)
               for new_particle in new_particles)


def test_systematic_even():
    particles = [Particle(np.array([[i]]), weight=1/10 if i % 2 == 0 else 0)
                 for i in range(20)]

    resamlper = SystematicResampler()

    new_particles = resamlper.resample(particles)

    # Weight all at even particles, so new should all be at even vector
    assert all(np.array_equal(np.array([[i//2*2]]), new_particle.state_vector)
               for i, new_particle in enumerate(new_particles))
