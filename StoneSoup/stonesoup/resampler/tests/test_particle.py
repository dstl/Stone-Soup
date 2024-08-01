import numpy as np
import pytest

from ...types.particle import Particle
from ...types.state import ParticleState
from ..particle import SystematicResampler, MultinomialResampler, StratifiedResampler, \
    ResidualResampler, ResidualMethod
from ..particle import ESSResampler


def test_residual_method_errors():
    particles = [Particle(np.array([[i]]), weight=1.5/20 if i % 2 == 0 else 0.5/20)
                 for i in range(20)]
    resampler = ResidualResampler(residual_method='This_is_not_a_valid_residual_method')

    with pytest.raises(ValueError):
        resampler.resample(particles)


def test_systematic_equal():
    particles = [Particle(np.array([[i]]), weight=1/20) for i in range(20)]

    resampler = SystematicResampler()

    new_particles = resampler.resample(particles)

    # Particles equal weight, so should end up with similar particles.
    assert all(np.array_equal(np.array([[i]]), new_particle.state_vector)
               for i, new_particle in enumerate(new_particles))


def test_systematic_single():
    particles = [Particle(np.array([[i]]), weight=1 if i == 10 else 0)
                 for i in range(20)]

    resampler = SystematicResampler()

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    assert all(np.array_equal(np.array([[10]]), new_particle.state_vector)
               for new_particle in new_particles)


def test_systematic_even():
    particles = [Particle(np.array([[i]]), weight=1/10 if i % 2 == 0 else 0)
                 for i in range(20)]

    resampler = SystematicResampler()

    new_particles = resampler.resample(particles)

    # Weight all at even particles, so new should all be at even vector
    assert all(np.array_equal(np.array([[i//2*2]]), new_particle.state_vector)
               for i, new_particle in enumerate(new_particles))


def test_systematic_downsample():
    particles = [Particle(np.array([[i]]), weight=1/40) for i in range(40)]

    resampler = SystematicResampler()

    new_particles = resampler.resample(particles, 20)

    # Check that the resampler downsamples the particles correctly
    assert len(new_particles) == 20


def test_systematic_upsample():
    particles = [Particle(np.array([[i]]), weight=1/20) for i in range(20)]

    resampler = SystematicResampler()

    new_particles = resampler.resample(particles, 40)

    # Check that the resampler upsamples the particles correctly
    assert len(new_particles) == 40


def test_ess_zero():
    particles = [Particle(np.array([[i]]), weight=(i+1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)  # Needed for assert line

    resampler = ESSResampler(0)  # Should never resample

    new_particles = resampler.resample(particles)  # Should not have changed

    assert new_particles == particles


def test_ess_n():
    particles = [Particle(np.array([[i]]), weight=(i+1) / 55)
                 for i in range(10)]

    resampler = ESSResampler(10)  # This resampler should always resample

    new_particles = resampler.resample(particles)  # All weights should be equal due to resampling

    assert all([w == 1/10 for w in new_particles.weight])


def test_ess_inequality():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    resampler1 = ESSResampler()  # This resampler should not resample
    resampler2 = ESSResampler(3025/385 + 0.01)  # This resampler should resample

    new_particles1 = resampler1.resample(particles)
    new_particles2 = resampler2.resample(particles)

    assert new_particles1 == particles and all([w == 1 / 10 for w in new_particles2.weight])


def test_ess_empty_threshold():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    resampler = ESSResampler()
    resampler.resample(particles)
    assert resampler.threshold == 5


def test_ess_upsample():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    # This resampler should not resample
    resampler1 = ESSResampler(resampler=MultinomialResampler())
    # This resampler should resample
    resampler2 = ESSResampler(3025 / 385 + 0.01, resampler=MultinomialResampler())

    new_particles1 = resampler1.resample(particles, nparts=20)
    new_particles2 = resampler2.resample(particles, nparts=20)

    assert new_particles1 == particles and all([w == 1 / 20 for w in new_particles2.weight])
    assert len(new_particles2) == 20


def test_ess_downsample():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    # This resampler should not resample
    resampler1 = ESSResampler(resampler=MultinomialResampler())
    # This resampler should resample
    resampler2 = ESSResampler(3025 / 385 + 0.01, resampler=MultinomialResampler())

    new_particles1 = resampler1.resample(particles, nparts=5)
    new_particles2 = resampler2.resample(particles, nparts=5)

    assert new_particles1 == particles and all([w == 1 / 5 for w in new_particles2.weight])
    assert len(new_particles2) == 5


def test_ess_multinomial():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    # This resampler should not resample
    resampler1 = ESSResampler(resampler=MultinomialResampler())
    # This resampler should resample
    resampler2 = ESSResampler(3025/385 + 0.01, resampler=MultinomialResampler())

    new_particles1 = resampler1.resample(particles)
    new_particles2 = resampler2.resample(particles)

    assert new_particles1 == particles and all([w == 1 / 10 for w in new_particles2.weight])


def test_ess_stratified():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    # This resampler should not resample
    resampler1 = ESSResampler(resampler=StratifiedResampler())
    # This resampler should resample
    resampler2 = ESSResampler(3025/385 + 0.01, resampler=StratifiedResampler())

    new_particles1 = resampler1.resample(particles)
    new_particles2 = resampler2.resample(particles)

    assert new_particles1 == particles and all([w == 1 / 10 for w in new_particles2.weight])


def test_ess_residual():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    subresampler = ResidualResampler()

    # This resampler should not resample
    resampler1 = ESSResampler(resampler=subresampler)

    # This resampler should resample
    resampler2 = ESSResampler(3025/385 + 0.01, resampler=subresampler)

    new_particles1 = resampler1.resample(particles)
    new_particles2 = resampler2.resample(particles)

    assert new_particles1 == particles and all([w == 1 / 10 for w in new_particles2.weight])


def test_ess_residual_non_default():
    particles = [Particle(np.array([[i]]), weight=(i + 1) / 55)
                 for i in range(10)]
    particles = ParticleState(None, particle_list=particles)

    subresampler = ResidualResampler(residual_method=ResidualMethod.STRATIFIED)

    # This resampler should not resample
    resampler1 = ESSResampler(resampler=subresampler)

    # This resampler should resample
    resampler2 = ESSResampler(3025/385 + 0.01, resampler=subresampler)

    new_particles1 = resampler1.resample(particles)
    new_particles2 = resampler2.resample(particles)

    assert new_particles1 == particles and all([w == 1 / 10 for w in new_particles2.weight])


def test_multinomial_single():
    particles = [Particle(np.array([[i]]), weight=1 if i == 10 else 0)
                 for i in range(20)]

    resampler = MultinomialResampler()

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    assert all(np.array_equal(np.array([[10]]), new_particle.state_vector)
               for new_particle in new_particles)


def test_multinomial_even():
    particles = [Particle(np.array([[i]]), weight=1/10 if i % 2 == 0 else 0)
                 for i in range(20)]

    resampler = MultinomialResampler()

    new_particles = resampler.resample(particles)

    # Weight all at even particles, so new should all be at even vector
    assert all(new_particle.state_vector % 2 == 0 for new_particle in new_particles)


def test_multinomial_nparts():
    particles = [Particle(np.array([[i]]), weight=1 / 20) for i in range(20)]

    resampler = MultinomialResampler()

    new_particles = resampler.resample(particles)

    # Check correct number of new particles
    assert len(new_particles) == len(particles)


def test_multinomial_downsample():
    particles = [Particle(np.array([[i]]), weight=1 / 40) for i in range(40)]

    resampler = MultinomialResampler()

    new_particles = resampler.resample(particles, 20)

    # Check that the resampler downsamples the particles correctly
    assert len(new_particles) == 20


def test_multinomial_upsample():
    particles = [Particle(np.array([[i]]), weight=1 / 20) for i in range(20)]

    resampler = MultinomialResampler()

    new_particles = resampler.resample(particles, 40)

    # Check that the resampler upsamples the particles correctly
    assert len(new_particles) == 40


def test_stratified_equal():
    particles = [Particle(np.array([[i]]), weight=1 / 20) for i in range(20)]

    resampler = StratifiedResampler()

    new_particles = resampler.resample(particles)

    # Particles equal weight, so should end up with similar particles.
    assert all(np.array_equal(np.array([[i]]), new_particle.state_vector)
               for i, new_particle in enumerate(new_particles))


def test_stratified_single():
    particles = [Particle(np.array([[i]]), weight=1 if i == 10 else 0)
                 for i in range(20)]

    resampler = StratifiedResampler()

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    assert all(np.array_equal(np.array([[10]]), new_particle.state_vector)
               for new_particle in new_particles)


def test_stratified_even():
    particles = [Particle(np.array([[i]]), weight=1/10 if i % 2 == 0 else 0)
                 for i in range(20)]

    resampler = StratifiedResampler()

    new_particles = resampler.resample(particles)

    # Weight all at even particles, so new should all be at even vector
    assert all(new_particle.state_vector % 2 == 0 for new_particle in new_particles)


def test_stratified_nparts():
    particles = [Particle(np.array([[i]]), weight=1 / 20) for i in range(20)]

    resampler = StratifiedResampler()

    new_particles = resampler.resample(particles)

    # Check correct number of new particles
    assert len(new_particles) == len(particles)


def test_stratified_downsample():
    particles = [Particle(np.array([[i]]), weight=1 / 40) for i in range(40)]

    resampler = StratifiedResampler()

    new_particles = resampler.resample(particles, 20)

    # Check that the resampler downsamples the particles correctly
    assert len(new_particles) == 20


def test_stratified_upsample():
    particles = [Particle(np.array([[i]]), weight=1 / 20) for i in range(20)]

    resampler = StratifiedResampler()

    new_particles = resampler.resample(particles, 40)

    # Check that the resampler upsamples the particles correctly
    assert len(new_particles) == 40


def test_residual_single():
    particles = [Particle(np.array([[i]]), weight=1 if i == 10 else 0)
                 for i in range(20)]

    resampler = ResidualResampler()

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    assert all(np.array_equal(np.array([[10]]), new_particle.state_vector)
               for new_particle in new_particles)


def test_residual_equal():
    particles = [Particle(np.array([[i]]), weight=1 / 20 if i % 2 == 0 else 1 / 20) for i in
                 range(20)]

    resampler = ResidualResampler()

    new_particles = resampler.resample(particles)

    # All particles have weight = 1/N so all should be resampled once
    assert all(np.array_equal(np.array([[i]]), new_particle.state_vector)
               for i, new_particle in enumerate(new_particles))
    assert resampler.residual_method == ResidualMethod.MULTINOMIAL


def test_residual_multinomial_alternating():
    particles = [Particle(np.array([[i]]), weight=3 / 80 if i % 2 == 0 else 5 / 80) for i in
                 range(20)]

    resampler = ResidualResampler(residual_method=ResidualMethod.MULTINOMIAL)

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    odd = 0
    even = 0
    for particle in new_particles:
        if particle.state_vector[0] % 2 == 0:
            even += 1
        else:
            odd += 1

    # Odd indices should all automatically get resampled once each (50%), plus a chance that their
    # residual also gets resampled
    assert odd >= even
    assert resampler.residual_method == ResidualMethod.MULTINOMIAL


def test_residual_stratified_alternating():
    # Repeat alternating test with stratified method for residual resampling
    particles = [Particle(np.array([[i]]), weight=3 / 80 if i % 2 == 0 else 5 / 80) for i in
                 range(20)]

    resampler = ResidualResampler(residual_method=ResidualMethod.STRATIFIED)

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    odd = 0
    even = 0
    for particle in new_particles:
        if particle.state_vector[0] % 2 == 0:
            even += 1
        else:
            odd += 1

    # Odd indices should all automatically get resampled once each (50%), plus a chance that their
    # residual also gets resampled
    assert odd >= even
    assert resampler.residual_method == ResidualMethod.STRATIFIED


def test_residual_systematic_alternating():
    # Repeat alternating test with systematic method for residual resampling
    particles = [Particle(np.array([[i]]), weight=3 / 80 if i % 2 == 0 else 5 / 80) for i in
                 range(20)]

    resampler = ResidualResampler(residual_method=ResidualMethod.SYSTEMATIC)

    new_particles = resampler.resample(particles)

    # Weight all at single particle at index/vector 10, so all should be
    # at same point
    odd = 0
    even = 0
    for particle in new_particles:
        if particle.state_vector[0] % 2 == 0:
            even += 1
        else:
            odd += 1

    # Odd indices should all automatically get resampled once each (50%), plus a chance that their
    # residual also gets resampled
    assert odd >= even
    assert resampler.residual_method == ResidualMethod.SYSTEMATIC
