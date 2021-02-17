# -*- coding: utf-8 -*-
import pytest
import numpy as np

from ..particle import Particle, Particles


def test_particle():
    particle1 = Particle(np.array([[0]]), weight=0.1)

    assert np.array_equal(particle1.state_vector, np.array([[0]]))
    assert particle1.weight == 0.1

    particle2 = Particle(np.array([[0]]), weight=0.1, parent=particle1)

    assert particle2.parent is particle1

    particle3 = Particle(np.array([[0]]), weight=0.1, parent=particle2)

    assert particle3.parent is particle2
    assert particle3.parent.parent is None


def test_particles():
    particles1 = Particles(np.array([[0, 0, 0]]), weight=[0.1, 0.1, 0.1])
    particle1 = Particle(np.array([[0]]), weight=0.1)

    assert np.array_equal(particles1.state_vector, np.array([[0, 0, 0]]))
    assert np.array_equal(particles1.weight, np.array([0.1, 0.1, 0.1]))
    assert np.array_equal(particles1[0].state_vector, particle1.state_vector)
    assert particles1[0].weight == particle1.weight
    assert particles1[0].parent == particle1.parent
    assert particles1.ndim == 1

    particles2 = Particles(np.array([[0, 0, 0]]), weight=[0.1, 0.1, 0.1], parent=particles1)

    assert particles2.parent is particles1

    particles3 = Particles(np.array([[0, 0, 0]]), weight=[0.1, 0.1, 0.1], parent=particles2)

    assert particles3.parent is particles2
    assert particles3.parent.parent is None

    particle_list = [Particle(np.array([[0]]), weight=0.1),
                     Particle(np.array([[0]]), weight=0.1),
                     Particle(np.array([[0]]), weight=0.1),
                     ]
    particles_from_list = Particles(particle_list=particle_list)

    assert np.array_equal(particles_from_list.state_vector, particles1.state_vector)
    assert np.array_equal(particles_from_list.weight, particles1.weight)
    assert particles_from_list.parent == particles1.parent

    with pytest.raises(ValueError):
        # Should never happen that only some particles have parents. Should give ValueError.
        particle_list_fail = [Particle(np.array([[0]]), weight=0.1, parent=None),
                              Particle(np.array([[0]]), weight=0.1, parent=particle1),
                              Particle(np.array([[0]]), weight=0.1, parent=None),
                              ]

        Particles(particle_list=particle_list_fail)

        # Cannot set Particles object with both a particle list and a state vector/weight
        Particles(np.array([[0, 0, 0]]), weight=[0.1, 0.1, 0.1], particle_list=particle_list)
