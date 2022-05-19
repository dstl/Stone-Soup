# -*- coding: utf-8 -*-
import numpy as np

from ..particle import Particle


def test_particle():
    particle1 = Particle(np.array([[0]]), weight=0.1)

    assert np.array_equal(particle1.state_vector, np.array([[0]]))
    assert particle1.weight == 0.1

    particle2 = Particle(np.array([[0]]), weight=0.1, parent=particle1)

    assert particle2.parent is particle1

    particle3 = Particle(np.array([[0]]), weight=0.1, parent=particle2)

    assert particle3.parent is particle2
    assert particle3.parent.parent is None
