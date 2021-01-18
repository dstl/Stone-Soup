# -*- coding: utf-8 -*-

from typing import MutableSequence
import numpy as np

from ..base import Property
from .array import StateVector, StateVectors
from .base import Type


class Particle(Type):
    """
    Particle type

    A particle type which contains a state and weight
    """
    state_vector: StateVector = Property(doc="State vector")
    weight: float = Property(doc='Weight of particle')
    parent: 'Particle' = Property(default=None, doc='Parent particle')

    def __init__(self, state_vector, weight, parent=None, *args, **kwargs):
        if parent:
            parent.parent = None
        if state_vector is not None and not isinstance(state_vector, StateVector):
            state_vector = StateVector(state_vector)
        super().__init__(state_vector, weight, parent, *args, **kwargs)

    @property
    def ndim(self):
        return self.state_vector.shape[0]


class Particles(Type):
    """
    Particle type

    A collection of particles. Contains a state and weight for each particle
    """
    state_vector: StateVectors = Property(default=None, doc="State vectors of particles")
    weight: MutableSequence[float] = Property(default=None, doc='Weights of particles')
    parent: 'Particles' = Property(default=None, doc='Parent particles')
    particle_list: MutableSequence[Particle] = Property(default=None,
                                                        doc='List of Particle objects')

    def __init__(self, state_vector=None, weight=None, parent=None, particle_list=None,
                 *args, **kwargs):
        if state_vector is not None:
            assert particle_list is None,\
                "Particles object cannot use both state_vector and particle_list"

        if particle_list and isinstance(particle_list, list):
            state_vector = StateVectors([particle.state_vector for particle in particle_list])
            weight = np.array([particle.weight for particle in particle_list])
            parent_list = [particle.parent for particle in particle_list
                           if particle.parent is not None]
            if parent_list:
                parent = Particles(particle_list=parent_list)

        if parent:
            parent.parent = None

        if state_vector is not None and not isinstance(state_vector, StateVectors):
            state_vector = StateVectors(state_vector)
        if weight is not None and isinstance(weight, np.ndarray):
            weight = np.array(weight)
        super().__init__(state_vector, weight, parent, particle_list, *args, **kwargs)

    def __getitem__(self, item):
        if self.parent:
            p = self.parent[item]
        else:
            p = None

        particle = Particle(state_vector=self.state_vector[:, item],
                            weight=self.weight[item],
                            parent=p)
        return particle

    def __len__(self):
        return self.state_vector.shape[1]

    @property
    def ndim(self):
        return self.state_vector.shape[0]
