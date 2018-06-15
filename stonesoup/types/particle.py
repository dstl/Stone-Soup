# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from .state import State
from .base import Type


class Particle(State):
    """
    Particle type

    A particle type which contains a state and weight
    """

    weight = Property(float, doc='Weight of particle')
    parent = Property(None, default=None, doc='Parent particle')

    def __init__(self, state_vector, weight,
                 timestamp=None, parent=None, *args, **kwargs):
        if parent:
            parent.parent = None
        super().__init__(state_vector, weight, timestamp, parent, *args,
                         **kwargs)

class ParticleState(Type):
    """Particle State type

    This is a particle state object which describes the state as a
    distribution of particles"""

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")
    particles = Property([Particle], doc='List of particles representing state')

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return np.mean([p.state_vector for p in self.particles],axis=0)

    @property
    def state_vector(self):
        """The mean value of the particle states"""
        return self.mean

    @property
    def covar(self):
        cov = np.cov(np.hstack([p.state_vector for p in self.particles]))
        # Fix one dimensional covariances being returned as zero dimension arrays
        if not cov.shape:
            cov = cov.reshape(1,1)
        return cov