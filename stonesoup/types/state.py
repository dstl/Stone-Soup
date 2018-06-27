# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from .array import StateVector, CovarianceMatrix
from .base import Type
from .particle import Particle


class State(Type):
    """State type.

    Most simple state type, which only has time and a state vector."""
    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")
    state_vector = Property(StateVector, doc='State vector.')

    def __init__(self, state_vector, *args, **kwargs):
        state_vector = state_vector.view(StateVector)
        super().__init__(state_vector, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        if self.state_vector is not None:
            return self.state_vector.shape[0]


class GaussianState(State):
    """Gaussian State type

    This is a simple Gaussian state object, which, as the name suggests,
    is described by a Gaussian state distribution.
    """
    covar = Property(CovarianceMatrix, doc='Covariance matrix of state.')

    def __init__(self, state_vector, covar, *args, **kwargs):
        covar = covar.view(CovarianceMatrix)
        super().__init__(state_vector, covar, *args, **kwargs)
        if self.state_vector.shape[0] != self.covar.shape[0]:
            raise ValueError(
                "state vector and covar should have same dimensions")

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector


class ParticleState(Type):
    """Particle State type

    This is a particle state object which describes the state as a
    distribution of particles"""

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")
    particles = Property([Particle],
                         doc='List of particles representing state')

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return np.average([p.state_vector for p in self.particles], axis=0,
                          weights=[p.weight for p in self.particles])

    @property
    def state_vector(self):
        """The mean value of the particle states"""
        return self.mean

    @property
    def covar(self):
        cov = np.cov(np.hstack([p.state_vector for p in self.particles]),
                     ddof=0, aweights=[p.weight for p in self.particles])
        # Fix one dimensional covariances being returned with zero dimension
        if not cov.shape:
            cov = cov.reshape(1, 1)
        return cov
