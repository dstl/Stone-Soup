# -*- coding: utf-8 -*-
import datetime
from collections import MutableSequence

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
        state_vector = StateVector(state_vector)
        super().__init__(state_vector, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]


class StateMutableSequence(Type, MutableSequence):
    """A mutable sequence for :class:`~.State` instances

    This sequence acts like a regular list object for States, as well as
    proxying state attributes to the last state in the sequence. This sequence
    can also be indexed/sliced by :class:`datetime.datetime` instances.

    Example
    -------
    >>> t0 = datetime.datetime(2018, 1, 1, 14, 00)
    >>> t1 = t0 + datetime.timedelta(minutes=1)
    >>> state0 = State([[0]], t0)
    >>> sequence = StateMutableSequence([state0])
    >>> print(sequence.state_vector, sequence.timestamp)
    [[0]] 2018-01-01 14:00:00
    >>> sequence.append(State([[1]], t1))
    >>> for state in sequence[t1:]:
    ...     print(state.state_vector, state.timestamp)
    [[1]] 2018-01-01 14:01:00
    """

    states = Property(
        [State],
        default=None,
        doc="The initial list of states. Default `None` which initialises"
            "with empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        super().__init__(states, *args, **kwargs)

    def __len__(self):
        return self.states.__len__()

    def __setitem__(self, index, value):
        return self.states.__setitem__(index, value)

    def __delitem__(self, index):
        return self.states.__delitem__(index)

    def __getitem__(self, index):
        if isinstance(index, slice) and (
                isinstance(index.start, datetime.datetime)
                or isinstance(index.stop, datetime.datetime)):
            items = []
            for state in self.states:
                try:
                    if index.start and state.timestamp < index.start:
                        continue
                    if index.stop and state.timestamp >= index.stop:
                        continue
                except TypeError as exc:
                    raise TypeError(
                        'both indices must be `datetime.datetime` objects for'
                        'time slice') from exc
                items.append(state)
            return StateMutableSequence(items[::index.step])
        elif isinstance(index, datetime.datetime):
            for state in self.states:
                if state.timestamp == index:
                    return state
            else:
                raise IndexError('timestamp not found in states')
        elif isinstance(index, slice):
            return StateMutableSequence(self.states.__getitem__(index))
        else:
            return self.states.__getitem__(index)

    def __getattr__(self, item):
        if item.startswith("_"):
            # Don't proxy special/private attributes to `state`
            raise AttributeError(
                "{!r} object has no attribute {!r}".format(
                    type(self).__name__, item))
        else:
            return getattr(self.state, item)

    def insert(self, index, value):
        return self.states.insert(index, value)

    @property
    def state(self):
        return self.states[-1]


class GaussianState(State):
    """Gaussian State type

    This is a simple Gaussian state object, which, as the name suggests,
    is described by a Gaussian state distribution.
    """
    covar = Property(CovarianceMatrix, doc='Covariance matrix of state.')

    def __init__(self, state_vector, covar, *args, **kwargs):
        covar = CovarianceMatrix(covar)
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
        result = np.average([p.state_vector for p in self.particles], axis=0,
                            weights=[p.weight for p in self.particles])
        # Convert type as may have type of weights
        return result.astype(np.float, copy=False)

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
