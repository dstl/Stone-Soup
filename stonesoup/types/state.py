# -*- coding: utf-8 -*-
import collections
import datetime
from collections.abc import MutableSequence
from typing import List, OrderedDict

import numpy as np

from .array import StateVector, CovarianceMatrix
from .base import Type
from .particle import Particle
from ..base import Property


class State(Type):
    """State type.

    Most simple state type, which only has time and a state vector."""
    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")
    state_vector = Property(StateVector, doc='State vector.')

    def __init__(self, state_vector, *args, **kwargs):
        # Don't cast away subtype of state_vector if not necessary
        if state_vector is not None \
                and not isinstance(state_vector, StateVector):
            state_vector = StateVector(state_vector)
        super().__init__(state_vector, *args, **kwargs)

    @property
    def ndim(self):
        """The number of dimensions represented by the state."""
        return self.state_vector.shape[0]


class ASDState(Type):
    """ASD State type

    For the use of Accumulated State Densities.
    """

    multi_state_vector = Property(StateVector,
                                  doc="State vector of all timestamps")
    timestamps = Property(List[datetime.datetime],
                          doc="List fo all timestamps which have a state in"
                          + " the ASDState")
    max_nstep = Property(int,
                         doc="Decides when the state is pruned in a prediction"
                         + "step. If 0 then there is no pruning ")

    def __init__(self, multi_state_vector, timestamps,
                 max_nstep=0, *args, **kwargs):
        if multi_state_vector is not None and timestamps is not None:
            multi_state_vector = StateVector(multi_state_vector)
            if not isinstance(timestamps,list):
                timestamps = list([timestamps])
            else:
                timestamps = timestamps
            self.max_nstep = max_nstep
        super().__init__(multi_state_vector, timestamps,
                         max_nstep, *args, **kwargs)

    @property
    def state_vector(self):
        """The State vector of the newest timestamp"""
        return self.multi_state_vector[0:self.ndim]

    @state_vector.setter
    def state_vector(self, value):
        """set the state vector. In this case the multi_state_vector
        is overwritten.
        """
        self.multi_state_vector = StateVector(value)

    @property
    def timestamp(self):
        """The newest timestamp"""
        return self.timestamps[0]

    @timestamp.setter
    def timestamp(self, value):
        """insert a new timestamp in the list of timestamps if it is not in
        there already
        """
        if value not in self.timestamps:
            self.timestamps.insert(0, value)

    @property
    def ndim(self):
        """Dimension of one State"""
        return int(self.multi_state_vector.shape[0] / len(self.timestamps))

    @property
    def nstep(self):
        """Number of timesteps which are in the ASDState"""
        return len(self.timestamps)

    @property
    def state_list(self):
        """Generates a list of all States in the ASD State one for each
            timestep
        """
        ndim = self.ndim
        vectors = [StateVector(self.multi_state_vector[i:i + ndim]) for i in
                   range(0, self.multi_state_vector.shape[0] - ndim, ndim)]
        states = [State(state_vector=vector, timestamp=timestamp)
                  for vector, timestamp in zip(vectors, self.timestamps)]
        return states


State.register(ASDState)


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
        elif not isinstance(states, list):
            # Ensure states is a list
            states = [states]
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


class ASDGaussianState(ASDState):
    """ASDGaussian State type

    This is a simple Accumulated State Density Gaussian state object, which as
    the name suggests is described by a Gaussian state distribution.
    """
    correlation_matrices = Property(collections.OrderedDict,
                                    default=collections.OrderedDict({}), doc=
                                    "Dict of Correlation Matrices, consisting "
                                    + "of P_{l|l}, P_{l|l+1} and F_{l+1|l} "
                                    + "built in the Kalman predictor "
                                    + "and Kalman updater")

    multi_covar = Property(CovarianceMatrix, doc="Covariance of all timesteps")

    @property
    def covar(self):
        return self.multi_covar[0:self.ndim, 0:self.ndim]

    @covar.setter
    def covar(self, value):
        self.mutli_covar = CovarianceMatrix(value)

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector[0:self.ndim]

    @property
    def state_list(self):
        ndim = self.ndim
        states = super().state_list
        covars = [CovarianceMatrix(self.multi_covar[i:i + ndim, i:i + ndim])
                  for i in range(0, self.multi_covar.shape[0] - ndim, ndim)]
        states = [GaussianState(state_vector=state.state_vector,
                  timestamp=state.timestamp, covar=matrix) for
                  state, matrix in zip(states, covars)]
        return states


class WeightedGaussianState(GaussianState):
    """Weighted Gaussian State Type

    Gaussian State object with an associated weight.  Used as components
    for a GaussianMixtureState.
    """
    weight = Property(float, default=0, doc="Weight of the Gaussian State.")


class ASDWeightedGaussianState(ASDGaussianState):
    """ASD Weighted Gaussian State Type

    ASD Gaussian State object with an associated weight.  Used as components
    for a GaussianMixtureState.
    """
    weight = Property(float, default=0, doc="Weight of the Gaussian State.")

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
