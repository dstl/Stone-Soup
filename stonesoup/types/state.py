# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from .base import Type


class StateVector(Type, np.ndarray):
    """State vector wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *Nx1* vector.
    """
    value = Property(np.ndarray, doc='Array')

    def __new__(cls, value, *args, **kwargs):
        array = np.array(value)
        if not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(
                "state vector shape should be Nx1 dimensions: got {}".format(
                    array.shape))
        return array.view(cls)


class CovarianceMatrix(Type, np.ndarray):
    """Covariance matrix wrapper for :class:`numpy.ndarray`.

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *NxN* matrix.
    """
    value = Property(np.ndarray, doc='Array')

    def __new__(cls, value, *args, **kwargs):
        array = np.array(value)
        if not (array.ndim == 2 and array.shape[0] == array.shape[1]):
            raise ValueError("Covariance should be square NxN matrix: got {}"
                             "".format(array.shape))
        return array.view(cls)


class State(Type):
    """State type.

    Most simple state type, which only has time and a state vector."""
    timestamp = Property(datetime.datetime, doc="Timestamp of the state.")
    state_vector = Property(StateVector, doc='State vector.')

    def __init__(self, timestamp, state_vector, *args, **kwargs):
        state_vector.view(StateVector)
        super().__init__(timestamp, state_vector, *args, **kwargs)

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

    def __init__(self, timestamp, state_vector, covar, *args, **kwargs):
        covar.view(CovarianceMatrix)
        super().__init__(timestamp, state_vector, covar, *args, **kwargs)
        if self.state_vector.shape[0] != self.covar.shape[0]:
            raise ValueError(
                "state vector and covar should have same dimensions")

    @property
    def mean(self):
        """The state mean, equivalent to state vector"""
        return self.state_vector
