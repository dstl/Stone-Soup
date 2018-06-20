# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ..base import Property
from .base import Type


class StateVector(np.ndarray):
    """State vector wrapper for :class:`numpy.ndarray`

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *Nx1* vector. It's called same as to
    :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        if not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(
                "state vector shape should be Nx1 dimensions: got {}".format(
                    array.shape))
        return array.view(cls)

    def __array_wrap__(self, array):
        return np.asarray(array)


class CovarianceMatrix(np.ndarray):
    """Covariance matrix wrapper for :class:`numpy.ndarray`.

    This class returns a view to a :class:`numpy.ndarray`, but ensures that
    its initialised at a *NxN* matrix. It's called similar to
    :func:`numpy.asarray`.
    """

    def __new__(cls, *args, **kwargs):
        array = np.asarray(*args, **kwargs)
        if not array.ndim == 2:
            raise ValueError("Covariance should have ndim of 2: got {}"
                             "".format(array.ndim))
        return array.view(cls)

    def __array_wrap__(self, array):
        return np.asarray(array)


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
