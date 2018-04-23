# -*- coding: utf-8 -*-
import numpy as np

from ..base import Base, Property


class Type(Base):
    """Base type"""


class Probability(Type, float):
    """Probability type.

    Same as float, but value must be between 0 and 1."""
    value = Property(float)

    def __new__(cls, value, *args, **kwargs):
        if not 0 <= value <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return super().__new__(cls, value, *args, **kwargs)


class StateVector(Type):
    """State Vector type"""
    state = Property(np.ndarray, doc="state")
    covar = Property(np.ndarray, doc="state covariance")

    def __init__(self, state, covar, *args, **kwargs):
        if not state.shape[1] == 1:
            raise ValueError(
                "state shape should be Nx1 dimensions: got {}".format(
                    state.shape))

        if not state.shape[0] == covar.shape[0] == covar.shape[1]:
            raise ValueError(
                "covar shape should compliment state dimension i.e. "
                "{0}x{0}: got {1}".format(state.shape[0], covar.shape))
        super().__init__(state, covar, *args, **kwargs)


class GaussianState(Type):
    """Gaussian State type

    This is a simple Gaussian state object, which, as the name suggests,
    is described by a Gaussian state distribution.

    Parameters
    ==========
    ndim : int
        The number of state dimensions
    mean : numpy.ndarray
        The state mean
    covar : numpy.ndarray
        The state covariance.
    """

    mean = Property(np.ndarray, doc="state mean")
    covar = Property(np.ndarray, doc="state covariance")

    def __init__(self, mean=None, covar=None, *args, **kwargs):
        if mean is not None:
            if not mean.shape[1] == 1:
                raise ValueError(
                    "state shape should be Nx1 dimensions: got {}".format(
                        mean.shape))
        if covar is not None:
            if not mean.shape[0] == covar.shape[0] == covar.shape[1]:
                raise ValueError(
                    "covar shape should compliment state dimension i.e. "
                    "{0}x{0}: got {1}".format(mean.shape[0], covar.shape))

        super().__init__(mean, covar, *args, **kwargs)

    @property
    def ndim(self):
        if(self.mean is not None):
            return self.mean.shape[0]
        elif(self.covar is not None):
            return self.covar.shape[0]
        else:
            return None
