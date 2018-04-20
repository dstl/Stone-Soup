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
