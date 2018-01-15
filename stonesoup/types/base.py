# -*- coding: utf-8 -*-
import numpy as np

from ..base import Base, Property


class Type(Base):
    """Base type"""
    pass


class StateVector(Type):
    """State Vector type

    Parameters
    ==========
    state : numpy.ndarray
        state.
    covar : numpy.ndarray
        state covariance.
    """
    state = Property(np.ndarray)
    covar = Property(np.ndarray)

    def __init__(self, state, covar, *args, **kwargs):
        if not state.shape[1] == 1:
            raise ValueError(
                "state shape should be Nx1 dimensions: got {state.shape}")

        if not state.shape[0] == covar.shape[0] == covar.shape[1]:
            raise ValueError(
                "covar shape should compliment state dimension i.e."
                "{state.shape[0]}x{state.shape[0]}: got {covar.shape}")
        super().__init__(state, covar, *args, **kwargs)