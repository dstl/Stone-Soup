# -*- coding: utf-8 -*-

# imports
import numpy as np
from ..types import GaussianState
from ..types import State

class singleStateMahalanobis(GaussianState, State):
    # A method which returns the Mahalanobis distance of the true state from the inferred state
    # Find a way of doing this that doesn't use an inv function
    @property
    def calculateMahalanobis(self):
        sdiff = np.subtract(State.state_vector, GaussianState.mean)
        return np.sqrt(sdiff.T @ np.linalg.inv(GaussianState.covar) @ sdiff)
