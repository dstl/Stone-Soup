# -*- coding: utf-8 -*-
from .base import Type


class Track(Type):
    """Track type

    Attributes
    ----------
    estimates : list of State
        Estimated state of the track
    state : State
        Most recent estimate state
    state_vector : StateVector
        Most recent estimate state vector
    covar : CovarianceMatrix
        Most recent estimate state covariance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimates = []

    @property
    def state(self):
        return self.estimates[-1]

    @property
    def state_vector(self):
        return self.state.state_vector

    @property
    def covar(self):
        return self.state.covar
