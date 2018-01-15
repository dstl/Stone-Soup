# -*- coding: utf-8 -*-
from .base import Type


class Track(Type):
    """Track type

    Parameters
    ==========
    detections : list of Detection
        Detection state vectors associated with track
    estimates : list of StateVector
        Estimated state vectors associated with track
    state_vector : StateVector
        Most recent estimate state vector
    state : numpy.ndarray
        Most recent estimate state
    covar : numpy.ndarray
        Most recent estimate state covariance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detections = []
        self.estimates = []

    @property
    def state_vector(self):
        return self.estimates[-1]

    @property
    def state(self):
        return self.state_vector.state

    @property
    def covar(self):
        return self.state_vector.covar
