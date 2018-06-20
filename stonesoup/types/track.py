# -*- coding: utf-8 -*-
from .base import Type, Property
from ..types import State


class Track(Type):
    """Track type

    Parameters
    ----------

    Attributes
    ----------
    state : State
        Most recent state
    state_vector : StateVector
        Most recent state vector
    covar : CovarianceMatrix
        Most recent state covariance
    timestamp : :class:`datetime.datetime`
        Most recent state timestamp
    """

    states = Property(
        [State],
        default=None,
        doc="The initial states of the track. Default `None` which initialises"
            "with empty list.")

    def __init__(self, states=None, *args, **kwargs):
        if states is None:
            states = []
        super().__init__(states, *args, **kwargs)

    @property
    def state(self):
        return self.states[-1]

    @property
    def state_vector(self):
        return self.state.state_vector

    @property
    def covar(self):
        return self.state.covar

    @property
    def timestamp(self):
        return self.state.timestamp

    @property
    def mean(self):
        return self.state.mean
