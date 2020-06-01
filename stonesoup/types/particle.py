# -*- coding: utf-8 -*-

from ..base import Property
from .array import StateVector
from .base import Type


class Particle(Type):
    """
    Particle type

    A particle type which contains a state and weight
    """
    state_vector = Property(StateVector, doc="State vector")
    weight = Property(float, doc='Weight of particle')
    parent = Property(None, default=None, doc='Parent particle')
    dynamic_model = Property(int, doc='Assigned dynamic model')
    prior_state = Property(list, doc='The prior estimate containing all state estimates')

    def __init__(self, state_vector, weight, dynamic_model, prior_state, parent=None, *args, **kwargs):
        if parent:
            parent.parent = None
        if state_vector is not None and not isinstance(state_vector, StateVector):
            state_vector = StateVector(state_vector)
        super().__init__(state_vector, weight, parent, *args, **kwargs)

    @property
    def ndim(self):
        return self.state_vector.shape[0]
Particle.parent.cls = Particle  # noqa:E305
