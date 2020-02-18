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

    def __init__(self, state_vector, weight, dynamic_model, parent=None, *args, **kwargs):
        if parent:
            parent.parent = None
        super().__init__(state_vector, weight, dynamic_model, parent, *args, **kwargs)


Particle.parent.cls = Particle  # noqa:E305


class RaoBlackwellisedParticle(Type):
    """
    Particle type

    A particle type which contains a state and weight
    """
    state_vector = Property(StateVector, doc="State vector")
    weight = Property(float, doc='Weight of particle')
    parent = Property(None, default=None, doc='Parent particle')
    dynamic_model = Property(int, doc='Assigned dynamic model')
    model_probabilities = Property(list, doc="The dynamic probabilities of changing models")

    def __init__(self, state_vector, weight, dynamic_model, model_probabilities,
                 parent=None, *args, **kwargs):

        if parent:
            parent.parent = None
        super().__init__(state_vector, weight, dynamic_model, model_probabilities,
                         parent, *args, **kwargs)


RaoBlackwellisedParticle.parent.cls = Particle
