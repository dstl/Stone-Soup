# -*- coding: utf-8 -*-

from ..base import Property
from .array import StateVector
from .base import Type


class TwoLineElement(Type):
    """
    TwoLineElement type

    A TwoLineElement type which contains an orbital state vector and meta data associated with object
    """
    
    state_vector = Property(StateVector, doc="Orbital state vector")
    weight = Property(float, doc='Weight of particle')
    parent = Property(None, default=None, doc='Parent particle')

    def __init__(self, state_vector, weight, parent=None, *args, **kwargs):
        if parent:
            parent.parent = None
        super().__init__(state_vector, weight, parent, *args, **kwargs)
Particle.parent.cls = Particle  # noqa:E305
