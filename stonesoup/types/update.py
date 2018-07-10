# -*- coding: utf-8 -*-
from .array import CovarianceMatrix
from .base import Type, Property
from .state import State, GaussianState, ParticleState
from ..types import Detection, Prediction


class Update(Type):
    """ Update type

    This is the base Update class. """

    prediction = Property(Prediction,
                          doc = 'Track prediction prior to updating')
    measurement = Property(Detection,
                           doc = 'Measurement used to update the track')

class StateUpdate(State, Update):
    """ StateUpdate type

    Most simple state Update type, which only has time and a state vector.
    """


class GaussianStateUpdate(Update, GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state Update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class ParticleStateUpdate(Update, ParticleState):
    """ParticleStateUpdate type

    This is a simple Particle state Update object.
    """
