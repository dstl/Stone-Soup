# -*- coding: utf-8 -*-
from ..base import Property
from .base import Type
from .state import State, GaussianState, ParticleState
from ..types import MeasurementPrediction, Detection


class Update(Type):
    """ Update type

    The base update class. Updates are returned by Updater objects
     and contain the information that was used to perform the updating"""


class StateUpdate(State, Update):
    """ StateUpdate type

    Most simple state update type, where everything only has time
    and a state vector. Requires a prior state that was updated,
    the prediction of the measurement and the measurement
    used to update the prior.
    """

    prediction = Property(State,
                          doc="Prior state before updating")
    measurement_prediction = Property(MeasurementPrediction,
                                      doc="Prediction in measurement space")
    measurement = Property(Detection,
                           doc="Detection used to update state")


class GaussianStateUpdate(StateUpdate, GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class ParticleStateUpdate(StateUpdate, ParticleState):
    """ParticleStateUpdate type

    This is a simple Particle state update object.
    """
