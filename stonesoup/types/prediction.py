# -*- coding: utf-8 -*-
from ..base import Property
from .array import CovarianceMatrix
from .base import Type
from .state import State, GaussianState, ParticleState, SqrtGaussianState


class Prediction(Type):
    """ Prediction type

    This is the base prediction class. """


class MeasurementPrediction(Type):
    """ Prediction type

    This is the base measurement prediction class. """


class StatePrediction(State, Prediction):
    """ StatePrediction type

    Most simple state prediction type, which only has time and a state vector.
    """


class StateMeasurementPrediction(State, MeasurementPrediction):
    """ MeasurementPrediction type

    Most simple measurement prediction type, which only has time and a state
    vector.
    """


class GaussianStatePrediction(Prediction, GaussianState):
    """ GaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class SqrtGaussianStatePrediction(Prediction, SqrtGaussianState):
    """ SqrtGaussianStatePrediction type

    This is a Gaussian state prediction object, with the covariance held
    as the square root of the covariance matrix
    """


class GaussianMeasurementPrediction(MeasurementPrediction, GaussianState):
    """ GaussianMeasurementPrediction type

    This is a simple Gaussian measurement prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """

    cross_covar = Property(CovarianceMatrix,
                           doc="The state-measurement cross covariance matrix",
                           default=None)

    def __init__(self, state_vector, covar, timestamp=None,
                 cross_covar=None, *args, **kwargs):
        if(cross_covar is not None
           and cross_covar.shape[1] != state_vector.shape[0]):
            raise ValueError("cross_covar should have the same number of \
                             columns as the number of rows in state_vector")
        super().__init__(state_vector, covar, timestamp,
                         cross_covar, *args, **kwargs)


class ParticleStatePrediction(Prediction, ParticleState):
    """ParticleStatePrediction type

    This is a simple Particle state prediction object.
    """


class ParticleMeasurementPrediction(MeasurementPrediction, ParticleState):
    """MeasurementStatePrediction type

    This is a simple Particle measurement prediction object.
    """
