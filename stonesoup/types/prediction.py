# -*- coding: utf-8 -*-
from ..base import Property
from .array import CovarianceMatrix
from .base import Type
from .state import (State, GaussianState, ParticleState, WeightedGaussianState,
                    GaussianMixtureState)


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


class WeightedGaussianStatePrediction(Prediction, WeightedGaussianState):
    """ WeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class GaussianMixtureStatePrediction(Prediction, GaussianMixtureState):
    """ GaussianMixtureStatePrediction type

    This is a simple Gaussian Mixture state prediction object, which, as the
    name suggests, is described by a Gaussian distribution.
    """


class WeightedGaussianMeasurementPrediction(MeasurementPrediction,
                                            WeightedGaussianState):
    """ WeightedGaussianMeasurementPrediction type

    An augmented GaussianMeasurementPrediction type that also incorporates a
    weight.
    """
    cross_covar = Property(CovarianceMatrix,
                           doc="The state-measurement cross covariance matrix",
                           default=None)


class GaussianMixtureMeasurementPrediction(MeasurementPrediction,
                                           GaussianMixtureState):
    """GaussianMixtureMeasurementPrediction type

    A Gaussian Mixture measurement prediction that "quacks" both like a
    GaussianMixtureState, as well as a GaussianMeasurementPrediction
    """
    cross_covar = Property(CovarianceMatrix,
                           doc="The state-measurement cross covariance matrix",
                           default=None)