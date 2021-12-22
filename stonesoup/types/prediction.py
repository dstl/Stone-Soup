# -*- coding: utf-8 -*-

from stonesoup.types.state import CreatableFromState
from ..base import Property
from .array import CovarianceMatrix
from .base import Type
from .state import (State, GaussianState, ParticleState, SqrtGaussianState, InformationState,
                    TaggedWeightedGaussianState, WeightedGaussianState)
from ..models.transition.base import TransitionModel


class Prediction(Type, CreatableFromState):
    """ Prediction type

    This is the base prediction class. """
    transition_model: TransitionModel = Property(
        default=None, doc='The transition model used to make the prediction')


class MeasurementPrediction(Type, CreatableFromState):
    """ Prediction type

    This is the base measurement prediction class. """


class StatePrediction(Prediction, State):
    """ StatePrediction type

    Most simple state prediction type, which only has time and a state vector.
    """


class InformationStatePrediction(Prediction, InformationState):
    """ InformationStatePrediction type

    Information state prediction type: contains state vector, precision matrix and timestamp
    """


class StateMeasurementPrediction(MeasurementPrediction, State):
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


class WeightedGaussianStatePrediction(Prediction, WeightedGaussianState):
    """ WeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution
    with an associated weight.
    """


class TaggedWeightedGaussianStatePrediction(Prediction,
                                            TaggedWeightedGaussianState):
    """ TaggedWeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution, with an associated
    weight and unique tag.
    """


class GaussianMeasurementPrediction(MeasurementPrediction, GaussianState):
    """ GaussianMeasurementPrediction type

    This is a simple Gaussian measurement prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """

    cross_covar: CovarianceMatrix = Property(
        default=None, doc="The state-measurement cross covariance matrix")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cross_covar is not None \
                and self.cross_covar.shape[1] != self.state_vector.shape[0]:
            raise ValueError("cross_covar should have the same number of "
                             "columns as the number of rows in state_vector")


# Don't need to support Sqrt Covar for MeasurementPrediction
CreatableFromState.class_mapping[MeasurementPrediction][SqrtGaussianState] = \
    GaussianMeasurementPrediction


class ParticleStatePrediction(Prediction, ParticleState):
    """ParticleStatePrediction type

    This is a simple Particle state prediction object.
    """


class ParticleMeasurementPrediction(MeasurementPrediction, ParticleState):
    """MeasurementStatePrediction type

    This is a simple Particle measurement prediction object.
    """
