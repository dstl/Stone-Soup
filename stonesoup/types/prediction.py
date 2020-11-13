# -*- coding: utf-8 -*-
from ..base import Property
from .array import CovarianceMatrix
from .base import Type
from .state import (State, GaussianState, ParticleState, SqrtGaussianState,
                    TaggedWeightedGaussianState)


def _from_state(cls, state, prediction_type=None, **kwargs):
    try:
        state_type = next(type_ for type_ in type(state).mro() if type_ in cls.class_mapping)
    except StopIteration:
        raise TypeError(f'{cls.__name__} type not defined for {type(state).__name__}')
    if prediction_type is None:
        prediction_type = cls.class_mapping[state_type]

    # Use current state kwargs that also properties of prediction type
    new_kwargs = {
        name: getattr(state, name)
        for name in state_type.properties.keys() & prediction_type.properties.keys()}
    # And replace them with any newly defined kwargs
    new_kwargs.update(kwargs)

    # Special case for SqrtGaussian
    if cls is Prediction and issubclass(state_type, SqrtGaussianState):
        new_kwargs['sqrt_covar'] = new_kwargs.pop('covar')

    return prediction_type(**new_kwargs)


class Prediction(Type):
    """ Prediction type

    This is the base prediction class. """
    class_mapping = {}
    from_state = classmethod(_from_state)

    def __init_subclass__(cls, **kwargs):
        state_type = cls.__bases__[-1]
        Prediction.class_mapping[state_type] = cls
        super().__init_subclass__(**kwargs)


class MeasurementPrediction(Type):
    """ Prediction type

    This is the base measurement prediction class. """
    class_mapping = {}
    from_state = classmethod(_from_state)

    def __init_subclass__(cls, **kwargs):
        state_type = cls.__bases__[-1]
        MeasurementPrediction.class_mapping[state_type] = cls
        super().__init_subclass__(**kwargs)


class StatePrediction(Prediction, State):
    """ StatePrediction type

    Most simple state prediction type, which only has time and a state vector.
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


class WeightedGaussianStatePrediction(Prediction, TaggedWeightedGaussianState):
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
MeasurementPrediction.class_mapping[SqrtGaussianState] = GaussianMeasurementPrediction


class ParticleStatePrediction(Prediction, ParticleState):
    """ParticleStatePrediction type

    This is a simple Particle state prediction object.
    """


class ParticleMeasurementPrediction(MeasurementPrediction, ParticleState):
    """MeasurementStatePrediction type

    This is a simple Particle measurement prediction object.
    """
