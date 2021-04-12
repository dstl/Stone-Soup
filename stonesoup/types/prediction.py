# -*- coding: utf-8 -*-
from typing import Any, Union

from .array import CovarianceMatrix
from .base import Type
from .state import (State, GaussianState, ParticleState, SqrtGaussianState,
                    TaggedWeightedGaussianState, WeightedGaussianState, StateMutableSequence,
                    CompositeState)
from ..base import Property
from ..models.transition.base import TransitionModel


def _from_state(
        cls,
        state: State,
        *args: Any,
        prediction_type: Union['Prediction', 'MeasurementPrediction', None] = None,
        **kwargs: Any) -> Union['Prediction', 'MeasurementPrediction']:
    """Return new (Measurement)Prediction instance of suitable type using existing properties

    Parameters
    ----------
    state: State
        :class:`~.State` to use existing properties from, and identify prediction type from
    \\*args: Sequence
        Arguments to pass to newly created prediction, replacing those with same name on ``state``
        parameter.
    prediction_type: :class:`~.Prediction` or :class:`~.MeasurementPrediction`, optional
        Type to use for prediction, overriding one from :attr:`class_mapping`.
    \\*\\*kwargs: Mapping
        New property names and associate value for use in newly created prediction, replacing those
        on the ``state`` parameter.
    """
    # Handle being initialised with state sequence
    if isinstance(state, StateMutableSequence):
        state = state.state
    try:
        state_type = next(type_ for type_ in type(state).mro() if type_ in cls.class_mapping)
    except StopIteration:
        raise TypeError(f'{cls.__name__} type not defined for {type(state).__name__}')
    if prediction_type is None:
        prediction_type = cls.class_mapping[state_type]

    args_property_names = {
        name for n, name in enumerate(prediction_type.properties) if n < len(args)}
    # Use current state kwargs that also properties of prediction type
    new_kwargs = {
        name: getattr(state, name)
        for name in state_type.properties.keys() & prediction_type.properties.keys()
        if name not in args_property_names}
    # And replace them with any newly defined kwargs
    new_kwargs.update(kwargs)

    return prediction_type(*args, **new_kwargs)


class Prediction(Type):
    """ Prediction type

    This is the base prediction class. """
    class_mapping = {}
    from_state = classmethod(_from_state)
    transition_model: TransitionModel = Property(
        default=None, doc='The transition model used to make the prediction')

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
MeasurementPrediction.class_mapping[SqrtGaussianState] = GaussianMeasurementPrediction


class ParticleStatePrediction(Prediction, ParticleState):
    """ParticleStatePrediction type

    This is a simple Particle state prediction object.
    """


class ParticleMeasurementPrediction(MeasurementPrediction, ParticleState):
    """MeasurementStatePrediction type

    This is a simple Particle measurement prediction object.
    """


class CompositePrediction(Prediction, CompositeState):
    pass


class CompositeMeasurementPrediction(MeasurementPrediction, CompositeState):
    pass
