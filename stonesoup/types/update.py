# -*- coding: utf-8 -*-
from ..base import Property
from .base import Type
from .hypothesis import Hypothesis
from .state import (State, GaussianState, ParticleState,
                    WeightedGaussianState, GaussianMixtureState)


class Update(Type):
    """ Update type

    The base update class. Updates are returned by :class:'~.Updater' objects
    and contain the information that was used to perform the updating"""

    hypothesis = Property(Hypothesis,
                          doc="Hypothesis used for updating")


class StateUpdate(Update, State):
    """ StateUpdate type

    Most simple state update type, where everything only has time
    and a state vector. Requires a prior state that was updated,
    and the hypothesis used to update the prior.
    """


class GaussianStateUpdate(Update, GaussianState):
    """ GaussianStateUpdate type

    This is a simple Gaussian state update object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class ParticleStateUpdate(Update, ParticleState):
    """ParticleStateUpdate type

    This is a simple Particle state update object.
    """


class WeightedGaussianStateUpdate(Update, WeightedGaussianState):
    """ WeightedGaussianStateUpdate type

    This is an Gaussian state update object, which is augmented with a
    weight  property.
    """


class GaussianMixtureStateUpdate(Update, GaussianMixtureState):
    """ GaussianMixtureStateUpdate type

    This is GaussianMixtureStateUpdate type, which is can be views as a
    wrapper around a collection of WeightedGaussianStateUpdate objects.
    """

    def __init__(self, components, *args, **kwargs):
        super().__init__(components, *args, **kwargs)
        if any([ not isinstance(component, Update)
                 for component in components]):
            raise TypeError("All components must be subclasses of Update")